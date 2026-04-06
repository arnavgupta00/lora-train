#!/usr/bin/env python3
"""
Subagent Client Module

Rate-limited client for calling subagents to propose corrected SQL.
Max 2 parallel calls. Retry budget: 1 initial + 1 retry.
"""

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import SubagentConfig
from .metadata import RepairExample


@dataclass
class SubagentProposal:
    """Result of a subagent proposal."""
    
    success: bool
    corrected_sql: str
    model_used: str
    attempts: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class SubagentStats:
    """Statistics about subagent usage."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    total_latency_ms: float = 0.0
    
    by_model: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "success_rate": round(100 * self.successful_calls / max(1, self.total_calls), 2),
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.total_calls), 2),
            "by_model": self.by_model,
            "error_types": self.error_types,
        }


REPAIR_SYSTEM_PROMPT = """You are an expert SQL repair assistant.
Given a database schema, question, hints, broken SQL, and optional error message,
output ONLY the corrected SQL query.

Rules:
- Output ONLY the corrected SQL query
- Do NOT include explanations, markdown, or code fences
- Do NOT include multiple queries or alternatives
- Use the exact table and column names from the schema
- Fix the specific error while preserving correct parts of the query"""


def build_repair_prompt(example: RepairExample) -> str:
    """Build the user prompt for repair."""
    parts = [
        f"Schema:\n{example.schema_context}",
        f"Hints:\n{example.hints}" if example.hints else "Hints:\nNone",
        f"Question:\n{example.question}",
        f"Broken SQL:\n{example.broken_sql}",
    ]
    
    if example.error:
        parts.append(f"Error:\n{example.error}")
    
    if example.failure_type_hint:
        parts.append(f"Failure Type:\n{example.failure_type_hint}")
    
    return "\n\n".join(parts)


def clean_sql_response(response: str) -> str:
    """
    Clean SQL response from subagent.
    
    Removes:
    - Code fences
    - Explanatory text
    - Multiple queries
    """
    if not response:
        return ""
    
    text = response.strip()
    
    # Remove code fences
    if "```" in text:
        # Extract content from code fence
        match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            # Remove all code fences
            text = re.sub(r'```(?:sql)?', '', text, flags=re.IGNORECASE).strip()
    
    # Remove leading explanation patterns
    patterns_to_strip = [
        r'^(?:Here(?:\s+is)?|The|This|Corrected|Fixed)[\s\S]*?:\s*',
        r'^(?:The\s+)?corrected\s+(?:SQL\s+)?query\s*(?:is|:)\s*',
    ]
    
    for pattern in patterns_to_strip:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    
    # Take only the first query if multiple
    # Split on semicolon but be careful about string literals
    if ';' in text:
        # Simple split - take first statement
        parts = text.split(';')
        text = parts[0].strip()
        if text:
            text += ';'  # Add back semicolon if there was content
    
    # Remove trailing semicolon for consistency
    text = text.rstrip(';').strip()
    
    return text


def validate_sql_response(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that response looks like valid SQL.
    
    Returns:
        (is_valid, error_message)
    """
    if not sql:
        return False, "Empty response"
    
    sql_upper = sql.upper().strip()
    
    # Must start with SQL keyword
    if not re.match(r'^(SELECT|WITH|INSERT|UPDATE|DELETE)', sql_upper):
        return False, "Does not start with SQL keyword"
    
    # Check for prose indicators
    if sql_upper.startswith("THE ") or sql_upper.startswith("HERE "):
        return False, "Starts with prose"
    
    # Very short
    if len(sql) < 10:
        return False, "SQL too short"
    
    return True, None


class SubagentClient:
    """
    Client for calling subagents to propose corrected SQL.
    
    This is a mock implementation. In production, this would call
    the actual Claude API.
    """
    
    def __init__(self, config: SubagentConfig):
        self.config = config
        self.stats = SubagentStats()
        self._semaphore = asyncio.Semaphore(config.max_parallel)
        self._call_fn: Optional[Callable] = None
    
    def set_call_function(self, fn: Callable[[str, str, str], str]) -> None:
        """
        Set the function to call for proposals.
        
        Function signature: (system_prompt, user_prompt, model) -> response
        """
        self._call_fn = fn
    
    def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        """
        Call the model. Override this in subclass or use set_call_function.
        """
        if self._call_fn:
            return self._call_fn(system_prompt, user_prompt, model)
        
        # Mock implementation for testing
        raise NotImplementedError(
            "SubagentClient requires a call function. "
            "Use set_call_function() or override _call_model()."
        )
    
    def propose_correction(
        self,
        example: RepairExample,
        retry_with_feedback: bool = True,
    ) -> SubagentProposal:
        """
        Get a corrected SQL proposal for an example.
        
        Args:
            example: The repair example (without corrected_sql filled)
            retry_with_feedback: Whether to retry with error feedback
        
        Returns:
            SubagentProposal with results
        """
        start_time = time.time()
        attempts = 0
        last_error = None
        model = self.config.model
        
        user_prompt = build_repair_prompt(example)
        
        # First attempt
        attempts += 1
        self.stats.total_calls += 1
        self.stats.by_model[model] = self.stats.by_model.get(model, 0) + 1
        
        try:
            response = self._call_model(REPAIR_SYSTEM_PROMPT, user_prompt, model)
            corrected_sql = clean_sql_response(response)
            
            is_valid, error = validate_sql_response(corrected_sql)
            
            if is_valid:
                latency_ms = (time.time() - start_time) * 1000
                self.stats.successful_calls += 1
                self.stats.total_latency_ms += latency_ms
                
                return SubagentProposal(
                    success=True,
                    corrected_sql=corrected_sql,
                    model_used=model,
                    attempts=attempts,
                    latency_ms=latency_ms,
                )
            else:
                last_error = error
        except Exception as e:
            last_error = str(e)
            self.stats.error_types[type(e).__name__] = \
                self.stats.error_types.get(type(e).__name__, 0) + 1
        
        # Retry if enabled
        if retry_with_feedback and attempts < (self.config.initial_attempts + self.config.retry_attempts):
            self.stats.retried_calls += 1
            attempts += 1
            
            # Add feedback to prompt
            feedback_prompt = user_prompt + f"\n\nPrevious attempt failed: {last_error}\nPlease try again, outputting ONLY the corrected SQL."
            
            try:
                # Try with fallback model
                retry_model = self.config.fallback_model if last_error else model
                self.stats.by_model[retry_model] = self.stats.by_model.get(retry_model, 0) + 1
                
                response = self._call_model(REPAIR_SYSTEM_PROMPT, feedback_prompt, retry_model)
                corrected_sql = clean_sql_response(response)
                
                is_valid, error = validate_sql_response(corrected_sql)
                
                if is_valid:
                    latency_ms = (time.time() - start_time) * 1000
                    self.stats.successful_calls += 1
                    self.stats.total_latency_ms += latency_ms
                    
                    return SubagentProposal(
                        success=True,
                        corrected_sql=corrected_sql,
                        model_used=retry_model,
                        attempts=attempts,
                        latency_ms=latency_ms,
                    )
                else:
                    last_error = error
            except Exception as e:
                last_error = str(e)
                self.stats.error_types[type(e).__name__] = \
                    self.stats.error_types.get(type(e).__name__, 0) + 1
        
        # Failed after all attempts
        latency_ms = (time.time() - start_time) * 1000
        self.stats.failed_calls += 1
        self.stats.total_latency_ms += latency_ms
        
        return SubagentProposal(
            success=False,
            corrected_sql="",
            model_used=model,
            attempts=attempts,
            latency_ms=latency_ms,
            error=last_error,
        )
    
    def propose_corrections_batch(
        self,
        examples: List[RepairExample],
        max_parallel: Optional[int] = None,
    ) -> List[Tuple[RepairExample, SubagentProposal]]:
        """
        Get proposals for a batch of examples.
        
        Uses thread pool with rate limiting.
        """
        max_workers = min(
            max_parallel or self.config.max_parallel,
            self.config.max_parallel,
        )
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.propose_correction, ex): ex
                for ex in examples
            }
            
            for future in as_completed(futures):
                example = futures[future]
                try:
                    proposal = future.result()
                    results.append((example, proposal))
                except Exception as e:
                    results.append((example, SubagentProposal(
                        success=False,
                        corrected_sql="",
                        model_used=self.config.model,
                        attempts=1,
                        latency_ms=0,
                        error=str(e),
                    )))
        
        return results
    
    def get_stats(self) -> SubagentStats:
        """Get subagent usage statistics."""
        return self.stats


class MockSubagentClient(SubagentClient):
    """Mock client for testing that returns gold SQL as correction."""
    
    def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        """Return a mock response based on reference SQL in metadata."""
        # This is just for testing - extracts gold SQL from the prompt context
        # In real usage, the subagent would generate the correction
        
        # Simulate some latency
        time.sleep(0.1)
        
        # Return a mock SELECT statement
        return "SELECT mock_column FROM mock_table"
