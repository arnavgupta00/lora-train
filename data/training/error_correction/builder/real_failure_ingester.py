#!/usr/bin/env python3
"""
Real Failure Ingester Module

Parses evaluation results and builds repair contexts from real failures.
T10 is the primary source; T11_1 is auxiliary.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from .config import BuilderConfig, PathConfig
from .metadata import (
    ContaminationSource,
    ExampleMetadata,
    Pool,
    RepairExample,
    SourceType,
)
from .schema_builder import SchemaBuilder
from .taxonomy import (
    FailureType,
    TargetFailureMode,
    get_sampling_weight,
    get_suggested_target_mode,
    infer_failure_type,
)


@dataclass
class RealFailure:
    """A real failure from evaluation."""
    
    # Identity
    question_id: int
    db_id: str
    
    # Content
    question: str
    hints: str
    schema_ddl: str
    gold_sql: str
    predicted_sql: str
    
    # Failure info
    exec_failed: bool
    wrong_result: bool
    error_message: Optional[str]
    error_category: Optional[str]
    difficulty: str
    
    # Source
    source: str  # "t10" or "t11_1"
    source_run: str


@dataclass
class IngestionStats:
    """Statistics from ingestion."""
    
    total_loaded: int = 0
    failures_found: int = 0
    exec_failed_count: int = 0
    wrong_result_count: int = 0
    skipped_no_gold: int = 0
    skipped_duplicate: int = 0
    
    by_db: Dict[str, int] = field(default_factory=dict)
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loaded": self.total_loaded,
            "failures_found": self.failures_found,
            "exec_failed_count": self.exec_failed_count,
            "wrong_result_count": self.wrong_result_count,
            "skipped_no_gold": self.skipped_no_gold,
            "skipped_duplicate": self.skipped_duplicate,
            "by_db": self.by_db,
            "by_difficulty": self.by_difficulty,
            "by_source": self.by_source,
        }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class RealFailureIngester:
    """Ingester for real failures from evaluation results."""
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.paths = config.paths
        self.schema_builder = SchemaBuilder(self.paths, config.schema)
        self.stats = IngestionStats()
        
        # Cache for loaded data
        self._eval_results: Dict[str, List[Dict]] = {}
        self._bird_dev: Dict[str, Dict[int, Dict]] = {}
        self._predictions: Dict[str, Dict[int, Dict]] = {}
        self._seen_ids: Set[Tuple[int, str]] = set()
    
    def _load_source_data(self, source: str) -> None:
        """Load all data for a source (t10 or t11_1)."""
        if source in self._eval_results:
            return
        
        # Load per-example results
        results_path = self.paths.per_example_results(source)
        if results_path.exists():
            self._eval_results[source] = load_jsonl(results_path)
        else:
            self._eval_results[source] = []
        
        # Load bird dev prompts (for question, hints, schema, gold)
        dev_path = self.paths.bird_dev_prompts(source)
        if dev_path.exists():
            dev_data = load_jsonl(dev_path)
            self._bird_dev[source] = {d['question_id']: d for d in dev_data}
        else:
            self._bird_dev[source] = {}
        
        # Load predictions (for broken SQL)
        pred_path = self.paths.predictions_file(source)
        if pred_path.exists():
            pred_data = load_jsonl(pred_path)
            self._predictions[source] = {p.get('question_id', i): p for i, p in enumerate(pred_data)}
        else:
            self._predictions[source] = {}
    
    def _extract_schema_from_prompt(self, prompt_data: Dict) -> str:
        """Extract schema DDL from prompt data."""
        # Try to get from t10_prompt
        if 't10_prompt' in prompt_data:
            user_content = prompt_data['t10_prompt'].get('user', '')
            # Extract schema section
            if 'Schema:' in user_content:
                parts = user_content.split('Schema:')
                if len(parts) > 1:
                    schema_part = parts[1]
                    # Find end of schema (before Hints: or Question:)
                    for marker in ['Hints:', 'Question:']:
                        if marker in schema_part:
                            schema_part = schema_part.split(marker)[0]
                    return schema_part.strip()
        
        # Fallback: load from database
        return ""
    
    def _extract_hints_from_prompt(self, prompt_data: Dict) -> str:
        """Extract hints from prompt data."""
        # Direct evidence field
        if 'evidence' in prompt_data:
            return prompt_data['evidence'] or ""
        
        # Try from t10_prompt
        if 't10_prompt' in prompt_data:
            user_content = prompt_data['t10_prompt'].get('user', '')
            if 'Hints:' in user_content:
                parts = user_content.split('Hints:')
                if len(parts) > 1:
                    hints_part = parts[1]
                    if 'Question:' in hints_part:
                        hints_part = hints_part.split('Question:')[0]
                    return hints_part.strip()
        
        return ""
    
    def _build_failure(
        self,
        eval_result: Dict,
        source: str,
    ) -> Optional[RealFailure]:
        """Build a RealFailure from eval result."""
        question_id = eval_result.get('question_id')
        db_id = eval_result.get('db_id', '')
        
        if question_id is None:
            return None
        
        # Check for duplicate
        key = (question_id, db_id)
        if key in self._seen_ids:
            self.stats.skipped_duplicate += 1
            return None
        
        # Get dev prompt data
        dev_data = self._bird_dev.get(source, {}).get(question_id, {})
        if not dev_data:
            return None
        
        # Get prediction data
        pred_data = self._predictions.get(source, {}).get(question_id, {})
        
        # Extract fields
        question = dev_data.get('question', '')
        gold_sql = dev_data.get('gold_sql', eval_result.get('gold_sql', ''))
        
        if not gold_sql:
            self.stats.skipped_no_gold += 1
            return None
        
        # Get predicted SQL from eval result or predictions
        predicted_sql = eval_result.get('predicted_sql', '')
        if not predicted_sql and pred_data:
            predicted_sql = pred_data.get('predicted_sql', pred_data.get('sql', ''))
        
        if not predicted_sql:
            return None
        
        # Extract schema and hints
        schema_ddl = self._extract_schema_from_prompt(dev_data)
        hints = self._extract_hints_from_prompt(dev_data)
        
        # Mark as seen
        self._seen_ids.add(key)
        
        # Build source_run identifier
        source_run = f"{source}_baseline_3090"
        
        return RealFailure(
            question_id=question_id,
            db_id=db_id,
            question=question,
            hints=hints,
            schema_ddl=schema_ddl,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            exec_failed=eval_result.get('exec_failed', False),
            wrong_result=eval_result.get('wrong_result', False),
            error_message=eval_result.get('pred_error'),
            error_category=eval_result.get('error_category'),
            difficulty=eval_result.get('difficulty', 'simple'),
            source=source,
            source_run=source_run,
        )
    
    def iter_failures(
        self,
        include_t11_1: bool = True,
        limit: Optional[int] = None,
    ) -> Generator[RealFailure, None, None]:
        """
        Iterate over real failures.
        
        T10 is primary, T11_1 is auxiliary (only used for supplementary coverage).
        
        Args:
            include_t11_1: Whether to include T11_1 failures
            limit: Maximum number of failures to yield
        
        Yields:
            RealFailure objects
        """
        count = 0
        
        # Load T10 (primary source)
        self._load_source_data("t10")
        
        for eval_result in self._eval_results.get("t10", []):
            self.stats.total_loaded += 1
            
            # Only failures
            if not eval_result.get('exec_failed') and not eval_result.get('wrong_result'):
                continue
            
            failure = self._build_failure(eval_result, "t10")
            if failure:
                self.stats.failures_found += 1
                self.stats.by_source["t10"] = self.stats.by_source.get("t10", 0) + 1
                self.stats.by_db[failure.db_id] = self.stats.by_db.get(failure.db_id, 0) + 1
                self.stats.by_difficulty[failure.difficulty] = \
                    self.stats.by_difficulty.get(failure.difficulty, 0) + 1
                
                if failure.exec_failed:
                    self.stats.exec_failed_count += 1
                if failure.wrong_result:
                    self.stats.wrong_result_count += 1
                
                yield failure
                count += 1
                
                if limit and count >= limit:
                    return
        
        # Load T11_1 (auxiliary) if enabled
        if include_t11_1:
            self._load_source_data("t11_1")
            
            for eval_result in self._eval_results.get("t11_1", []):
                self.stats.total_loaded += 1
                
                if not eval_result.get('exec_failed') and not eval_result.get('wrong_result'):
                    continue
                
                failure = self._build_failure(eval_result, "t11_1")
                if failure:
                    self.stats.failures_found += 1
                    self.stats.by_source["t11_1"] = self.stats.by_source.get("t11_1", 0) + 1
                    self.stats.by_db[failure.db_id] = self.stats.by_db.get(failure.db_id, 0) + 1
                    self.stats.by_difficulty[failure.difficulty] = \
                        self.stats.by_difficulty.get(failure.difficulty, 0) + 1
                    
                    if failure.exec_failed:
                        self.stats.exec_failed_count += 1
                    if failure.wrong_result:
                        self.stats.wrong_result_count += 1
                    
                    yield failure
                    count += 1
                    
                    if limit and count >= limit:
                        return
    
    def build_repair_context(
        self,
        failure: RealFailure,
        use_compact_schema: bool = True,
    ) -> RepairExample:
        """
        Build a RepairExample from a real failure.
        
        Note: corrected_sql is left empty - must be filled by subagent.
        """
        # Build schema context
        if failure.schema_ddl and not use_compact_schema:
            schema_context = failure.schema_ddl
            schema_meta = {
                "schema_context_type": "full_schema",
                "tables_kept": [],
                "columns_kept": 0,
            }
        else:
            schema_context, schema_meta = self.schema_builder.build_context(
                db_id=failure.db_id,
                question=failure.question,
                hints=failure.hints,
                broken_sql=failure.predicted_sql,
                use_compact=use_compact_schema,
            )
        
        # Infer failure type
        failure_type = infer_failure_type(
            exec_failed=failure.exec_failed,
            wrong_result=failure.wrong_result,
            error_category=failure.error_category,
            error_message=failure.error_message,
            broken_sql=failure.predicted_sql,
            gold_sql=failure.gold_sql,
        )
        
        target_mode = get_suggested_target_mode(failure_type)
        
        # Calculate sampling weight
        sampling_weight = get_sampling_weight(
            failure_type=failure_type,
            db_family=failure.db_id,
            source_type="real_failure",
        )
        
        # Build example ID
        example_id = f"real_{failure.source}_{failure.db_id}_{failure.question_id}"
        
        # Build metadata
        metadata = ExampleMetadata(
            example_id=example_id,
            split="train",  # Will be assigned later
            source_type=SourceType.REAL_FAILURE.value,
            source_run=failure.source_run,
            original_question_id=failure.question_id,
            db_family=failure.db_id,
            schema_context_type=schema_meta.get("schema_context_type", "compact_relevant"),
            schema_tables_kept=schema_meta.get("tables_kept", []),
            schema_columns_kept=schema_meta.get("columns_kept", 0),
            failure_type=failure_type.value,
            target_failure_mode=target_mode.value,
            exec_failed_originally=failure.exec_failed,
            error_message=failure.error_message,
            contamination_source=ContaminationSource.BIRD_DEV_DIRECT.value,
            benchmark_clean=False,
            pool=Pool.INTERNAL.value,
            difficulty=failure.difficulty,
            sampling_weight=sampling_weight,
            corrected_sql_source="subagent_validated",  # Will be filled
            reference_sql_source="gold",
            reference_sql=failure.gold_sql,
            verification_method="execution_match",
            verification_passed=False,  # Not verified yet
            subagent_used=True,  # Will need subagent
        )
        
        # Build repair example
        example = RepairExample(
            schema_context=schema_context,
            question=failure.question,
            hints=failure.hints,
            broken_sql=failure.predicted_sql,
            error=failure.error_message,
            failure_type_hint=failure_type.value if failure.exec_failed else None,
            corrected_sql="",  # To be filled by subagent
            metadata=metadata,
        )
        
        return example
    
    def get_stats(self) -> IngestionStats:
        """Get ingestion statistics."""
        return self.stats


def load_all_failures(
    config: BuilderConfig,
    limit: Optional[int] = None,
) -> List[RealFailure]:
    """Convenience function to load all failures."""
    ingester = RealFailureIngester(config)
    return list(ingester.iter_failures(
        include_t11_1=config.include_t11_1,
        limit=limit,
    ))
