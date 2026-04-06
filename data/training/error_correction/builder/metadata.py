#!/usr/bin/env python3
"""
Metadata Module

Defines the metadata schema for error-correction examples.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ContaminationSource(str, Enum):
    """Contamination source categories."""
    
    # BIRD dev/eval derived - always internal-only
    BIRD_DEV_DIRECT = "bird_dev_direct"
    BIRD_GOLD_REFERENCE = "bird_gold_reference"
    BIRD_PREDICTION_BROKEN = "bird_prediction_broken"
    
    # Clean sources - eligible for clean pool
    NON_BENCHMARK = "non_benchmark"
    TRANSFORMED_ARCHETYPE = "transformed_archetype"


class Pool(str, Enum):
    """Dataset pool assignment."""
    
    INTERNAL = "internal"  # Full dataset, may contain benchmark-derived
    CLEAN = "clean"  # Benchmark-safe, no BIRD dev/eval derived
    REJECTED = "rejected"  # Failed validation or uncertain lineage


class SourceType(str, Enum):
    """Source type for the example."""
    
    REAL_FAILURE = "real_failure"
    SYNTHETIC_CORRUPTION = "synthetic_corruption"
    CONTRASTIVE = "contrastive"
    MANUAL_FIX = "manual_fix"
    DETERMINISTIC_FIX = "deterministic_fix"


class CorrectedSqlSource(str, Enum):
    """Source of the corrected SQL."""
    
    GOLD_ALIGNED = "gold_aligned"  # From gold/reference SQL
    SYNTHETIC_PARENT = "synthetic_parent"  # Parent SQL for synthetic corruption
    SUBAGENT_VALIDATED = "subagent_validated"  # Subagent proposal that passed validation
    MANUAL = "manual"  # Manually curated
    DETERMINISTIC = "deterministic"  # Deterministic repair


class ReferenceSqlSource(str, Enum):
    """Source of the reference SQL for validation."""
    
    GOLD = "gold"  # From benchmark gold SQL
    SYNTHETIC_PARENT = "synthetic_parent"  # Parent SQL (for synthetic)
    MANUAL_CURATED = "manual_curated"  # Manually provided reference


class VerificationMethod(str, Enum):
    """Method used to verify corrected SQL."""
    
    EXECUTION_MATCH = "execution_match"
    NORMALIZED_MATCH = "normalized_match"
    BOTH = "both"


class ContaminationRisk(str, Enum):
    """Benchmark contamination risk level."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class SchemaContextType(str, Enum):
    """Type of schema context provided."""
    
    COMPACT_RELEVANT = "compact_relevant"
    FULL_SCHEMA = "full_schema"
    WIDER_SCHEMA = "wider_schema"


@dataclass
class SubagentMetadata:
    """Metadata about subagent usage for this example."""
    
    used: bool = False
    model: Optional[str] = None
    retries: int = 0
    candidate_sql: Optional[str] = None
    accept_reason: Optional[str] = None
    reject_reason: Optional[str] = None


@dataclass
class SchemaMetadata:
    """Metadata about schema context."""
    
    context_type: str = "compact_relevant"
    tables_kept: List[str] = field(default_factory=list)
    columns_kept: int = 0
    relations: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class CorruptionMetadata:
    """Metadata for synthetic corruption examples."""
    
    transform: Optional[str] = None
    family: Optional[str] = None
    severity: Optional[str] = None  # "minor", "moderate", "major"
    parent_sql: Optional[str] = None


@dataclass
class ExampleMetadata:
    """Complete metadata for an error-correction example."""
    
    # Identity
    example_id: str = ""
    split: str = "train"  # "train" or "dev"
    
    # Source information
    source_type: str = "real_failure"
    source_run: Optional[str] = None
    original_question_id: Optional[int] = None
    original_example_id: Optional[str] = None
    
    # Database and schema
    db_family: str = ""
    schema_context_type: str = "compact_relevant"
    schema_tables_kept: List[str] = field(default_factory=list)
    schema_columns_kept: int = 0
    relations: List[str] = field(default_factory=list)
    schema_notes: Optional[str] = None
    notes: Optional[str] = None
    compact_schema_stats: Optional[Dict[str, Any]] = None
    
    # Failure classification
    failure_type: str = ""
    target_failure_mode: str = ""
    exec_failed_originally: bool = False
    error_message: Optional[str] = None
    
    # Contamination and pool assignment
    contamination_source: str = "bird_dev_direct"
    contamination_risk: str = ContaminationRisk.HIGH.value
    benchmark_clean: bool = False
    pool: str = "internal"
    
    # Difficulty and weighting
    difficulty: str = "simple"  # "simple", "moderate", "challenging"
    sampling_weight: float = 1.0
    
    # Corrected SQL provenance
    corrected_sql_source: str = "subagent_validated"
    reference_sql_source: str = "gold"
    reference_sql: Optional[str] = None  # Stored for verification
    
    # Verification
    verification_method: str = "execution_match"
    verification_passed: bool = False
    
    # Subagent usage
    subagent_used: bool = False
    subagent_model: Optional[str] = None
    subagent_retries: int = 0
    subagent_candidate_sql: Optional[str] = None
    subagent_accept_reason: Optional[str] = None
    subagent_reject_reason: Optional[str] = None
    
    # Corruption metadata (for synthetic)
    corruption_transform: Optional[str] = None
    corruption_family: Optional[str] = None
    corruption_severity: Optional[str] = None
    parent_sql: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values for cleaner output."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != [] and v != "":
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExampleMetadata":
        """Create from dictionary."""
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    def is_clean_eligible(self) -> bool:
        """Check if this example is eligible for the clean pool."""
        return self.contamination_risk != ContaminationRisk.HIGH.value and self.contamination_source in [
            ContaminationSource.NON_BENCHMARK.value,
            ContaminationSource.TRANSFORMED_ARCHETYPE.value,
        ]
    
    def validate(self) -> List[str]:
        """Validate metadata and return list of errors."""
        errors = []
        
        if not self.example_id:
            errors.append("Missing example_id")
        
        if not self.db_family:
            errors.append("Missing db_family")
        
        if not self.failure_type:
            errors.append("Missing failure_type")
        
        if not self.target_failure_mode:
            errors.append("Missing target_failure_mode")
        
        if self.benchmark_clean and self.contamination_source not in [
            ContaminationSource.NON_BENCHMARK.value,
            ContaminationSource.TRANSFORMED_ARCHETYPE.value,
        ]:
            errors.append(
                f"benchmark_clean=True but contamination_source={self.contamination_source}"
            )
        
        if self.pool == Pool.CLEAN.value and not self.benchmark_clean:
            errors.append("pool=clean but benchmark_clean=False")

        if self.pool == Pool.CLEAN.value and self.contamination_risk == ContaminationRisk.HIGH.value:
            errors.append("pool=clean but contamination_risk=high")
        
        return errors


@dataclass
class RepairExample:
    """A complete error-correction example."""
    
    # Content
    schema_context: str = ""
    question: str = ""
    hints: str = ""
    broken_sql: str = ""
    error: Optional[str] = None
    failure_type_hint: Optional[str] = None
    corrected_sql: str = ""
    
    # Metadata
    metadata: ExampleMetadata = field(default_factory=ExampleMetadata)
    
    def to_sft_format(self) -> Dict[str, Any]:
        """Convert to SFT training format with messages."""
        system_prompt = (
            "You are an expert SQL repair assistant. "
            "Given schema, question, hints, broken SQL, and optional database error, "
            "output the corrected SQL query only."
        )
        
        # Build user content
        user_parts = [
            f"Schema:\n{self.schema_context}",
            f"Hints:\n{self.hints}" if self.hints else "Hints:\nNone",
            f"Question:\n{self.question}",
            f"Broken SQL:\n{self.broken_sql}",
        ]
        
        if self.error:
            user_parts.append(f"Error:\n{self.error}")
        
        if self.failure_type_hint:
            user_parts.append(f"Failure Type:\n{self.failure_type_hint}")
        
        user_content = "\n\n".join(user_parts)
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.corrected_sql},
            ],
            "metadata": self.metadata.to_dict(),
        }
    
    def validate(self) -> List[str]:
        """Validate example and return list of errors."""
        errors = self.metadata.validate()
        
        if not self.schema_context:
            errors.append("Missing schema_context")
        
        if not self.question:
            errors.append("Missing question")
        
        if not self.broken_sql:
            errors.append("Missing broken_sql")
        
        if not self.corrected_sql:
            errors.append("Missing corrected_sql")
        
        # Check for prose contamination in corrected SQL
        if self.corrected_sql:
            lower_sql = self.corrected_sql.lower().strip()
            if lower_sql.startswith("the ") or lower_sql.startswith("here "):
                errors.append("corrected_sql appears to contain prose")
            if "```" in self.corrected_sql:
                errors.append("corrected_sql contains code fences")
        
        return errors
