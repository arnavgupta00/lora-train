#!/usr/bin/env python3
"""
Contamination Module

Centralized logic for:
- Contamination source tagging
- Pool routing (clean vs internal vs rejected)
- Contamination reporting
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .metadata import ContaminationRisk, ContaminationSource, Pool, ExampleMetadata


class RoutingDecision(str, Enum):
    """Routing decision for an example."""
    
    CLEAN = "clean"
    INTERNAL = "internal"
    REJECTED = "rejected"


@dataclass
class RoutingResult:
    """Result of contamination routing."""
    
    decision: RoutingDecision
    contamination_source: ContaminationSource
    reasons: List[str] = field(default_factory=list)


@dataclass
class ContaminationStats:
    """Statistics about contamination routing."""
    
    total_processed: int = 0
    routed_clean: int = 0
    routed_internal: int = 0
    routed_rejected: int = 0
    
    # By source
    bird_dev_direct: int = 0
    bird_gold_reference: int = 0
    bird_prediction_broken: int = 0
    non_benchmark: int = 0
    transformed_archetype: int = 0
    
    # By rejection reason
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "routed_clean": self.routed_clean,
            "routed_internal": self.routed_internal,
            "routed_rejected": self.routed_rejected,
            "clean_percentage": round(100 * self.routed_clean / max(1, self.total_processed), 2),
            "internal_percentage": round(100 * self.routed_internal / max(1, self.total_processed), 2),
            "rejected_percentage": round(100 * self.routed_rejected / max(1, self.total_processed), 2),
            "by_source": {
                "bird_dev_direct": self.bird_dev_direct,
                "bird_gold_reference": self.bird_gold_reference,
                "bird_prediction_broken": self.bird_prediction_broken,
                "non_benchmark": self.non_benchmark,
                "transformed_archetype": self.transformed_archetype,
            },
            "rejection_reasons": self.rejection_reasons,
        }


# Known BIRD dev database IDs
BIRD_DEV_DB_IDS: Set[str] = {
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
}


# Known non-benchmark sources (safe for clean pool)
NON_BENCHMARK_SOURCES: Set[str] = {
    "bird_train",
    "spider",
    "spider_train",
    "wikisql",
    "custom",
    "synthetic_schema",
}


def is_bird_dev_database(db_id: str) -> bool:
    """Check if database is a BIRD dev database."""
    return db_id.lower() in {d.lower() for d in BIRD_DEV_DB_IDS}


def tag_contamination_source(
    source_type: str,
    source_run: Optional[str],
    db_id: str,
    reference_sql_source: str,
    broken_sql_source: Optional[str] = None,
    parent_source: Optional[str] = None,
) -> ContaminationSource:
    """
    Determine the contamination source for an example.
    
    Args:
        source_type: Type of source (real_failure, synthetic_corruption, etc.)
        source_run: Run identifier (e.g., "t10_baseline_3090")
        db_id: Database identifier
        reference_sql_source: Source of reference SQL ("gold", "synthetic_parent", etc.)
        broken_sql_source: Source of broken SQL (optional)
        parent_source: Source of parent SQL for synthetic examples
    
    Returns:
        ContaminationSource enum value
    """
    # Real failures from BIRD dev/eval are always internal-only
    if source_type == "real_failure":
        if source_run and ("t10" in source_run.lower() or "t11" in source_run.lower()):
            return ContaminationSource.BIRD_DEV_DIRECT
        if is_bird_dev_database(db_id):
            return ContaminationSource.BIRD_DEV_DIRECT
    
    # Check reference SQL source
    if reference_sql_source == "gold" and is_bird_dev_database(db_id):
        return ContaminationSource.BIRD_GOLD_REFERENCE
    
    # Check broken SQL source
    if broken_sql_source and "prediction" in broken_sql_source.lower():
        if is_bird_dev_database(db_id):
            return ContaminationSource.BIRD_PREDICTION_BROKEN
    
    # Synthetic examples - check parent source
    if source_type == "synthetic_corruption":
        if parent_source in NON_BENCHMARK_SOURCES:
            return ContaminationSource.NON_BENCHMARK
        if is_bird_dev_database(db_id):
            # Using BIRD dev gold as parent → internal only
            return ContaminationSource.BIRD_GOLD_REFERENCE
    
    # Contrastive and manual examples from non-benchmark sources
    if source_type in ("contrastive", "manual_fix", "deterministic_fix"):
        if parent_source in NON_BENCHMARK_SOURCES:
            return ContaminationSource.NON_BENCHMARK
        if not is_bird_dev_database(db_id):
            return ContaminationSource.NON_BENCHMARK
    
    # Default: if we can't determine, be conservative
    if is_bird_dev_database(db_id):
        return ContaminationSource.BIRD_DEV_DIRECT
    
    return ContaminationSource.NON_BENCHMARK


def is_sufficiently_transformed(
    original_question: str,
    transformed_question: str,
    original_schema: str,
    transformed_schema: str,
) -> bool:
    """
    Check if a transformed example is sufficiently different from original.
    
    For an example to qualify as "transformed_archetype":
    1. Schema surface must be changed (different table/column names)
    2. Identifiers must be changed (different values, literals)
    3. Question phrasing must be materially different
    4. Cannot be reverse-engineered to original
    """
    # Basic length check
    if len(transformed_question) < 10 or len(transformed_schema) < 50:
        return False
    
    # Check question similarity (should be low)
    original_words = set(original_question.lower().split())
    transformed_words = set(transformed_question.lower().split())
    
    if len(original_words) > 0:
        overlap = len(original_words & transformed_words) / len(original_words)
        if overlap > 0.6:  # More than 60% word overlap = too similar
            return False
    
    # Check schema similarity (should be low)
    original_tables = set(extract_table_names(original_schema))
    transformed_tables = set(extract_table_names(transformed_schema))
    
    if len(original_tables) > 0:
        table_overlap = len(original_tables & transformed_tables) / len(original_tables)
        if table_overlap > 0.3:  # More than 30% table name overlap = too similar
            return False
    
    return True


def extract_table_names(schema_text: str) -> List[str]:
    """Extract table names from schema DDL."""
    import re
    tables = []
    for match in re.finditer(r'CREATE\s+TABLE\s+`?(\w+)`?', schema_text, re.IGNORECASE):
        tables.append(match.group(1).lower())
    return tables


def route_example(
    contamination_source: ContaminationSource,
    validation_passed: bool,
    reference_sql_valid: bool,
    has_required_fields: bool,
) -> RoutingResult:
    """
    Route an example to the appropriate pool.
    
    Args:
        contamination_source: Tagged contamination source
        validation_passed: Whether the example passed verification
        reference_sql_valid: Whether reference SQL is valid
        has_required_fields: Whether all required metadata is present
    
    Returns:
        RoutingResult with decision and reasons
    """
    reasons = []
    
    # Check rejection conditions first
    if not validation_passed:
        reasons.append("Validation failed")
        return RoutingResult(
            decision=RoutingDecision.REJECTED,
            contamination_source=contamination_source,
            reasons=reasons,
        )
    
    if not reference_sql_valid:
        reasons.append("Reference SQL invalid or missing")
        return RoutingResult(
            decision=RoutingDecision.REJECTED,
            contamination_source=contamination_source,
            reasons=reasons,
        )
    
    if not has_required_fields:
        reasons.append("Missing required metadata fields")
        return RoutingResult(
            decision=RoutingDecision.REJECTED,
            contamination_source=contamination_source,
            reasons=reasons,
        )
    
    # Route based on contamination source
    if contamination_source in (
        ContaminationSource.NON_BENCHMARK,
        ContaminationSource.TRANSFORMED_ARCHETYPE,
    ):
        reasons.append(f"Clean source: {contamination_source.value}")
        return RoutingResult(
            decision=RoutingDecision.CLEAN,
            contamination_source=contamination_source,
            reasons=reasons,
        )
    else:
        reasons.append(f"Benchmark-derived: {contamination_source.value}")
        return RoutingResult(
            decision=RoutingDecision.INTERNAL,
            contamination_source=contamination_source,
            reasons=reasons,
        )


class ContaminationRouter:
    """Router for contamination-based pool assignment."""
    
    def __init__(self):
        self.stats = ContaminationStats()
        self._routing_log: List[Dict[str, Any]] = []
    
    def tag_and_route(
        self,
        example_id: str,
        source_type: str,
        source_run: Optional[str],
        db_id: str,
        reference_sql_source: str,
        validation_passed: bool,
        reference_sql_valid: bool,
        has_required_fields: bool,
        broken_sql_source: Optional[str] = None,
        parent_source: Optional[str] = None,
    ) -> RoutingResult:
        """
        Tag contamination source and route example.
        
        Returns:
            RoutingResult with decision and metadata
        """
        # Tag source
        contamination_source = tag_contamination_source(
            source_type=source_type,
            source_run=source_run,
            db_id=db_id,
            reference_sql_source=reference_sql_source,
            broken_sql_source=broken_sql_source,
            parent_source=parent_source,
        )
        
        # Route
        result = route_example(
            contamination_source=contamination_source,
            validation_passed=validation_passed,
            reference_sql_valid=reference_sql_valid,
            has_required_fields=has_required_fields,
        )
        
        # Update stats
        self.stats.total_processed += 1
        
        if result.decision == RoutingDecision.CLEAN:
            self.stats.routed_clean += 1
        elif result.decision == RoutingDecision.INTERNAL:
            self.stats.routed_internal += 1
        else:
            self.stats.routed_rejected += 1
            for reason in result.reasons:
                self.stats.rejection_reasons[reason] = \
                    self.stats.rejection_reasons.get(reason, 0) + 1
        
        # Update source counts
        if contamination_source == ContaminationSource.BIRD_DEV_DIRECT:
            self.stats.bird_dev_direct += 1
        elif contamination_source == ContaminationSource.BIRD_GOLD_REFERENCE:
            self.stats.bird_gold_reference += 1
        elif contamination_source == ContaminationSource.BIRD_PREDICTION_BROKEN:
            self.stats.bird_prediction_broken += 1
        elif contamination_source == ContaminationSource.NON_BENCHMARK:
            self.stats.non_benchmark += 1
        elif contamination_source == ContaminationSource.TRANSFORMED_ARCHETYPE:
            self.stats.transformed_archetype += 1
        
        # Log routing decision
        self._routing_log.append({
            "example_id": example_id,
            "contamination_source": contamination_source.value,
            "decision": result.decision.value,
            "reasons": result.reasons,
        })
        
        return result
    
    def update_metadata(
        self,
        metadata: ExampleMetadata,
        routing_result: RoutingResult,
    ) -> ExampleMetadata:
        """Update example metadata with routing result."""
        metadata.contamination_source = routing_result.contamination_source.value
        metadata.pool = routing_result.decision.value
        metadata.benchmark_clean = routing_result.decision == RoutingDecision.CLEAN
        if routing_result.contamination_source == ContaminationSource.NON_BENCHMARK:
            metadata.contamination_risk = ContaminationRisk.NONE.value
        elif routing_result.contamination_source == ContaminationSource.TRANSFORMED_ARCHETYPE:
            metadata.contamination_risk = ContaminationRisk.LOW.value
        else:
            metadata.contamination_risk = ContaminationRisk.HIGH.value
        return metadata
    
    def get_stats(self) -> ContaminationStats:
        """Get routing statistics."""
        return self.stats
    
    def get_routing_log(self) -> List[Dict[str, Any]]:
        """Get full routing log."""
        return self._routing_log
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate contamination report."""
        return {
            "summary": self.stats.to_dict(),
            "routing_decisions": self._routing_log[:100],  # First 100 for sample
            "total_logged": len(self._routing_log),
        }
