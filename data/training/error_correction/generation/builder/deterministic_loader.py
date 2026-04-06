#!/usr/bin/env python3
"""
Deterministic Repair Loader

Loads high-confidence repaired examples from prior repair logs and converts them
into validated deterministic-fix examples.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import BuilderConfig
from .metadata import RepairExample, SourceType
from .real_failure_ingester import RealFailure, RealFailureIngester
from .taxonomy import FailureType, get_sampling_weight, get_suggested_target_mode
from .verifier import Verifier


@dataclass
class DeterministicStats:
    """Statistics for deterministic repair ingestion."""

    total_seen: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    by_failure_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_seen": self.total_seen,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "by_failure_type": dict(self.by_failure_type),
        }


class DeterministicRepairLoader:
    """Load deterministic fixes from existing repair logs."""

    def __init__(self, config: BuilderConfig, verifier: Optional[Verifier] = None):
        self.config = config
        self.verifier = verifier or Verifier(config.paths)
        self.stats = DeterministicStats()

    def _build_failure_index(self) -> Dict[Tuple[str, int], RealFailure]:
        ingester = RealFailureIngester(self.config)
        index: Dict[Tuple[str, int], RealFailure] = {}
        for failure in ingester.iter_failures(
            include_t11_1=self.config.include_t11_1,
            limit=None,
        ):
            index[(failure.db_id, failure.question_id)] = failure
        return index

    def load_examples(
        self,
        limit: Optional[int] = None,
    ) -> List[RepairExample]:
        """Load, validate, and return deterministic-fix examples."""
        repair_log = self.config.paths.repair_log_file("t10")
        if not repair_log.exists():
            return []

        failures = self._build_failure_index()
        ingester = RealFailureIngester(self.config)
        results: List[RepairExample] = []

        with open(repair_log, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.stats.total_seen += 1

                if not row.get("final_accepted"):
                    self.stats.total_rejected += 1
                    continue

                key = (row.get("db_id", ""), row.get("question_id"))
                if key not in failures:
                    self.stats.total_rejected += 1
                    continue

                failure = failures[key]
                example = ingester.build_repair_context(
                    failure,
                    use_compact_schema=random.random() < self.config.schema.compact_ratio,
                )
                example.corrected_sql = row.get("final_sql", "").strip()
                if not example.corrected_sql:
                    self.stats.total_rejected += 1
                    continue

                passed, _ = self.verifier.verify_real_failure_repair(
                    corrected_sql=example.corrected_sql,
                    reference_sql=failure.gold_sql,
                    broken_sql=failure.predicted_sql,
                    db_id=failure.db_id,
                )
                if not passed:
                    self.stats.total_rejected += 1
                    continue

                try:
                    failure_type = FailureType(row.get("failure_type", example.metadata.failure_type))
                except ValueError:
                    failure_type = FailureType(example.metadata.failure_type)

                example.metadata.source_type = SourceType.DETERMINISTIC_FIX.value
                example.metadata.failure_type = failure_type.value
                example.metadata.target_failure_mode = get_suggested_target_mode(failure_type).value
                example.metadata.corrected_sql_source = "deterministic"
                example.metadata.reference_sql_source = "gold"
                example.metadata.reference_sql = failure.gold_sql
                example.metadata.verification_passed = True
                example.metadata.subagent_used = False
                example.metadata.subagent_candidate_sql = example.corrected_sql
                example.metadata.subagent_accept_reason = row.get("final_reason")
                example.metadata.notes = "Deterministic repair from accepted repair log"
                example.metadata.relations = row.get("extracted_relations", []) or []
                example.metadata.original_question_id = failure.question_id
                example.metadata.sampling_weight = get_sampling_weight(
                    failure_type=failure_type,
                    db_family=failure.db_id,
                    source_type=SourceType.DETERMINISTIC_FIX.value,
                )
                example.metadata.example_id = (
                    f"deterministic_t10_{failure.db_id}_{failure.question_id}"
                )
                example.failure_type_hint = failure_type.value

                results.append(example)
                self.stats.total_accepted += 1
                self.stats.by_failure_type[failure_type.value] = (
                    self.stats.by_failure_type.get(failure_type.value, 0) + 1
                )

                if limit is not None and len(results) >= limit:
                    break

        return results

    def get_stats(self) -> DeterministicStats:
        return self.stats
