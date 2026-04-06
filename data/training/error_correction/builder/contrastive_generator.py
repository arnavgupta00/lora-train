#!/usr/bin/env python3
"""
Contrastive Example Generator

Builds high-value contrastive/output-discipline examples by applying targeted,
archetype-driven corruptions to known-good SQL. Corrected SQL remains the
trusted parent query, while metadata marks these examples as contrastive.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from .config import BuilderConfig
from .metadata import SourceType
from .synthetic_generator import GoldSqlSource, SyntheticGenerator
from .taxonomy import FailureType, TargetFailureMode, get_sampling_weight


@dataclass
class ContrastivePattern:
    """Targeted pattern definition."""

    pattern_id: str
    name: str
    transform_name: str
    failure_type: FailureType
    target_failure_mode: TargetFailureMode
    description: str


CONTRASTIVE_PATTERNS: List[ContrastivePattern] = [
    ContrastivePattern(
        pattern_id="id_vs_name",
        name="ID vs Name",
        transform_name="wrong_column",
        failure_type=FailureType.WRONG_RETURN_FIELD,
        target_failure_mode=TargetFailureMode.RETURN_FIELD_MISMATCH,
        description="Return-field mismatch on a near-miss identifier.",
    ),
    ContrastivePattern(
        pattern_id="count_distinct",
        name="COUNT vs COUNT DISTINCT",
        transform_name="count_distinct_to_count",
        failure_type=FailureType.WRONG_COUNT_GRANULARITY,
        target_failure_mode=TargetFailureMode.WRONG_COUNT_GRANULARITY,
        description="Drop DISTINCT in a counting query.",
    ),
    ContrastivePattern(
        pattern_id="count_star",
        name="COUNT Column vs COUNT Star",
        transform_name="count_to_count_star",
        failure_type=FailureType.WRONG_COUNT_GRANULARITY,
        target_failure_mode=TargetFailureMode.WRONG_COUNT_GRANULARITY,
        description="Collapse a column count into COUNT(*).",
    ),
    ContrastivePattern(
        pattern_id="missing_distinct",
        name="Missing DISTINCT",
        transform_name="remove_distinct",
        failure_type=FailureType.MISSING_DISTINCT,
        target_failure_mode=TargetFailureMode.DISTINCT_SHAPE_ERROR,
        description="Remove DISTINCT from a query where unique results matter.",
    ),
    ContrastivePattern(
        pattern_id="wrong_cohort",
        name="Wrong Cohort Definition",
        transform_name="drop_filter",
        failure_type=FailureType.WRONG_COHORT_DEFINITION,
        target_failure_mode=TargetFailureMode.WRONG_DENOMINATOR,
        description="Drop a cohort-defining WHERE condition.",
    ),
    ContrastivePattern(
        pattern_id="wrong_denominator",
        name="Wrong Denominator",
        transform_name="wrong_denominator",
        failure_type=FailureType.WRONG_DENOMINATOR,
        target_failure_mode=TargetFailureMode.WRONG_DENOMINATOR,
        description="Change the denominator inside a ratio/percentage query.",
    ),
    ContrastivePattern(
        pattern_id="wrong_table_side",
        name="Wrong Table Side",
        transform_name="wrong_join_side_column",
        failure_type=FailureType.WRONG_TABLE_SIDE_ERROR,
        target_failure_mode=TargetFailureMode.WRONG_TABLE_OWNERSHIP,
        description="Read a field from the wrong side of a join.",
    ),
    ContrastivePattern(
        pattern_id="alias_error",
        name="Alias Error",
        transform_name="wrong_table_alias",
        failure_type=FailureType.ALIAS_ERROR,
        target_failure_mode=TargetFailureMode.COLUMN_ERROR,
        description="Swap a qualified reference onto the wrong alias.",
    ),
    ContrastivePattern(
        pattern_id="join_path",
        name="Wrong Join Path",
        transform_name="wrong_join_path",
        failure_type=FailureType.JOIN_PATH_ERROR,
        target_failure_mode=TargetFailureMode.JOIN_PATH_ERROR,
        description="Mutate an ON-clause key onto the wrong join path.",
    ),
    ContrastivePattern(
        pattern_id="table_family",
        name="Table Family Confusion",
        transform_name="table_family_confusion",
        failure_type=FailureType.TABLE_FAMILY_CONFUSION,
        target_failure_mode=TargetFailureMode.PHASE_TABLE_CONFUSION,
        description="Swap in a nearby but wrong table family member.",
    ),
    ContrastivePattern(
        pattern_id="temporal_anchor",
        name="Temporal Anchor",
        transform_name="wrong_temporal_anchor",
        failure_type=FailureType.TEMPORAL_ANCHOR_ERROR,
        target_failure_mode=TargetFailureMode.TEMPORAL_ANCHOR_ERROR,
        description="Use the wrong date column or date anchor.",
    ),
    ContrastivePattern(
        pattern_id="output_rank_shape",
        name="Output Shape",
        transform_name="change_limit",
        failure_type=FailureType.OUTPUT_SHAPE_ERROR,
        target_failure_mode=TargetFailureMode.OUTPUT_FORMAT_ERROR,
        description="Change LIMIT to the wrong output shape.",
    ),
    ContrastivePattern(
        pattern_id="sort_direction",
        name="Sort Direction",
        transform_name="wrong_order_direction",
        failure_type=FailureType.OUTPUT_SHAPE_ERROR,
        target_failure_mode=TargetFailureMode.OUTPUT_FORMAT_ERROR,
        description="Reverse the intended ranking direction.",
    ),
    ContrastivePattern(
        pattern_id="truncate_sql",
        name="Degenerate SQL",
        transform_name="truncate",
        failure_type=FailureType.DEGENERATE_OR_TRUNCATED_SQL,
        target_failure_mode=TargetFailureMode.TRUNCATION_ERROR,
        description="Truncate the SQL into a degenerate output.",
    ),
]


@dataclass
class ContrastiveStats:
    """Statistics for contrastive generation."""

    total_attempted: int = 0
    total_generated: int = 0
    by_pattern: Dict[str, int] = field(default_factory=dict)
    by_failure_type: Dict[str, int] = field(default_factory=dict)
    by_clean_status: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_attempted": self.total_attempted,
            "total_generated": self.total_generated,
            "by_pattern": dict(self.by_pattern),
            "by_failure_type": dict(self.by_failure_type),
            "by_clean_status": dict(self.by_clean_status),
        }


class ContrastiveGenerator:
    """Generate targeted contrastive examples from trusted gold SQL sources."""

    def __init__(
        self,
        config: BuilderConfig,
        query_timeout_seconds: Optional[float] = None,
    ):
        self.config = config
        self.synthetic = SyntheticGenerator(
            config,
            query_timeout_seconds=query_timeout_seconds,
        )
        self.stats = ContrastiveStats()
        self._counter = 0

    def add_gold_source(self, source: GoldSqlSource) -> None:
        self.synthetic.add_gold_source(source)

    def load_gold_from_bird_train(self, train_file) -> int:
        return self.synthetic.load_gold_from_bird_train(train_file)

    def load_gold_from_bird_dev(self, dev_file) -> int:
        return self.synthetic.load_gold_from_bird_dev(dev_file)

    def load_gold_from_generated_files(self, files, source_name: str) -> int:
        return self.synthetic.load_gold_from_generated_files(files, source_name)

    def get_patterns(self) -> List[ContrastivePattern]:
        return CONTRASTIVE_PATTERNS

    def _pattern_by_transform(self, transform_name: str) -> ContrastivePattern:
        for pattern in CONTRASTIVE_PATTERNS:
            if pattern.transform_name == transform_name:
                return pattern
        raise ValueError(f"Unknown contrastive transform: {transform_name}")

    def _candidate_sources_for_transform(
        self,
        sources: List[GoldSqlSource],
        transform_name: str,
    ) -> List[GoldSqlSource]:
        """Pick sources that are structurally compatible with a transform."""
        compatible: List[GoldSqlSource] = []
        for source in sources:
            sql_upper = source.sql.upper()
            if transform_name == "count_distinct_to_count" and "COUNT(DISTINCT" not in sql_upper:
                continue
            if transform_name == "count_to_count_star" and "COUNT(" not in sql_upper:
                continue
            if transform_name == "remove_distinct" and "DISTINCT" not in sql_upper:
                continue
            if transform_name == "wrong_denominator" and ("/" not in source.sql or "COUNT" not in sql_upper):
                continue
            if transform_name in {"wrong_join_side_column", "wrong_table_alias", "wrong_join_path"} and " JOIN " not in sql_upper:
                continue
            if transform_name == "change_limit" and "LIMIT" not in sql_upper:
                continue
            if transform_name == "wrong_order_direction" and not any(token in sql_upper for token in (" ASC", " DESC")):
                continue
            if transform_name == "wrong_temporal_anchor" and not any(
                token in sql_upper for token in ("DATE", "TIME", "_TS", "_ON", "_AT")
            ):
                continue
            if transform_name == "table_family_confusion" and source.schema_ddl and source.schema_ddl.upper().count("CREATE TABLE") < 2:
                continue
            compatible.append(source)
        return compatible or sources

    def generate_one(
        self,
        source: GoldSqlSource,
        transform_name: str,
        use_compact_schema: bool = True,
    ):
        self.stats.total_attempted += 1
        pattern = self._pattern_by_transform(transform_name)
        example = self.synthetic.generate_one(
            source=source,
            transform_name=transform_name,
            use_compact_schema=use_compact_schema,
        )
        if example is None:
            return None

        self._counter += 1
        example.metadata.example_id = (
            f"contrastive_{pattern.pattern_id}_{source.db_id}_{self._counter}"
        )
        example.metadata.source_type = SourceType.CONTRASTIVE.value
        example.metadata.failure_type = pattern.failure_type.value
        example.metadata.target_failure_mode = pattern.target_failure_mode.value
        example.metadata.difficulty = "moderate"
        example.metadata.notes = f"Contrastive pattern: {pattern.description}"
        example.metadata.corrected_sql_source = "synthetic_parent"
        example.metadata.sampling_weight = get_sampling_weight(
            failure_type=pattern.failure_type,
            db_family=source.db_id,
            source_type=SourceType.CONTRASTIVE.value,
        )
        example.failure_type_hint = pattern.failure_type.value

        clean_status = "clean" if example.metadata.benchmark_clean else "internal"
        self.stats.total_generated += 1
        self.stats.by_pattern[pattern.pattern_id] = (
            self.stats.by_pattern.get(pattern.pattern_id, 0) + 1
        )
        self.stats.by_failure_type[pattern.failure_type.value] = (
            self.stats.by_failure_type.get(pattern.failure_type.value, 0) + 1
        )
        self.stats.by_clean_status[clean_status] = (
            self.stats.by_clean_status.get(clean_status, 0) + 1
        )
        return example

    def generate_batch(
        self,
        count: int,
        clean_only: bool = False,
        transform_names: Optional[List[str]] = None,
    ) -> Generator:
        if count <= 0:
            return

        sources = list(self.synthetic._gold_sources)
        if clean_only:
            sources = [source for source in sources if source.is_benchmark_safe]
        if not sources:
            return

        patterns = transform_names or [pattern.transform_name for pattern in CONTRASTIVE_PATTERNS]
        generated = 0
        attempts = 0
        max_attempts = count * 40
        used: Set[Tuple[str, Optional[int], str]] = set()

        while generated < count and attempts < max_attempts:
            attempts += 1
            transform_name = patterns[generated % len(patterns)]
            candidate_sources = self._candidate_sources_for_transform(sources, transform_name)
            source = random.choice(candidate_sources)
            dedup_key = (source.db_id, source.question_id, transform_name)
            if dedup_key in used:
                continue

            example = self.generate_one(
                source=source,
                transform_name=transform_name,
                use_compact_schema=self.synthetic.schema_builder.should_use_compact(),
            )
            if example is None:
                continue

            used.add(dedup_key)
            yield example
            generated += 1

    def get_stats(self) -> ContrastiveStats:
        return self.stats
