#!/usr/bin/env python3
"""
Synthetic Generator Module

Generates synthetic corruption examples from known-good SQL.
Parent SQL = Corrected SQL (no subagent needed).
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from .config import BuilderConfig, PathConfig
from .contamination import (
    ContaminationSource,
    BIRD_DEV_DB_IDS,
    NON_BENCHMARK_SOURCES,
    is_bird_dev_database,
)
from .corruption import (
    CORRUPTION_TRANSFORMS,
    CorruptionFamily,
    CorruptionResult,
    apply_random_corruption,
    apply_specific_corruption,
)
from .metadata import ExampleMetadata, Pool, RepairExample, SourceType
from .schema_builder import SchemaBuilder, SchemaInfo, load_schema_from_db
from .taxonomy import (
    FailureType,
    get_sampling_weight,
    get_suggested_target_mode,
)
from .verifier import Verifier


@dataclass
class GoldSqlSource:
    """A source of gold SQL for synthetic corruption."""
    
    sql: str
    db_id: str
    question: str
    hints: str
    source_name: str  # "bird_train", "spider", "bird_dev", etc.
    question_id: Optional[int] = None
    is_benchmark_safe: bool = True  # Safe for clean pool


@dataclass
class SyntheticStats:
    """Statistics from synthetic generation."""
    
    total_attempted: int = 0
    successful: int = 0
    failed_corruption: int = 0
    failed_verification: int = 0
    
    by_transform: Dict[str, int] = field(default_factory=dict)
    by_family: Dict[str, int] = field(default_factory=dict)
    by_db: Dict[str, int] = field(default_factory=dict)
    by_clean_status: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_attempted": self.total_attempted,
            "successful": self.successful,
            "failed_corruption": self.failed_corruption,
            "failed_verification": self.failed_verification,
            "success_rate": round(100 * self.successful / max(1, self.total_attempted), 2),
            "by_transform": self.by_transform,
            "by_family": self.by_family,
            "by_db": self.by_db,
            "by_clean_status": self.by_clean_status,
        }


class SyntheticGenerator:
    """Generator for synthetic corruption examples."""
    
    def __init__(
        self,
        config: BuilderConfig,
        query_timeout_seconds: Optional[float] = None,
    ):
        self.config = config
        self.paths = config.paths
        self.schema_builder = SchemaBuilder(self.paths, config.schema)
        self.verifier = Verifier(
            self.paths,
            query_timeout_seconds=query_timeout_seconds,
        )
        self.stats = SyntheticStats()
        
        # Cache
        self._schemas: Dict[str, SchemaInfo] = {}
        self._gold_sources: List[GoldSqlSource] = []
    
    def _get_schema(self, db_id: str) -> SchemaInfo:
        """Get schema for a database (cached)."""
        if db_id not in self._schemas:
            db_path = self.paths.database_path(db_id)
            self._schemas[db_id] = load_schema_from_db(db_path)
        return self._schemas[db_id]
    
    def add_gold_source(self, source: GoldSqlSource) -> None:
        """Add a gold SQL source."""
        self._gold_sources.append(source)
    
    def load_gold_from_bird_train(self, train_file: Path) -> int:
        """
        Load gold SQL from BIRD train set (safe for clean pool).
        
        Returns number of sources loaded.
        """
        count = 0
        
        if not train_file.exists():
            return 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract fields
                    sql = data.get('gold_sql', data.get('sql', ''))
                    db_id = data.get('db_id', '')
                    question = data.get('question', '')
                    hints = data.get('evidence', data.get('hints', ''))
                    
                    if sql and db_id:
                        self._gold_sources.append(GoldSqlSource(
                            sql=sql,
                            db_id=db_id,
                            question=question,
                            hints=hints,
                            source_name="bird_train",
                            question_id=data.get('question_id'),
                            is_benchmark_safe=True,
                        ))
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        return count
    
    def load_gold_from_bird_dev(self, dev_file: Path) -> int:
        """
        Load gold SQL from BIRD dev set (internal only).
        
        Returns number of sources loaded.
        """
        count = 0
        
        if not dev_file.exists():
            return 0
        
        with open(dev_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    sql = data.get('gold_sql', '')
                    db_id = data.get('db_id', '')
                    question = data.get('question', '')
                    hints = data.get('evidence', data.get('hints', ''))
                    
                    if sql and db_id:
                        self._gold_sources.append(GoldSqlSource(
                            sql=sql,
                            db_id=db_id,
                            question=question,
                            hints=hints,
                            source_name="bird_dev",
                            question_id=data.get('question_id'),
                            is_benchmark_safe=False,  # Not safe for clean
                        ))
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        return count
    
    def generate_one(
        self,
        source: GoldSqlSource,
        transform_name: Optional[str] = None,
        use_compact_schema: bool = True,
    ) -> Optional[RepairExample]:
        """
        Generate one synthetic example from a gold source.
        
        Args:
            source: Gold SQL source
            transform_name: Specific transform to apply (random if None)
            use_compact_schema: Use compact schema context
        
        Returns:
            RepairExample if successful, None otherwise
        """
        self.stats.total_attempted += 1
        
        # Get schema
        try:
            schema = self._get_schema(source.db_id)
        except Exception:
            self.stats.failed_corruption += 1
            return None
        
        # Apply corruption
        if transform_name:
            result = apply_specific_corruption(source.sql, schema, transform_name)
        else:
            result = apply_random_corruption(source.sql, schema)
        
        if not result or not result.success:
            self.stats.failed_corruption += 1
            return None
        
        # Verify the pair
        passed, details = self.verifier.verify_synthetic(
            parent_sql=source.sql,
            broken_sql=result.broken_sql,
            db_id=source.db_id,
        )
        
        if not passed:
            self.stats.failed_verification += 1
            return None
        
        # Build schema context
        schema_context, schema_meta = self.schema_builder.build_context(
            db_id=source.db_id,
            question=source.question,
            hints=source.hints,
            broken_sql=result.broken_sql,
            use_compact=use_compact_schema,
        )
        
        # Determine contamination status
        if source.is_benchmark_safe:
            contamination_source = ContaminationSource.NON_BENCHMARK.value
            pool = Pool.CLEAN.value
            benchmark_clean = True
        else:
            contamination_source = ContaminationSource.BIRD_GOLD_REFERENCE.value
            pool = Pool.INTERNAL.value
            benchmark_clean = False
        
        # Map transform to failure type
        transform = next(
            (t for t in CORRUPTION_TRANSFORMS if t.name == result.transform_name),
            None
        )
        failure_type_str = transform.failure_type if transform else "wrong_return_field"
        
        try:
            failure_type = FailureType(failure_type_str)
        except ValueError:
            failure_type = FailureType.WRONG_RETURN_FIELD
        
        target_mode = get_suggested_target_mode(failure_type)
        
        # Build example ID
        example_id = f"synth_{source.source_name}_{source.db_id}_{self.stats.successful}"
        
        # Calculate sampling weight
        sampling_weight = get_sampling_weight(
            failure_type=failure_type,
            db_family=source.db_id,
            source_type="synthetic_corruption",
        )
        
        # Build metadata
        metadata = ExampleMetadata(
            example_id=example_id,
            split="train",
            source_type=SourceType.SYNTHETIC_CORRUPTION.value,
            source_run=None,
            original_question_id=source.question_id,
            db_family=source.db_id,
            schema_context_type=schema_meta.get("schema_context_type", "compact_relevant"),
            schema_tables_kept=schema_meta.get("tables_kept", []),
            schema_columns_kept=schema_meta.get("columns_kept", 0),
            failure_type=failure_type.value,
            target_failure_mode=target_mode.value,
            exec_failed_originally=False,  # Synthetic may or may not exec fail
            error_message=result.error_message,
            contamination_source=contamination_source,
            benchmark_clean=benchmark_clean,
            pool=pool,
            difficulty="simple",  # Synthetic are usually simpler
            sampling_weight=sampling_weight,
            corrected_sql_source="synthetic_parent",
            reference_sql_source="synthetic_parent",
            reference_sql=source.sql,
            verification_method="execution_match",
            verification_passed=True,
            subagent_used=False,
            corruption_transform=result.transform_name,
            corruption_family=result.family,
            corruption_severity=result.severity,
            parent_sql=source.sql,
        )
        
        # Build example
        example = RepairExample(
            schema_context=schema_context,
            question=source.question,
            hints=source.hints,
            broken_sql=result.broken_sql,
            error=result.error_message,
            failure_type_hint=failure_type.value,
            corrected_sql=source.sql,  # Parent SQL is the correction
            metadata=metadata,
        )
        
        # Update stats
        self.stats.successful += 1
        self.stats.by_transform[result.transform_name] = \
            self.stats.by_transform.get(result.transform_name, 0) + 1
        self.stats.by_family[result.family] = \
            self.stats.by_family.get(result.family, 0) + 1
        self.stats.by_db[source.db_id] = \
            self.stats.by_db.get(source.db_id, 0) + 1
        
        clean_status = "clean" if benchmark_clean else "internal"
        self.stats.by_clean_status[clean_status] = \
            self.stats.by_clean_status.get(clean_status, 0) + 1
        
        return example
    
    def generate_batch(
        self,
        count: int,
        clean_only: bool = False,
        prefer_transforms: Optional[List[str]] = None,
    ) -> Generator[RepairExample, None, None]:
        """
        Generate a batch of synthetic examples.
        
        Args:
            count: Target number of examples
            clean_only: Only use benchmark-safe sources
            prefer_transforms: Preferred transform names
        
        Yields:
            RepairExample objects
        """
        if not self._gold_sources:
            return
        
        # Filter sources
        sources = self._gold_sources
        if clean_only:
            sources = [s for s in sources if s.is_benchmark_safe]
        
        if not sources:
            return
        
        generated = 0
        max_attempts = count * 3  # Allow some failures
        attempts = 0
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            
            # Pick random source
            source = random.choice(sources)
            
            # Pick transform
            transform_name = None
            if prefer_transforms:
                transform_name = random.choice(prefer_transforms)
            
            # Try to generate
            example = self.generate_one(
                source=source,
                transform_name=transform_name,
                use_compact_schema=self.schema_builder.should_use_compact(),
            )
            
            if example:
                yield example
                generated += 1
    
    def get_stats(self) -> SyntheticStats:
        """Get generation statistics."""
        return self.stats
