#!/usr/bin/env python3
"""
Configuration Module

Dataclasses for builder configuration, dataset targets, and paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetTargets:
    """Target sizes for dataset builds."""
    
    # Internal dataset (full, includes benchmark-derived)
    internal_train: int = 18000
    internal_dev: int = 2000
    
    # Clean dataset (benchmark-safe)
    clean_train_min: int = 8000
    clean_train_max: int = 12000
    clean_dev_min: int = 1000
    clean_dev_max: int = 1500
    
    # Composition caps (percentages)
    real_failures_min: float = 0.20
    real_failures_max: float = 0.35
    synthetic_min: float = 0.50
    synthetic_max: float = 0.65
    contrastive_manual_min: float = 0.10
    contrastive_manual_max: float = 0.25


@dataclass
class FailureFamilyQuotas:
    """Minimum counts for failure families in training set."""
    
    # High-value failure types
    wrong_return_field: int = 1500
    wrong_count_granularity: int = 1200
    wrong_denominator: int = 1000
    wrong_cohort_definition: int = 800
    join_path_error: int = 1200
    temporal_anchor_error: int = 600
    table_family_confusion: int = 800
    alias_error: int = 600
    wrong_table_side_error: int = 500
    missing_distinct: int = 500
    syntax_local_error: int = 400
    degenerate_or_truncated_sql: int = 300


@dataclass
class DatabaseFamilyQuotas:
    """Minimum counts for database families in training set."""
    
    # Priority weak families
    california_schools: int = 400
    financial: int = 400
    formula_1: int = 400
    thrombosis_prediction: int = 350
    debit_card_specializing: int = 300
    card_games: int = 350
    toxicology: int = 300
    
    # Robustness families
    codebase_community: int = 200
    european_football_2: int = 200
    student_club: int = 150
    superhero: int = 150


@dataclass
class SubagentConfig:
    """Configuration for subagent calls."""
    
    model: str = "claude-haiku-4.5"
    fallback_model: str = "claude-sonnet-4.5"
    max_parallel: int = 2
    initial_attempts: int = 1
    retry_attempts: int = 1  # After initial failure
    timeout_seconds: int = 60


@dataclass
class SchemaConfig:
    """Configuration for schema context building."""
    
    compact_ratio: float = 0.75  # 75% compact, 25% wider/full
    max_compact_tables: int = 8
    max_compact_columns_per_table: int = 15
    include_pk_fk: bool = True
    include_bridge_tables: bool = True


@dataclass
class PathConfig:
    """Paths to input files and output directories."""
    
    # Base paths
    project_root: Path = field(default_factory=lambda: Path("/Users/arnav/programming/lm"))
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training"
    
    @property
    def output_dir(self) -> Path:
        return self.training_dir / "error_correction"
    
    # Input: T10 artifacts
    @property
    def t10_eval_dir(self) -> Path:
        return self.project_root / "runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval"
    
    @property
    def t10_predictions_dir(self) -> Path:
        return self.project_root / "runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions"
    
    @property
    def t10_data_dir(self) -> Path:
        return self.training_dir / "t10"
    
    # Input: T11_1 artifacts (auxiliary)
    @property
    def t11_1_eval_dir(self) -> Path:
        return self.project_root / "runs/t11_1_baseline_3090/qwen3-1.7b/without-sampling/eval"
    
    @property
    def t11_1_predictions_dir(self) -> Path:
        return self.project_root / "runs/t11_1_baseline_3090/qwen3-1.7b/without-sampling/predictions"
    
    @property
    def t11_1_data_dir(self) -> Path:
        return self.training_dir / "t11_1"
    
    # Input: Database files
    @property
    def databases_dir(self) -> Path:
        return self.data_dir / "bird_eval_datasets/dev_databases"
    
    # Input: Failure archetypes
    @property
    def failure_archetypes_file(self) -> Path:
        return self.t10_eval_dir / "failure_archetypes_all_databases.md"
    
    # Specific input files
    def per_example_results(self, source: str = "t10") -> Path:
        if source == "t10":
            return self.t10_eval_dir / "per_example_results_t10.jsonl"
        elif source == "t11_1":
            return self.t11_1_eval_dir / "per_example_results_t11_1.jsonl"
        raise ValueError(f"Unknown source: {source}")
    
    def bird_dev_prompts(self, source: str = "t10") -> Path:
        if source == "t10":
            return self.t10_data_dir / "bird_dev_t10.jsonl"
        elif source == "t11_1":
            return self.t11_1_data_dir / "bird_dev_t11_1.jsonl"
        raise ValueError(f"Unknown source: {source}")
    
    def predictions_file(self, source: str = "t10") -> Path:
        if source == "t10":
            return self.t10_predictions_dir / "predictions_t10.jsonl"
        elif source == "t11_1":
            return self.t11_1_predictions_dir / "predictions_t11_1.jsonl"
        raise ValueError(f"Unknown source: {source}")
    
    def database_path(self, db_id: str) -> Path:
        return self.databases_dir / db_id / f"{db_id}.sqlite"


@dataclass
class BuilderConfig:
    """Main configuration for the dataset builder."""
    
    targets: DatasetTargets = field(default_factory=DatasetTargets)
    failure_quotas: FailureFamilyQuotas = field(default_factory=FailureFamilyQuotas)
    db_quotas: DatabaseFamilyQuotas = field(default_factory=DatabaseFamilyQuotas)
    subagent: SubagentConfig = field(default_factory=SubagentConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Build options
    version: str = "v1"
    build_size: str = "B"  # "A" for minimal, "B" for serious
    include_t11_1: bool = True  # Include auxiliary T11_1 failures
    dry_run: bool = False
    verbose: bool = True
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check input files exist
        for source in ["t10"] + (["t11_1"] if self.include_t11_1 else []):
            if not self.paths.per_example_results(source).exists():
                errors.append(f"Missing per_example_results for {source}")
            if not self.paths.bird_dev_prompts(source).exists():
                errors.append(f"Missing bird_dev_prompts for {source}")
            if not self.paths.predictions_file(source).exists():
                errors.append(f"Missing predictions for {source}")
        
        if not self.paths.databases_dir.exists():
            errors.append(f"Missing databases directory: {self.paths.databases_dir}")
        
        # Validate composition caps sum to reasonable range
        total_min = (self.targets.real_failures_min + 
                    self.targets.synthetic_min + 
                    self.targets.contrastive_manual_min)
        total_max = (self.targets.real_failures_max + 
                    self.targets.synthetic_max + 
                    self.targets.contrastive_manual_max)
        
        if total_min > 1.0:
            errors.append(f"Composition minimums sum to {total_min}, exceeds 100%")
        if total_max < 1.0:
            errors.append(f"Composition maximums sum to {total_max}, less than 100%")
        
        return errors
