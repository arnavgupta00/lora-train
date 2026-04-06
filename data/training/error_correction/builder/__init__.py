"""
SQL Error-Correction Dataset Builder

This package provides tools for building supervised datasets
for training SQL error-correction models.

Architecture:
- config.py: Configuration dataclasses
- metadata.py: Metadata schema for examples
- taxonomy.py: Failure type taxonomy
- schema_builder.py: Schema context construction
- corruption.py: Synthetic SQL corruption transforms
- contamination.py: Contamination tagging and routing
- verifier.py: Validation logic
- real_failure_ingester.py: Real failure parsing
- subagent_client.py: Subagent for corrected SQL proposals
- synthetic_generator.py: Synthetic corruption generation
- contrastive_generator.py: Contrastive example generation
- deduplicator.py: Deduplication logic
- splitter.py: Train/dev splitting
- writer.py: JSONL export and reporting
"""

from .config import BuilderConfig, DatasetTargets, PathConfig
from .metadata import ExampleMetadata, ContaminationSource, Pool, SourceType, RepairExample
from .taxonomy import FailureType, TargetFailureMode
from .contamination import ContaminationRouter, RoutingDecision
from .contrastive_generator import ContrastiveGenerator, ContrastivePattern, CONTRASTIVE_PATTERNS
from .deduplicator import Deduplicator
from .real_failure_ingester import RealFailureIngester, RealFailure
from .schema_builder import SchemaBuilder
from .splitter import split_pools
from .subagent_client import SubagentClient, MockSubagentClient
from .synthetic_generator import SyntheticGenerator
from .verifier import Verifier
from .writer import DatasetWriter

__all__ = [
    # Config
    "BuilderConfig",
    "DatasetTargets",
    "PathConfig",
    # Metadata
    "ExampleMetadata",
    "ContaminationSource",
    "Pool",
    "SourceType",
    "RepairExample",
    # Taxonomy
    "FailureType",
    "TargetFailureMode",
    # Components
    "ContaminationRouter",
    "RoutingDecision",
    "ContrastiveGenerator",
    "ContrastivePattern",
    "CONTRASTIVE_PATTERNS",
    "Deduplicator",
    "RealFailureIngester",
    "RealFailure",
    "SchemaBuilder",
    "split_pools",
    "SubagentClient",
    "MockSubagentClient",
    "SyntheticGenerator",
    "Verifier",
    "DatasetWriter",
]
