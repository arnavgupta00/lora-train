#!/usr/bin/env python3
"""
SQL Error-Correction Dataset Builder

Main entry point for building the error-correction dataset.

Usage:
    python build_dataset.py --version v1 --size B
    python build_dataset.py --dry-run  # Preview without writing
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import Manager, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from builder.config import BuilderConfig, DatasetTargets
from builder.contamination import ContaminationRouter, RoutingDecision
from builder.deduplicator import Deduplicator
from builder.metadata import Pool, RepairExample
from builder.real_failure_ingester import RealFailureIngester
from builder.schema_builder import SchemaBuilder
from builder.splitter import split_pools
from builder.subagent_client import MockSubagentClient, SubagentClient
from builder.synthetic_generator import SyntheticGenerator
from builder.taxonomy import FailureType
from builder.verifier import Verifier
from builder.writer import DatasetWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SQL Error-Correction Dataset"
    )
    parser.add_argument(
        "--version", "-v",
        default="v1",
        help="Version string for output files (default: v1)",
    )
    parser.add_argument(
        "--size",
        choices=["A", "B"],
        default="B",
        help="Build size: A (10k/1k) or B (18k/2k) (default: B)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview build without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--limit-real",
        type=int,
        default=None,
        help="Limit number of real failures to process",
    )
    parser.add_argument(
        "--limit-synthetic",
        type=int,
        default=None,
        help="Limit number of synthetic examples to generate",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic generation (for testing)",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Skip real failure ingestion (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, cpu_count()),
        help=f"Number of parallel workers (default: {min(4, cpu_count())})",
    )
    return parser.parse_args()


def log(msg: str, verbose: bool = True):
    """Print log message with timestamp."""
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")
        sys.stdout.flush()


def save_progress(
    examples: List[RepairExample],
    output_dir: Path,
    prefix: str,
    version: str,
) -> None:
    """Save progress incrementally to JSONL files."""
    clean_examples = [e for e in examples if e.metadata.pool == Pool.CLEAN]
    internal_examples = [e for e in examples if e.metadata.pool == Pool.INTERNAL]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clean examples
    if clean_examples:
        clean_file = output_dir / f"{prefix}_clean_{version}.jsonl"
        with open(clean_file, 'w') as f:
            for ex in clean_examples:
                f.write(json.dumps(ex.to_sft_format()) + '\n')
    
    # Save internal examples
    if internal_examples:
        internal_file = output_dir / f"{prefix}_internal_{version}.jsonl"
        with open(internal_file, 'w') as f:
            for ex in internal_examples:
                f.write(json.dumps(ex.to_sft_format()) + '\n')


def random_compact_choice(ratio: float) -> bool:
    """Randomly choose compact vs full schema based on ratio."""
    return random.random() < ratio


def generate_synthetic_batch_worker(args):
    """
    Worker function for parallel synthetic generation.
    
    Args:
        args: Tuple of (config, start_idx, count, seed)
    
    Returns:
        List of RepairExample objects
    """
    config, start_idx, count, seed = args
    
    # Set random seed for reproducibility within this worker
    random.seed(seed)
    
    # Create generator instance
    generator = SyntheticGenerator(config)
    
    # Load gold sources
    train_file = config.paths.t10_data_dir / "train_t10.jsonl"
    dev_file = config.paths.bird_dev_prompts("t10")
    
    if train_file.exists():
        generator.load_gold_from_bird_train(train_file)
    
    if dev_file.exists():
        generator.load_gold_from_bird_dev(dev_file)
    
    # Generate examples for this batch
    examples = []
    for example in generator.generate_batch(count=count, clean_only=False):
        examples.append(example)
        if len(examples) >= count:
            break
    
    return examples


def build_dataset(
    config: BuilderConfig,
    dry_run: bool = False,
    limit_real: Optional[int] = None,
    limit_synthetic: Optional[int] = None,
    skip_synthetic: bool = False,
    skip_real: bool = False,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Build the error-correction dataset.
    
    Pipeline order:
    1. Build candidates (real failures + synthetic)
    2. Validate
    3. Contamination route
    4. Dedup (per pool)
    5. Split train/dev (per pool)
    6. Export
    
    Returns:
        Build statistics and results
    """
    start_time = time.time()
    verbose = config.verbose
    
    # Initialize components
    log("Initializing components...", verbose)
    verifier = Verifier(config.paths)
    contamination_router = ContaminationRouter()
    deduplicator = Deduplicator(semantic_threshold=0.85)
    
    # Collect all examples
    all_examples: List[RepairExample] = []
    rejected_examples: List[RepairExample] = []
    
    # Stats tracking
    build_stats: Dict[str, Any] = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "version": config.version,
            "build_size": config.build_size,
            "include_t11_1": config.include_t11_1,
        },
        "real_failures": {},
        "synthetic": {},
        "contamination": {},
        "deduplication": {},
        "splitting": {},
        "verification": {"passed": 0, "failed": 0},
    }
    
    # =========================================================================
    # Phase 1: Real Failure Ingestion
    # =========================================================================
    if not skip_real:
        log("Phase 1: Ingesting real failures...", verbose)
        
        ingester = RealFailureIngester(config)
        
        # For now, we'll use a mock subagent that uses gold SQL
        # In production, this would call the actual Claude API
        subagent = MockSubagentClient(config.subagent)
        
        real_count = 0
        for failure in ingester.iter_failures(
            include_t11_1=config.include_t11_1,
            limit=limit_real,
        ):
            # Build repair context
            use_compact = random_compact_choice(config.schema.compact_ratio)
            example = ingester.build_repair_context(failure, use_compact)
            
            # For real failures, we need subagent to propose corrected SQL
            # For this implementation, we'll use the gold SQL as the correction
            # (In production, subagent would generate and we'd validate)
            
            # Use gold SQL as corrected SQL (mock subagent)
            example.corrected_sql = failure.gold_sql
            example.metadata.subagent_used = False  # Mock doesn't count
            example.metadata.corrected_sql_source = "gold_aligned"
            
            # Verify
            passed, details = verifier.verify_real_failure_repair(
                corrected_sql=example.corrected_sql,
                reference_sql=failure.gold_sql,
                broken_sql=example.broken_sql,
                db_id=failure.db_id,
            )
            
            example.metadata.verification_passed = passed
            
            if passed:
                build_stats["verification"]["passed"] += 1
            else:
                build_stats["verification"]["failed"] += 1
                rejected_examples.append(example)
                continue
            
            # Route via contamination
            routing = contamination_router.tag_and_route(
                example_id=example.metadata.example_id,
                source_type=example.metadata.source_type,
                source_run=example.metadata.source_run,
                db_id=failure.db_id,
                reference_sql_source=example.metadata.reference_sql_source,
                validation_passed=passed,
                reference_sql_valid=True,
                has_required_fields=True,
            )
            
            contamination_router.update_metadata(example.metadata, routing)
            
            if routing.decision == RoutingDecision.REJECTED:
                rejected_examples.append(example)
                continue
            
            all_examples.append(example)
            real_count += 1
            
            # Save progress every 100 examples
            if real_count % 100 == 0:
                log(f"  Progress: {real_count} real failures processed, saving checkpoint...", verbose)
                if not dry_run:
                    save_progress(all_examples, config.paths.output_dir, "progress_real", config.version)
        
        build_stats["real_failures"] = ingester.get_stats().to_dict()
        log(f"  Ingested {real_count} real failures", verbose)
        
        # Final save after real failures
        if not dry_run and real_count > 0:
            log(f"  Saving {real_count} real failures to disk...", verbose)
            save_progress(all_examples, config.paths.output_dir, "checkpoint_after_real", config.version)
    
    # =========================================================================
    # Phase 2: Synthetic Generation (Parallelized)
    # =========================================================================
    if not skip_synthetic:
        log(f"Phase 2: Generating synthetic examples using {num_workers} workers...", verbose)
        
        # Calculate target
        target_synthetic = limit_synthetic
        if target_synthetic is None:
            total_target = config.targets.internal_train
            target_synthetic = int(total_target * 0.6)  # ~60% synthetic
        
        log(f"  Target: {target_synthetic} synthetic examples", verbose)
        
        # Split work across workers
        batch_size = max(100, target_synthetic // num_workers)
        batches = []
        remaining = target_synthetic
        batch_idx = 0
        
        while remaining > 0:
            count = min(batch_size, remaining)
            seed = 42 + batch_idx  # Reproducible seeds
            batches.append((config, batch_idx, count, seed))
            remaining -= count
            batch_idx += 1
        
        log(f"  Split into {len(batches)} batches across {num_workers} workers", verbose)
        
        # Generate in parallel using threads (safer than processes for this use case)
        synthetic_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(generate_synthetic_batch_worker, batch): i 
                      for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_examples = future.result()
                    
                    # Process each example
                    for example in batch_examples:
                        # Route via contamination
                        routing = contamination_router.tag_and_route(
                            example_id=example.metadata.example_id,
                            source_type=example.metadata.source_type,
                            source_run=None,
                            db_id=example.metadata.db_family,
                            reference_sql_source=example.metadata.reference_sql_source,
                            validation_passed=True,
                            reference_sql_valid=True,
                            has_required_fields=True,
                            parent_source="bird_train" if example.metadata.benchmark_clean else "bird_dev",
                        )
                        
                        contamination_router.update_metadata(example.metadata, routing)
                        
                        if routing.decision == RoutingDecision.REJECTED:
                            rejected_examples.append(example)
                            continue
                        
                        all_examples.append(example)
                        synthetic_count += 1
                    
                    log(f"  Batch {batch_idx + 1}/{len(batches)} complete: {len(batch_examples)} examples ({synthetic_count}/{target_synthetic} total)", verbose)
                    
                    # Save progress every 500 examples
                    if synthetic_count % 500 == 0 or synthetic_count >= target_synthetic:
                        log(f"  Progress: {synthetic_count} synthetic examples generated, saving checkpoint...", verbose)
                        if not dry_run:
                            save_progress(all_examples, config.paths.output_dir, "progress_synthetic", config.version)
                
                except Exception as e:
                    log(f"  Batch {batch_idx + 1} failed: {e}", verbose)
        
        # Merge stats from generator (this is approximate since we used multiple instances)
        build_stats["synthetic"] = {
            "total_generated": synthetic_count,
            "note": "Generated using parallel workers"
        }
        log(f"  Generated {synthetic_count} synthetic examples", verbose)
        
        # Final save after synthetic generation
        if not dry_run and synthetic_count > 0:
            log(f"  Saving checkpoint after synthetic generation...", verbose)
            save_progress(all_examples, config.paths.output_dir, "checkpoint_after_synthetic", config.version)
    
    # =========================================================================
    # Phase 3: Deduplication
    # =========================================================================
    log("Phase 3: Deduplicating...", verbose)
    
    clean_examples, internal_examples = deduplicator.deduplicate_batch(all_examples)
    
    build_stats["deduplication"] = deduplicator.get_stats().to_dict()
    build_stats["contamination"] = contamination_router.get_stats().to_dict()
    
    log(f"  Clean: {len(clean_examples)}, Internal: {len(internal_examples)}", verbose)
    
    # =========================================================================
    # Phase 4: Split Train/Dev
    # =========================================================================
    log("Phase 4: Splitting train/dev...", verbose)
    
    split_result = split_pools(
        clean_examples=clean_examples,
        internal_examples=internal_examples,
        dev_ratio=0.1,
        db_quotas=config.db_quotas,
        failure_quotas=config.failure_quotas,
    )
    
    clean_train = split_result["clean_train"]
    clean_dev = split_result["clean_dev"]
    internal_train = split_result["internal_train"]
    internal_dev = split_result["internal_dev"]
    
    build_stats["splitting"] = {
        "clean": split_result["clean_stats"],
        "internal": split_result["internal_stats"],
    }
    
    log(f"  Clean: {len(clean_train)} train, {len(clean_dev)} dev", verbose)
    log(f"  Internal: {len(internal_train)} train, {len(internal_dev)} dev", verbose)
    
    # =========================================================================
    # Phase 5: Export
    # =========================================================================
    if not dry_run:
        log("Phase 5: Writing datasets...", verbose)
        
        writer = DatasetWriter(config.paths.output_dir, config.version)
        
        # Write main datasets
        writer.write_datasets(
            clean_train=clean_train,
            clean_dev=clean_dev,
            internal_train=internal_train,
            internal_dev=internal_dev,
        )
        
        # Write rejected
        writer.write_rejected(rejected_examples)
        
        # Write samples
        real_samples = [e for e in all_examples if e.metadata.source_type == "real_failure"]
        synth_samples = [e for e in all_examples if e.metadata.source_type == "synthetic_corruption"]
        
        writer.write_samples(
            real_failures=real_samples,
            synthetic=synth_samples,
            contrastive=[],
            hard_cases=[],
        )
        
        # Generate reports
        writer.generate_all_reports(
            clean_train=clean_train,
            clean_dev=clean_dev,
            internal_train=internal_train,
            internal_dev=internal_dev,
            config={
                "version": config.version,
                "build_size": config.build_size,
                "targets": asdict(config.targets),
            },
            build_stats=build_stats,
            contamination_report=contamination_router.generate_report(),
            dedup_report=deduplicator.generate_report(),
            verification_report=build_stats["verification"],
            subagent_report={"mock": True, "note": "Mock subagent used"},
        )
        
        log(f"  Wrote {len(writer.stats.files_written)} files", verbose)
    else:
        log("Dry run - skipping file writes", verbose)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    build_stats["elapsed_seconds"] = round(elapsed, 2)
    build_stats["end_time"] = datetime.now(timezone.utc).isoformat()
    
    log(f"Build complete in {elapsed:.1f}s", verbose)
    log(f"  Total examples: {len(all_examples)}", verbose)
    log(f"  Clean: {len(clean_train)} train + {len(clean_dev)} dev", verbose)
    log(f"  Internal: {len(internal_train)} train + {len(internal_dev)} dev", verbose)
    log(f"  Rejected: {len(rejected_examples)}", verbose)
    
    return build_stats


def main():
    args = parse_args()
    
    # Build config
    config = BuilderConfig(
        version=args.version,
        build_size=args.size,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    
    # Run build
    try:
        stats = build_dataset(
            config=config,
            dry_run=args.dry_run,
            limit_real=args.limit_real,
            limit_synthetic=args.limit_synthetic,
            skip_synthetic=args.skip_synthetic,
            skip_real=args.skip_real,
            num_workers=args.workers,
        )
        
        # Print summary
        if args.verbose:
            print("\nBuild Statistics:")
            print(json.dumps(stats, indent=2))
        
    except KeyboardInterrupt:
        print("\nBuild interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nBuild failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
