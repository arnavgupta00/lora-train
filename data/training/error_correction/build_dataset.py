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
from builder.contrastive_generator import ContrastiveGenerator
from builder.deduplicator import Deduplicator
from builder.deterministic_loader import DeterministicRepairLoader
from builder.metadata import Pool, RepairExample
from builder.real_failure_ingester import RealFailureIngester
from builder.schema_builder import SchemaBuilder
from builder.splitter import split_pools
from builder.subagent_client import MockSubagentClient
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
        "--limit-contrastive",
        type=int,
        default=None,
        help="Limit number of contrastive examples to generate",
    )
    parser.add_argument(
        "--limit-deterministic",
        type=int,
        default=None,
        help="Limit number of deterministic-fix examples to load",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic generation (for testing)",
    )
    parser.add_argument(
        "--skip-contrastive",
        action="store_true",
        help="Skip contrastive generation (for testing)",
    )
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Skip deterministic-fix ingestion (for testing)",
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
    parser.add_argument(
        "--parallel-backend",
        choices=["auto", "threads", "processes"],
        default="auto",
        help="Parallel backend for synthetic generation (default: auto)",
    )
    parser.add_argument(
        "--synthetic-chunk-size",
        type=int,
        default=None,
        help="Synthetic examples per work item; smaller values improve load balancing and progress reporting",
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=10.0,
        help="Abort individual SQLite queries that exceed this wall-clock time (default: 10s)",
    )
    parser.add_argument(
        "--dedup-workers",
        type=int,
        default=1,
        help="Parallel workers for Phase 5 dedup preprocessing (default: 1)",
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
    config, start_idx, count, seed, query_timeout_seconds = args
    started_at = time.time()
    
    # Set random seed for reproducibility within this worker
    random.seed(seed)
    
    # Create generator instance
    generator = SyntheticGenerator(
        config,
        query_timeout_seconds=query_timeout_seconds,
    )
    
    # Load gold sources
    train_file = config.paths.t10_data_dir / "train_t10.jsonl"
    dev_file = config.paths.bird_dev_prompts("t10")
    
    if train_file.exists():
        generator.load_gold_from_bird_train(train_file)
    
    if dev_file.exists():
        generator.load_gold_from_bird_dev(dev_file)

    generated_files = get_t12_generated_files(config, output_only=False)
    if generated_files:
        generator.load_gold_from_generated_files(generated_files, source_name="t12_generated")
    
    # Generate examples for this batch
    examples = []
    for example in generator.generate_batch(count=count, clean_only=False):
        examples.append(example)
        if len(examples) >= count:
            break
    
    return {
        "batch_idx": start_idx,
        "requested": count,
        "produced": len(examples),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "stats": generator.get_stats().to_dict(),
        "examples": examples,
    }


def generate_contrastive_batch_worker(args):
    """Worker function for parallel contrastive generation."""
    (
        config,
        start_idx,
        count,
        seed,
        query_timeout_seconds,
        clean_only,
        transform_names,
    ) = args
    started_at = time.time()

    random.seed(seed)

    generator = ContrastiveGenerator(
        config,
        query_timeout_seconds=query_timeout_seconds,
    )
    train_file = config.paths.train_file("t10")
    dev_file = config.paths.bird_dev_prompts("t10")

    if train_file.exists():
        generator.load_gold_from_bird_train(train_file)
    if dev_file.exists():
        generator.load_gold_from_bird_dev(dev_file)
    generated_files = get_t12_generated_files(config, output_only=False)
    if generated_files:
        generator.load_gold_from_generated_files(generated_files, source_name="t12_generated")

    examples = []
    for example in generator.generate_batch(
        count=count,
        clean_only=clean_only,
        transform_names=transform_names,
    ):
        examples.append(example)
        if len(examples) >= count:
            break

    return {
        "batch_idx": start_idx,
        "requested": count,
        "produced": len(examples),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "stats": generator.get_stats().to_dict(),
        "examples": examples,
    }


def resolve_parallel_backend(backend: str, num_workers: int) -> str:
    """Pick a safe default backend for the current worker count."""
    if backend != "auto":
        return backend
    return "processes" if num_workers > 4 else "threads"


def resolve_synthetic_chunk_size(
    target_synthetic: int,
    num_workers: int,
    requested_chunk_size: Optional[int],
) -> int:
    """Choose a chunk size that keeps workers busy while preserving visibility."""
    if requested_chunk_size is not None and requested_chunk_size > 0:
        return requested_chunk_size

    if target_synthetic <= 0:
        return 25

    auto_chunk = max(25, target_synthetic // max(1, num_workers * 4))
    return min(250, auto_chunk)


def resolve_build_targets(config: BuilderConfig) -> Dict[str, int]:
    """Resolve the requested A/B build into concrete totals."""
    train_target = config.targets.internal_train
    dev_target = config.targets.internal_dev
    total_target = train_target + dev_target
    return {
        "train_target": train_target,
        "dev_target": dev_target,
        "total_target": total_target,
    }


def get_t12_generated_files(config: BuilderConfig, output_only: bool = False) -> List[Path]:
    """Return generated T12 files that can act as clean, schema-inline sources."""
    generated_dir = config.paths.training_dir / "t12" / "generated"
    if not generated_dir.exists():
        return []

    excluded = {"t12_source_map.jsonl", "t12_audit_set.jsonl"}
    patterns = ["*output_discipline*.jsonl"] if output_only else ["*_raw*.jsonl", "*output_discipline*.jsonl"]
    files: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(generated_dir.glob(pattern)):
            if path.name in excluded or path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def build_dataset(
    config: BuilderConfig,
    dry_run: bool = False,
    limit_real: Optional[int] = None,
    limit_synthetic: Optional[int] = None,
    limit_contrastive: Optional[int] = None,
    limit_deterministic: Optional[int] = None,
    skip_synthetic: bool = False,
    skip_contrastive: bool = False,
    skip_deterministic: bool = False,
    skip_real: bool = False,
    num_workers: int = 4,
    parallel_backend: str = "auto",
    synthetic_chunk_size: Optional[int] = None,
    query_timeout_seconds: float = 10.0,
    dedup_workers: int = 1,
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
    resolved_targets = resolve_build_targets(config)
    
    # Initialize components
    log("Initializing components...", verbose)
    verifier = Verifier(
        config.paths,
        query_timeout_seconds=query_timeout_seconds,
    )
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
            "resolved_targets": resolved_targets,
        },
        "real_failures": {},
        "synthetic": {},
        "contrastive": {},
        "deterministic": {},
        "contamination": {},
        "deduplication": {},
        "splitting": {},
        "verification": {"passed": 0, "failed": 0},
    }

    def run_parallel_generation_phase(
        phase_name: str,
        target_count: int,
        kind: str,
        clean_only: bool = False,
        transform_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run either synthetic or contrastive generation in parallel."""
        if target_count <= 0:
            return {
                "backend": resolve_parallel_backend(parallel_backend, num_workers),
                "chunk_size": 0,
                "total_generated": 0,
                "total_attempted": 0,
                "failed_corruption": 0,
                "failed_verification": 0,
            }

        resolved_backend = resolve_parallel_backend(parallel_backend, num_workers)
        executor_cls = (
            ProcessPoolExecutor if resolved_backend == "processes" else ThreadPoolExecutor
        )
        chunk_size = resolve_synthetic_chunk_size(
            target_synthetic=target_count,
            num_workers=num_workers,
            requested_chunk_size=synthetic_chunk_size,
        )
        batches = []
        remaining = target_count
        batch_idx = 0
        while remaining > 0:
            count = min(chunk_size, remaining)
            seed = 42 + (1000 if kind == "contrastive" else 0) + batch_idx
            if kind == "contrastive":
                batches.append(
                    (
                        config,
                        batch_idx,
                        count,
                        seed,
                        query_timeout_seconds,
                        clean_only,
                        transform_names,
                    )
                )
            else:
                batches.append((config, batch_idx, count, seed, query_timeout_seconds))
            remaining -= count
            batch_idx += 1

        log(
            f"{phase_name}: generating {target_count} examples using {num_workers} workers "
            f"via {resolved_backend} in {len(batches)} chunks of up to {chunk_size}",
            verbose,
        )

        accepted = 0
        attempted = 0
        failed_corruption = 0
        failed_verification = 0
        next_checkpoint = 500
        phase_start = time.time()
        worker_fn = (
            generate_contrastive_batch_worker
            if kind == "contrastive"
            else generate_synthetic_batch_worker
        )

        with executor_cls(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_fn, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                future_batch_idx = futures[future]
                try:
                    batch_result = future.result()
                except Exception as exc:
                    log(f"  Chunk {future_batch_idx + 1} failed: {exc}", verbose)
                    continue

                batch_examples = batch_result["examples"]
                batch_stats = batch_result["stats"]
                attempted += batch_stats.get("total_attempted", 0)
                failed_corruption += batch_stats.get("failed_corruption", 0)
                failed_verification += batch_stats.get("failed_verification", 0)

                for example in batch_examples:
                    routing = contamination_router.tag_and_route(
                        example_id=example.metadata.example_id,
                        source_type=example.metadata.source_type,
                        source_run=example.metadata.source_run,
                        db_id=example.metadata.db_family,
                        reference_sql_source=example.metadata.reference_sql_source,
                        validation_passed=True,
                        reference_sql_valid=True,
                        has_required_fields=True,
                        parent_source=(
                            "bird_train" if example.metadata.benchmark_clean else "bird_dev"
                        ),
                    )
                    contamination_router.update_metadata(example.metadata, routing)
                    if routing.decision == RoutingDecision.REJECTED:
                        rejected_examples.append(example)
                        continue
                    all_examples.append(example)
                    accepted += 1

                elapsed = max(0.01, time.time() - phase_start)
                rate = accepted / elapsed
                remaining_examples = max(0, target_count - accepted)
                eta_seconds = remaining_examples / rate if rate > 0 else 0.0
                log(
                    "  Chunk "
                    f"{batch_result['batch_idx'] + 1}/{len(batches)} complete: "
                    f"{len(batch_examples)}/{batch_result['requested']} examples in "
                    f"{batch_result['elapsed_seconds']:.1f}s "
                    f"({accepted}/{target_count} accepted, {attempted} attempted, "
                    f"{failed_verification} verify fails, ETA {eta_seconds / 60:.1f} min)",
                    verbose,
                )

                if accepted >= next_checkpoint or accepted >= target_count:
                    if not dry_run:
                        prefix = f"progress_{kind}"
                        save_progress(all_examples, config.paths.output_dir, prefix, config.version)
                    while accepted >= next_checkpoint:
                        next_checkpoint += 500

        return {
            "backend": resolved_backend,
            "chunk_size": chunk_size,
            "total_generated": accepted,
            "total_attempted": attempted,
            "failed_corruption": failed_corruption,
            "failed_verification": failed_verification,
            "clean_only": clean_only,
            "transform_names": transform_names or [],
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
            example.metadata.subagent_candidate_sql = failure.gold_sql
            example.metadata.subagent_accept_reason = "Gold-aligned fallback repair"
            
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
    
    target_total = resolved_targets["total_target"]
    target_contrastive = (
        limit_contrastive if limit_contrastive is not None else int(target_total * 0.18)
    )
    target_deterministic = limit_deterministic if limit_deterministic is not None else 250
    target_synthetic = limit_synthetic
    if target_synthetic is None:
        target_synthetic = max(
            int(target_total * config.targets.synthetic_min),
            target_total - len(all_examples) - target_contrastive - target_deterministic + int(target_total * 0.12),
        )

    # =========================================================================
    # Phase 2: Synthetic Generation
    # =========================================================================
    if not skip_synthetic:
        build_stats["synthetic"] = run_parallel_generation_phase(
            phase_name="Phase 2",
            target_count=target_synthetic,
            kind="synthetic",
            clean_only=False,
        )
        log(
            f"  Generated {build_stats['synthetic']['total_generated']} synthetic examples",
            verbose,
        )
        if not dry_run and build_stats["synthetic"]["total_generated"] > 0:
            save_progress(
                all_examples,
                config.paths.output_dir,
                "checkpoint_after_synthetic",
                config.version,
            )

    # =========================================================================
    # Phase 3: Deterministic Repair Ingestion
    # =========================================================================
    if not skip_deterministic:
        log("Phase 3: Loading deterministic repairs...", verbose)
        deterministic_loader = DeterministicRepairLoader(config, verifier=verifier)
        deterministic_examples = deterministic_loader.load_examples(
            limit=target_deterministic,
        )
        for example in deterministic_examples:
            routing = contamination_router.tag_and_route(
                example_id=example.metadata.example_id,
                source_type=example.metadata.source_type,
                source_run=example.metadata.source_run,
                db_id=example.metadata.db_family,
                reference_sql_source=example.metadata.reference_sql_source,
                validation_passed=True,
                reference_sql_valid=True,
                has_required_fields=True,
                parent_source="bird_dev",
            )
            contamination_router.update_metadata(example.metadata, routing)
            if routing.decision == RoutingDecision.REJECTED:
                rejected_examples.append(example)
                continue
            all_examples.append(example)
        build_stats["deterministic"] = deterministic_loader.get_stats().to_dict()
        log(
            f"  Loaded {len(deterministic_examples)} deterministic-fix examples",
            verbose,
        )

    # =========================================================================
    # Phase 4: Contrastive / Output-Discipline Generation
    # =========================================================================
    if not skip_contrastive:
        transform_names = [pattern.transform_name for pattern in ContrastiveGenerator(config).get_patterns()]
        build_stats["contrastive"] = run_parallel_generation_phase(
            phase_name="Phase 4",
            target_count=target_contrastive,
            kind="contrastive",
            clean_only=True,
            transform_names=transform_names,
        )
        log(
            f"  Generated {build_stats['contrastive']['total_generated']} contrastive examples",
            verbose,
        )
        if not dry_run and build_stats["contrastive"]["total_generated"] > 0:
            save_progress(
                all_examples,
                config.paths.output_dir,
                "checkpoint_after_contrastive",
                config.version,
            )

    # =========================================================================
    # Phase 5: Deduplication
    # =========================================================================
    log("Phase 5: Deduplicating...", verbose)
    if dedup_workers > 1:
        log(f"  Dedup preprocessing in parallel with {dedup_workers} workers", verbose)

    dedup_start = time.time()

    def log_dedup_progress(processed: int, total: int, stats_obj) -> None:
        elapsed = max(0.01, time.time() - dedup_start)
        rate = processed / elapsed
        eta = (total - processed) / rate if rate > 0 else 0.0
        log(
            "  Dedup progress: "
            f"{processed}/{total} processed, "
            f"{stats_obj.total_output} kept, "
            f"{stats_obj.total_input - stats_obj.total_output} removed, "
            f"ETA {eta / 60:.1f} min",
            verbose,
        )

    clean_examples, internal_examples = deduplicator.deduplicate_batch(
        all_examples,
        progress_every=1000,
        progress_callback=log_dedup_progress,
        prepare_workers=max(1, dedup_workers),
    )
    
    build_stats["deduplication"] = deduplicator.get_stats().to_dict()
    build_stats["contamination"] = contamination_router.get_stats().to_dict()
    
    log(f"  Clean: {len(clean_examples)}, Internal: {len(internal_examples)}", verbose)
    
    # =========================================================================
    # Phase 6: Split Train/Dev
    # =========================================================================
    log("Phase 6: Splitting train/dev...", verbose)
    
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
    # Phase 7: Export
    # =========================================================================
    if not dry_run:
        log("Phase 7: Writing datasets...", verbose)
        
        writer = DatasetWriter(config.paths.output_dir, config.version)
        
        # Write main datasets
        log("  Writing train/dev JSONL files...", verbose)
        writer.write_datasets(
            clean_train=clean_train,
            clean_dev=clean_dev,
            internal_train=internal_train,
            internal_dev=internal_dev,
        )
        
        # Write rejected
        log("  Writing rejected/internal-only files...", verbose)
        writer.write_rejected(rejected_examples)
        writer.write_internal_only(internal_examples)
        
        # Write samples
        log("  Preparing sample files...", verbose)
        real_samples = [e for e in all_examples if e.metadata.source_type == "real_failure"]
        synth_samples = [e for e in all_examples if e.metadata.source_type == "synthetic_corruption"]
        contrastive_samples = [e for e in all_examples if e.metadata.source_type == "contrastive"]
        hard_case_samples = [
            e for e in all_examples
            if e.metadata.source_type in ("deterministic_fix", "manual_fix")
            or e.metadata.failure_type in (
                "wrong_denominator",
                "join_path_error",
                "temporal_anchor_error",
                "table_family_confusion",
            )
        ]
        
        log("  Writing sample files...", verbose)
        writer.write_samples(
            real_failures=real_samples,
            synthetic=synth_samples,
            contrastive=contrastive_samples,
            hard_cases=hard_case_samples,
        )
    else:
        log("Dry run - skipping file writes", verbose)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    build_stats["elapsed_seconds"] = round(elapsed, 2)
    build_stats["end_time"] = datetime.now(timezone.utc).isoformat()
    build_stats["verification"]["passed"] += (
        build_stats["synthetic"].get("total_generated", 0)
        + build_stats["contrastive"].get("total_generated", 0)
        + build_stats["deterministic"].get("total_accepted", 0)
    )
    build_stats["verification"]["failed"] += (
        build_stats["synthetic"].get("failed_verification", 0)
        + build_stats["contrastive"].get("failed_verification", 0)
    )

    if not dry_run:
        log("  Generating final reports...", verbose)
        writer.generate_all_reports(
            clean_train=clean_train,
            clean_dev=clean_dev,
            internal_train=internal_train,
            internal_dev=internal_dev,
            config={
                "version": config.version,
                "build_size": config.build_size,
                "targets": asdict(config.targets),
                "resolved_targets": resolved_targets,
            },
            build_stats=build_stats,
            contamination_report=contamination_router.generate_report(),
            dedup_report=deduplicator.generate_report(max_cross_pool_checks=200000),
            verification_report=build_stats["verification"],
            subagent_report=subagent.get_stats().to_dict() if not skip_real else {},
        )
        log(f"  Wrote {len(writer.stats.files_written)} files", verbose)
    
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
    config.apply_build_size()
    
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
            limit_contrastive=args.limit_contrastive,
            limit_deterministic=args.limit_deterministic,
            skip_synthetic=args.skip_synthetic,
            skip_contrastive=args.skip_contrastive,
            skip_deterministic=args.skip_deterministic,
            skip_real=args.skip_real,
            num_workers=args.workers,
            parallel_backend=args.parallel_backend,
            synthetic_chunk_size=args.synthetic_chunk_size,
            query_timeout_seconds=args.query_timeout_seconds,
            dedup_workers=args.dedup_workers,
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
