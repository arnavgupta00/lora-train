#!/usr/bin/env python3
"""
Writer Module

Export final JSONL datasets and generate reports.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metadata import RepairExample


@dataclass
class WriterStats:
    """Statistics from writing."""
    
    files_written: List[str] = field(default_factory=list)
    examples_written: Dict[str, int] = field(default_factory=dict)
    reports_generated: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_written": self.files_written,
            "examples_written": self.examples_written,
            "reports_generated": self.reports_generated,
            "total_examples": sum(self.examples_written.values()),
        }


class DatasetWriter:
    """Writer for error-correction datasets."""
    
    def __init__(self, output_dir: Path, version: str = "v1"):
        """
        Args:
            output_dir: Output directory for datasets
            version: Version string for filenames
        """
        self.output_dir = Path(output_dir)
        self.version = version
        self.stats = WriterStats()
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
    
    def _write_jsonl(
        self,
        examples: List[RepairExample],
        filename: str,
    ) -> int:
        """Write examples to JSONL file."""
        filepath = self.output_dir / filename
        
        count = 0
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in examples:
                data = ex.to_sft_format()
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
        
        self.stats.files_written.append(str(filepath))
        self.stats.examples_written[filename] = count
        
        return count
    
    def _write_json(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> None:
        """Write JSON report file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.stats.reports_generated.append(str(filepath))
    
    def write_datasets(
        self,
        clean_train: List[RepairExample],
        clean_dev: List[RepairExample],
        internal_train: List[RepairExample],
        internal_dev: List[RepairExample],
    ) -> None:
        """Write all dataset files."""
        # Internal (full) datasets
        self._write_jsonl(
            internal_train + clean_train,
            f"train_error_repair_{self.version}.jsonl",
        )
        self._write_jsonl(
            internal_dev + clean_dev,
            f"dev_error_repair_{self.version}.jsonl",
        )
        
        # Clean (benchmark-safe) datasets
        self._write_jsonl(
            clean_train,
            f"train_error_repair_{self.version}_clean.jsonl",
        )
        self._write_jsonl(
            clean_dev,
            f"dev_error_repair_{self.version}_clean.jsonl",
        )
    
    def write_rejected(
        self,
        rejected: List[RepairExample],
    ) -> None:
        """Write rejected examples."""
        self._write_jsonl(rejected, "rejected_examples.jsonl")
    
    def write_internal_only(
        self,
        internal_only: List[RepairExample],
    ) -> None:
        """Write internal-only examples (for reference)."""
        self._write_jsonl(internal_only, "internal_only_examples.jsonl")
    
    def write_samples(
        self,
        real_failures: List[RepairExample],
        synthetic: List[RepairExample],
        contrastive: List[RepairExample],
        hard_cases: List[RepairExample],
        sample_size: int = 50,
    ) -> None:
        """Write sample files for inspection."""
        self._write_jsonl(
            real_failures[:sample_size],
            "samples/real_failures_examples.jsonl",
        )
        self._write_jsonl(
            synthetic[:sample_size],
            "samples/synthetic_examples.jsonl",
        )
        self._write_jsonl(
            contrastive[:sample_size],
            "samples/contrastive_examples.jsonl",
        )
        self._write_jsonl(
            hard_cases[:sample_size],
            "samples/hard_cases_examples.jsonl",
        )
    
    def generate_manifest(
        self,
        config: Dict[str, Any],
        build_stats: Dict[str, Any],
    ) -> None:
        """Generate dataset manifest."""
        manifest = {
            "name": f"SQL Error-Correction Dataset {self.version}",
            "version": self.version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "description": "Supervised dataset for SQL error-correction model",
            "task": "Given broken SQL, output corrected SQL",
            "files": {
                "internal_train": f"train_error_repair_{self.version}.jsonl",
                "internal_dev": f"dev_error_repair_{self.version}.jsonl",
                "clean_train": f"train_error_repair_{self.version}_clean.jsonl",
                "clean_dev": f"dev_error_repair_{self.version}_clean.jsonl",
            },
            "statistics": self.stats.examples_written,
            "config": config,
            "build_stats": build_stats,
        }
        
        self._write_json(manifest, "dataset_manifest.json")
    
    def generate_family_summary(
        self,
        train: List[RepairExample],
        dev: List[RepairExample],
    ) -> None:
        """Generate database family summary."""
        summary: Dict[str, Dict[str, int]] = {}
        
        for ex in train:
            db = ex.metadata.db_family
            if db not in summary:
                summary[db] = {"train": 0, "dev": 0}
            summary[db]["train"] += 1
        
        for ex in dev:
            db = ex.metadata.db_family
            if db not in summary:
                summary[db] = {"train": 0, "dev": 0}
            summary[db]["dev"] += 1
        
        self._write_json({"by_database_family": summary}, "family_summary.json")
    
    def generate_failure_type_summary(
        self,
        train: List[RepairExample],
        dev: List[RepairExample],
    ) -> None:
        """Generate failure type summary."""
        summary: Dict[str, Dict[str, int]] = {}
        
        for ex in train:
            ft = ex.metadata.failure_type
            if ft not in summary:
                summary[ft] = {"train": 0, "dev": 0}
            summary[ft]["train"] += 1
        
        for ex in dev:
            ft = ex.metadata.failure_type
            if ft not in summary:
                summary[ft] = {"train": 0, "dev": 0}
            summary[ft]["dev"] += 1
        
        self._write_json({"by_failure_type": summary}, "failure_type_summary.json")
    
    def generate_source_mix_summary(
        self,
        train: List[RepairExample],
        dev: List[RepairExample],
    ) -> None:
        """Generate source mix summary."""
        summary: Dict[str, Dict[str, int]] = {}
        
        all_examples = train + dev
        
        for ex in all_examples:
            st = ex.metadata.source_type
            split = ex.metadata.split
            
            if st not in summary:
                summary[st] = {"train": 0, "dev": 0, "total": 0}
            
            summary[st][split] += 1
            summary[st]["total"] += 1
        
        # Calculate percentages
        total = len(all_examples)
        for st in summary:
            summary[st]["percentage"] = round(100 * summary[st]["total"] / max(1, total), 2)
        
        self._write_json({"by_source_type": summary}, "source_mix_summary.json")
    
    def generate_schema_context_summary(
        self,
        train: List[RepairExample],
        dev: List[RepairExample],
    ) -> None:
        """Generate schema context summary."""
        summary: Dict[str, int] = {}
        
        for ex in train + dev:
            sct = ex.metadata.schema_context_type
            summary[sct] = summary.get(sct, 0) + 1
        
        total = sum(summary.values())
        percentages = {k: round(100 * v / max(1, total), 2) for k, v in summary.items()}
        
        self._write_json({
            "by_schema_type": summary,
            "percentages": percentages,
        }, "schema_context_summary.json")
    
    def write_contamination_report(
        self,
        report: Dict[str, Any],
    ) -> None:
        """Write contamination report."""
        self._write_json(report, "contamination_report.json")
    
    def write_duplicate_report(
        self,
        report: Dict[str, Any],
    ) -> None:
        """Write duplicate report."""
        self._write_json(report, "duplicate_report.json")
    
    def write_verification_report(
        self,
        report: Dict[str, Any],
    ) -> None:
        """Write verification report."""
        self._write_json(report, "verification_report.json")
    
    def write_subagent_usage_report(
        self,
        report: Dict[str, Any],
    ) -> None:
        """Write subagent usage report."""
        self._write_json(report, "subagent_usage_report.json")
    
    def generate_readme(
        self,
        total_train: int,
        total_dev: int,
        clean_train: int,
        clean_dev: int,
    ) -> None:
        """Generate README.md for the dataset."""
        readme = f"""# SQL Error-Correction Dataset {self.version}

## Overview

Supervised dataset for training SQL error-correction models.

**Task**: Given schema, question, hints, broken SQL, and optional error message,
output the corrected SQL query only.

## Statistics

| Split | Count |
|-------|-------|
| Internal Train | {total_train} |
| Internal Dev | {total_dev} |
| Clean Train | {clean_train} |
| Clean Dev | {clean_dev} |

## Files

### Main Datasets

- `train_error_repair_{self.version}.jsonl` - Full training set (internal)
- `dev_error_repair_{self.version}.jsonl` - Full dev set (internal)
- `train_error_repair_{self.version}_clean.jsonl` - Benchmark-safe training set
- `dev_error_repair_{self.version}_clean.jsonl` - Benchmark-safe dev set

### Reports

- `dataset_manifest.json` - Build configuration and statistics
- `family_summary.json` - Examples by database family
- `failure_type_summary.json` - Examples by failure type
- `source_mix_summary.json` - Examples by source type
- `schema_context_summary.json` - Examples by schema context type
- `contamination_report.json` - Contamination routing decisions
- `duplicate_report.json` - Deduplication statistics
- `verification_report.json` - Verification results
- `subagent_usage_report.json` - Subagent proposal usage and outcomes

### Additional Files

- `rejected_examples.jsonl` - Examples that failed validation
- `internal_only_examples.jsonl` - Internal-only examples before clean filtering
- `samples/` - Sample examples for inspection

## Example Format

```json
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "Schema:\\n...\\n\\nHints:\\n...\\n\\nQuestion:\\n...\\n\\nBroken SQL:\\n..."}},
    {{"role": "assistant", "content": "SELECT ..."}}
  ],
  "metadata": {{...}}
}}
```

## Contamination Policy

- **Internal datasets**: May contain examples derived from BIRD dev/eval
- **Clean datasets**: Only non-benchmark sources (BIRD train, Spider, custom)

Clean datasets are safe for use when evaluating on BIRD benchmark.

## Usage

```python
import json

with open('train_error_repair_{self.version}_clean.jsonl') as f:
    for line in f:
        example = json.loads(line)
        messages = example['messages']
        metadata = example['metadata']
```
"""
        
        filepath = self.output_dir / "README.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        self.stats.files_written.append(str(filepath))
    
    def generate_all_reports(
        self,
        clean_train: List[RepairExample],
        clean_dev: List[RepairExample],
        internal_train: List[RepairExample],
        internal_dev: List[RepairExample],
        config: Dict[str, Any],
        build_stats: Dict[str, Any],
        contamination_report: Dict[str, Any],
        dedup_report: Dict[str, Any],
        verification_report: Dict[str, Any],
        subagent_report: Dict[str, Any],
    ) -> None:
        """Generate all reports."""
        all_train = internal_train + clean_train
        all_dev = internal_dev + clean_dev
        
        self.generate_manifest(config, build_stats)
        self.generate_family_summary(all_train, all_dev)
        self.generate_failure_type_summary(all_train, all_dev)
        self.generate_source_mix_summary(all_train, all_dev)
        self.generate_schema_context_summary(all_train, all_dev)
        self.write_contamination_report(contamination_report)
        self.write_duplicate_report(dedup_report)
        self.write_verification_report(verification_report)
        self.write_subagent_usage_report(subagent_report)
        self.generate_readme(
            total_train=len(all_train),
            total_dev=len(all_dev),
            clean_train=len(clean_train),
            clean_dev=len(clean_dev),
        )
    
    def get_stats(self) -> WriterStats:
        """Get writer statistics."""
        return self.stats
