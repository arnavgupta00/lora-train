# T12 — Final Archetype-Targeted Dataset Upgrade

## Overview

T12 is the final dataset revision for the SQL generator before model scaling. It adds 2,050 new archetype-targeted examples while maintaining benchmark cleanliness.

## Key Features

- **Benchmark-clean**: No eval schemas used directly - all training examples generated on synthetic non-eval schemas
- **Archetype-targeted**: Examples designed to address 7 weak database families
- **Soft reweighting**: Per-example sampling weights instead of naive duplication
- **Full traceability**: Complete provenance tracking for all examples

## Files

| File | Description |
|------|-------------|
| `train_t12.jsonl` | Training data (16,084 examples) |
| `dev_t12.jsonl` | Dev set (unchanged from T10, 665 examples) |
| `t12_audit_set.jsonl` | Audit set for manual review (96 examples) |
| `t12_build_manifest.json` | Build metadata and configuration |
| `t12_archetype_summary.json` | Per-family statistics |
| `t12_sampling_plan.json` | Sampling weight configuration |
| `t12_source_map.jsonl` | Provenance for each T12 addition |
| `t12_acceptance_report.json` | Validation statistics |
| `schema_family_registry.json` | Non-eval schema registry |
| `generated/` | Raw generated examples and schemas |

## Composition

- **Backbone**: 14,034 examples (from T11_2)
- **T12 Additions**: 2,050 examples (archetype-targeted)

### T12 Additions by Family

| Family | Count | Schema Source |
|--------|------:|---------------|
| california_schools-type | 360 | education_programs |
| financial-type | 340 | banking_system |
| formula_1-type | 300 | motorsport_events |
| thrombosis_prediction-type | 260 | clinical_records |
| debit_card_specializing-type | 160 | retail_analytics |
| card_games-type | 180 | media_catalog |
| toxicology-type | 150 | materials_chemistry |
| stability-pack | 300 | mixed_stable |

### Example Type Mix

- 55% canonical
- 25% contrastive
- 20% output-discipline

## Sampling Weights

Per-example weights by targeted archetype:
- rank-by-X return-Y: 1.8
- derived ratio/formula: 1.8
- multi-hop join chain: 2.0
- table-family disambiguation: 1.8
- distinct entity vs row count: 1.7
- cohort/denominator percentage: 1.9
- date-anchor / temporal SQLite logic: 1.8
- side-table text/metadata return-field: 1.6
- graph traversal / molecule-level: 1.7

## Validation

All accepted examples passed:
- Execution validation (SQLite syntax check)
- Schema validation (valid tables/columns)
- Duplicate filtering (vs T10/T11)
- Contamination filtering (no eval schema overlap)

## Usage

```bash
# Training
python train.py --data data/training/t12/train_t12.jsonl

# With sampling weights
# Use the sampling_weight field from each example for weighted sampling
```

## Benchmark Cleanliness

This dataset maintains strict benchmark cleanliness:

- ✅ No exact eval schemas used
- ✅ No trivial rewrites of eval questions
- ✅ All examples on synthetic non-eval schemas
- ✅ Different identifiers from eval databases
- ✅ Same reasoning patterns, different schema surface

## Build Information

- Created: 2026-04-05
- Builder: build_t12.py
- Backbone: T11_2
