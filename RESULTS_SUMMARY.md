# Results Summary (2026-03-28)

This repo is a small experiment: can a LoRA-tuned open model beat its own base model on a strict NL2SQL execution benchmark?

## Benchmark

- Task: `Schema + business rules + question -> SQL` (sqlite)
- Metric: **execution-match** (runs SQL on a seeded DB and compares result hashes)
- Test set: `90` examples across multiple domain schemas

## Headline Numbers

**Qwen2.5-14B-Instruct**

- Base: `11/90` = `12.2%` execution-match
- LoRA: `22/90` = `24.4%` execution-match
- Delta: `+11` correct, `+12.2pp`, `2.0x` relative

**Qwen2.5-Coder-32B-Instruct**

- Base: `16/90` = `17.8%` execution-match
- LoRA: `21/90` = `23.3%` execution-match
- Delta: `+5` correct, `+5.6pp`, `1.31x` relative

## Notes

- `exact_match` is near-zero because it requires the predicted SQL string to match the gold SQL almost exactly.
- Execution-match is strict: different column aliases or column order can fail even when the underlying rows/values are correct.

Full write-up: `reports/2026-03-28_run.md`.

