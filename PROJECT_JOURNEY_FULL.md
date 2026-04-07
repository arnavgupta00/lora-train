# LM Project Journey (Full Report)

Generated on: 2026-04-07
Workspace: `/Users/arnav/programming/lm`

## 1) Executive Snapshot

This project is a full text-to-SQL engineering program focused on BIRD dev-set performance improvement through:
- dataset iteration (`t2` -> `t12`),
- baseline and ablation runs,
- prompt contract hardening,
- LoRA adaptation,
- and a second-stage error-correction pipeline.

Current best result in this repo:
- `43.87% EX` (`673/1534`) using `T12 + V2 error-correction (LoRA repair model)`.

Strong baseline before repair:
- `40.94% EX` (`628/1534`) on T12 without-sampling.

## 2) What Was Built

### 2.1 Dataset Tracks Created

Top-level dataset tracks under `data/training/`: `10`
- `error_correction`, `t2`, `t7`, `t8`, `t9`, `t10`, `t11`, `t11_1`, `t11_2`, `t12`

Counted dataset files:
- Train JSONL files matching `*train*.jsonl`: `10`
- Dev JSONL files matching `*dev*.jsonl`: `13`

Examples:
- `data/training/t10/train_t10.jsonl`
- `data/training/t11_1/train_t11_1.jsonl`
- `data/training/t12/train_t12.jsonl`
- `data/training/error_correction/train_error_repair_v1_clean.jsonl`
- `data/training/t12/bird_dev_t12.jsonl`

### 2.2 Run Families and Artifacts

Major run families under `runs/` include:
- `runs/~1-2B/qwen3-1.7b`
- `runs/~7-8B/qwen2.5-7b`
- `runs/10-14B/qwen2.5-14b`
- `runs/t10_baseline_3090/qwen3-1.7b`
- `runs/t11_1_baseline_3090/qwen3-1.7b`
- `runs/t12_baseline_3090/without-sampling`
- `runs/t12_baseline_3090/error-correction-v2-base-qwen-3.5-2B`
- `runs/t12_baseline_3090/error-correction-v2-lora-qwen-3.52B`
- `runs/exported_artifacts/t10_lora_greedy`
- `runs/exported_artifacts/t11_1_lora_greedy`

Counted reporting artifacts:
- `eval_summary*.md`: `7`
- `eval_report*.json` / `evaluation_report*.json` / `repair_summary*.json`: `26`

## 3) Performance Timeline (How We Reached Here)

### 3.1 Early and Raw Baselines

Raw no-LoRA baseline (Qwen3-1.7B):
- `2.35% EX` (`36/1534`)
- Source: `runs/t10_baseline_3090/qwen3-1.7b/without-lora/eval/eval_summary_t10.md`

Earlier LoRA checkpoint baseline:
- `34.75% EX` (`533/1534`)
- Source: `runs/~1-2B/qwen3-1.7b/v1/eval_2_sft_lora/evaluation_report.json`

Historical 7B milestone:
- `44.26% EX` on prior Qwen2.5-7B run family
- Source: `runs/~7-8B/qwen2.5-7b/v1-3/RUN_LOG.md`

### 3.2 T10 -> T11.1 -> T12 Progression

T10 (`without-sampling`):
- `38.07% EX` (`584/1534`)
- Source: `runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_summary_t10.md`

T11.1 (`without-sampling`):
- `35.01% EX` (`537/1534`)
- Source: `runs/t11_1_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_summary_t11_1.md`

T12 (`without-sampling`) winner baseline:
- `40.94% EX` (`628/1534`)
- `13.49% EM` (`207/1534`)
- Source: `runs/t12_baseline_3090/without-sampling/eval/eval_summary_t12.md`

Interpretation:
- T11.1 showed a measurable dip vs T10.
- T12 recovered and exceeded T10 significantly.
- T12 also reduced execution-failure rate compared with earlier runs.

## 4) Error-Correction Program (Two Models)

You trained and evaluated two dedicated V2 repair variants on top of T12 outputs:

1. Base repair model
- Path: `runs/t12_baseline_3090/error-correction-v2-base-qwen-3.5-2B/repair_summary_t12.json`
- Repair accepted: `42` out of `906` attempts
- Resulting corrected total: `628 + 42 = 670`
- Effective EX: `43.68%`

2. LoRA repair model
- Path: `runs/t12_baseline_3090/error-correction-v2-lora-qwen-3.52B/repair_summary_t12.json`
- Repair accepted: `45` out of `906` attempts
- Resulting corrected total: `628 + 45 = 673`
- Effective EX: `43.87%` (current best in this repo)

Key outcome:
- The LoRA repair model improved over base repair by `+3` accepted fixes.
- This was the final step that pushed performance to the current best.

## 5) Publishing and Professionalization Milestones

A production-style HF release was completed for this work:
- Model repo: `https://huggingface.co/Arnav3035/garuda-sql-2b`
- Adapter metadata fixed to canonical base model ID: `Qwen/Qwen3.5-2B`
- Model card upgraded from template to full professional documentation
- Dev-set benchmark context corrected and aligned to visible BIRD dev entries

Model card now includes:
- progression story,
- base/no-LoRA baseline,
- T10/T11.1/T12 milestones,
- error-correction winner setup,
- usage instructions,
- benchmark deltas vs visible BIRD dev references.

## 6) Complete Achievement List

- Built a multi-version dataset pipeline from `t2` through `t12`.
- Produced a dedicated error-correction training dataset pipeline.
- Ran multiple ablations: with-sampling, without-sampling, without-lora.
- Improved from `2.35% EX` raw no-LoRA to `40.94% EX` T12 baseline.
- Trained two error-correction model variants and reached `43.87% EX` best.
- Standardized prompt/eval pipeline around T12 artifacts.
- Published a public HF adapter with a polished model card.
- Added transparent benchmark context and reproducible report references.

## 7) Where to Continue Next

- Push T12 baseline from `40.94%` to `45%+` before repair.
- Increase accepted repairs while keeping strict execution-match acceptance.
- Focus on weakest databases (`california_schools`, `financial`, `thrombosis_prediction`).
- Add an automated regression dashboard that tracks EX/EM/failure types per run.

---

This file is intended as the complete project record: what was built, what was tried, what worked, and how the current best score was reached.
