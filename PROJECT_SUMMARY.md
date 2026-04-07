# Project Summary

Generated on: 2026-04-07

## What We Built

- `10` dataset tracks under `data/training/`.
- `10` train JSONL files and `13` dev JSONL files tracked in current workspace.
- Multiple run families across 1.7B, 7B, and 14B experiments.
- `7` evaluation summary markdown files and `26` report JSON artifacts.

## Main Performance Journey

- Raw no-LoRA baseline (Qwen3-1.7B): `2.35% EX` (`36/1534`).
- Early LoRA baseline (1.7B): `34.75% EX`.
- T10 baseline: `38.07% EX`.
- T11.1 baseline: `35.01% EX`.
- T12 baseline (winner before repair): `40.94% EX` (`628/1534`).

## Error-Correction Models (Two Variants)

- Base repair model: `42` accepted repairs, effective `43.68% EX`.
- LoRA repair model: `45` accepted repairs, effective `43.87% EX`.

Current best in this repo:
- `43.87% EX` (`673/1534`) with `T12 + V2 LoRA error-correction`.

## Publishing Milestone

- Public HF model published: `Arnav3035/garuda-sql-2b`.
- Model card upgraded with full journey, usage, and benchmark context.
- PEFT base model path fixed to canonical `Qwen/Qwen3.5-2B`.

## Bottom Line

This project evolved from a low raw baseline into a structured, reproducible NL2SQL system with measurable gains through dataset iteration, LoRA training, and a two-model repair stage, ending at `43.87% EX` on BIRD dev.
