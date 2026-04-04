# T11.1 Training, Prediction, and Evaluation

`t11_1` is self-contained like `t10`: dataset builder, train launcher, prediction script, evaluation script, and local prompt utilities live in this folder.

## Files

- `train_t11_1.jsonl`: training set
- `dev_t11_1.jsonl`: dev set
- `t11_1_utils.py`: prompt helpers / validation
- `build_eval_prompts.py`: builds `bird_dev_t11_1.jsonl`
- `predict_t11_1.py`: runs inference on T11.1 prompts
- `evaluate_t11_1.py`: evaluates predictions against BIRD
- `train_t11_1.sh`: training launcher

## Build Eval Prompts

```bash
python data/training/t11_1/build_eval_prompts.py \
  --full_prompts_file data/training/t10/bird_dev_t10.jsonl \
  --output data/training/t11_1/bird_dev_t11_1.jsonl
```

This writes:

- `data/training/t11_1/bird_dev_t11_1.jsonl`
- `data/training/t11_1/bird_dev_t11_1.stats.json`

## Train

```bash
bash data/training/t11_1/train_t11_1.sh \
  --config training/configs/t11_1_baseline_3090.yaml
```

## Predict

```bash
python data/training/t11_1/predict_t11_1.py \
  --model_id "Qwen/Qwen3-1.7B" \
  --adapter_dir "./runs/t11_1_baseline_3090" \
  --prompts_file data/training/t11_1/bird_dev_t11_1.jsonl \
  --output_dir "./runs/t11_1_baseline_3090/predictions"
```

This writes:

- `predictions_t11_1.jsonl`
- `predictions_t11_1.json`
- `raw_outputs_t11_1.jsonl`
- `generation_config.json`

## Evaluate

```bash
python data/training/t11_1/evaluate_t11_1.py \
  --predictions_file ./runs/t11_1_baseline_3090/predictions/predictions_t11_1.jsonl \
  --output_dir ./runs/t11_1_baseline_3090/eval
```

This writes:

- `eval_report_t11_1.json`
- `eval_summary_t11_1.md`
- `per_example_results.jsonl`
