# RABR

RABR and its baselines: POG, BUNT, BGN, BYOB, Zero-shot, LLM4B, Text2Bundle, AICL, KD4BG and DDBC.

## Install

Use Python 3.11. GPU experiments need CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
git clone https://github.com/BundleRec/bundle_recommendation.git data/raw/BundleRec
git -C data/raw/BundleRec checkout 087da2e2b4d8738ef23f3f6f2005505e8685edde
python scripts/prepare_data.py \
  --source data/raw/BundleRec/dataset --output data/processed \
  --include-pretrain-history --require-history
```

## Settings

Edit `configs/rq1.json` through `configs/rq4.json` to choose methods, domains, seeds and parameter values. Method settings are in `configs/methods/`; model settings are in `configs/backbones/`.

Replace the `${...}` paths in those files, or export the corresponding environment variables. Set local language-model and encoder paths, plus the inputs needed by each selected baseline:

| Method | Additional inputs |
| --- | --- |
| POG | Image, text and collaborative features |
| BUNT | Item tags and historical bundles |
| BYOB | Source checkout, user interactions and candidate pools |
| Text2Bundle | LightGCN, FM and RoBERTa features |
| AICL | Two evaluator models |
| KD4BG | Teacher and student models |
| DDBC | CLHE features |

## Run

```bash
python scripts/prepare_experiments.py \
  --config configs/rq1.json configs/rq2.json configs/rq3.json configs/rq4.json \
  --data data/processed --output results/experiments \
  --backbone-config llama31-8b=configs/backbones/llama31-8b.json \
  --backbone-config qwen25-7b=configs/backbones/qwen25-7b.json \
  --backbone-config default=configs/backbones/default.json

python scripts/run_experiments.py \
  --queue results/experiments/queue.json --gpus 0,1 --cpu-workers 0 --cpu-threads 8

python scripts/evaluate.py \
  --queue results/experiments/queue.json --output results/experiments/summary.json
```

Pass fewer RQ files to run a subset. Rerun the same queue command to resume; completed jobs are skipped. Use a new output directory after changing code, data or settings.

Predictions and metrics are saved under `results/experiments/runs/`; the combined results are in `summary.json`.
