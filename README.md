# RABR

Code for risk-adaptive bundle recommendation and a shared evaluation of POG, BUNT, BGN, BYOB, Zero-shot, LLM4B, Text2Bundle, AICL, KD4BG, and DDBC. The experiment suites cover recommendation quality, empirical risk, component ablations, and parameter sensitivity.

This repository reconstructs the methods on a documented chronological BundleRec protocol. Original manuscript splits and trained checkpoints are not distributed. Generated results describe this protocol; they do not reproduce the manuscript's numerical tables. Method adaptations and source references are recorded in [traditional methods](docs/traditional-methods.md), [generative methods](docs/generative-methods.md), and [RABR protocol](docs/rabr-protocol.md).

## Implementation scope

RABR follows Equation 8 with candidate feedback between rounds, history-only relevance, finite-grid calibration and matching component ablations. RQ1 size/budget dependencies and paired seed-level significance analysis are implemented. A failed certificate remains a failed certificate.

The learned baselines require their actual feature and interaction inputs. POG requires image/TextCNN/collaborative features; BUNT requires native tags and historical bundles; BYOB requires its pinned source, user interactions and candidate pools; Text2Bundle requires RoBERTa/LightGCN/FM assets; DDBC requires CLHE features and supplied seed items for its native path. Those assets are not silently replaced with hashes, generic text vectors or intent proxies. See [paper alignment](docs/paper-alignment.md) for the evaluation adapters and remaining source/data limitations.

Small integration checks establish execution, gradient flow and recovery, not model convergence or equivalence to unavailable author checkpoints. BUNT feedback-assisted and DDBC supplied-seed results remain separate from query-only comparisons.

## Install

Use Python 3.11 on Linux for the full method suite. CUDA is recommended for experiment profiles. Data preparation and evaluation also work on macOS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[models,diffusion,distillation,quantization]'
```

The `distillation` extra adds PEFT for KD4BG student adaptation. It can be omitted for configurations that do not train a student. Model loading uses local files by default. No model weights or research data are included.

## Prepare data

```bash
git clone https://github.com/BundleRec/bundle_recommendation.git data/raw/BundleRec
git -C data/raw/BundleRec checkout 087da2e2b4d8738ef23f3f6f2005505e8685edde
python scripts/prepare_data.py --source data/raw/BundleRec/dataset --output data/processed --mode raw --include-pretrain-history --require-history
```

This command selects the history-eligible cohort before splitting and records that choice. The preparation command validates catalog IDs, separates whole timestamp groups, mines patterns from training targets, and writes hashes for every input file. Training, validation, calibration, and test are separate partitions. See [reproduction instructions](docs/reproduce.md) for source layouts, prepared-data import, output-size choices, queue recovery, and aggregation.

## Configure models

Provide local snapshots of Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and all-MiniLM-L6-v2:

```bash
export RABR_LLAMA_PATH=/path/to/Llama-3.1-8B-Instruct
export RABR_QWEN_PATH=/path/to/Qwen2.5-7B-Instruct
export RABR_ENCODER_PATH=/path/to/all-MiniLM-L6-v2
export RABR_STUDENT_MODEL_PATH=/path/to/student-instruct-model
export RABR_KD_TEACHER_PATH=/path/to/teacher-instruct-model
export RABR_AICL_RATER1_PATH=/path/to/first-evaluator-model
export RABR_AICL_RATER2_PATH=/path/to/different-evaluator-model
```

Configure the native asset manifests described in the method documents before launching the full suite. The backbone files in `configs/backbones/` set temperature, token limits, precision, and device. Snapshot provenance should be retained alongside downloaded weights. A run records the local file inventory; the inventory is not a cryptographic hash of model weights.

## Run experiments

```bash
python scripts/build_suite.py \
  --suite configs/suites/rq1.json configs/suites/rq2.json configs/suites/rq3.json configs/suites/rq4.json \
  --data data/processed --output results/all-rqs \
  --backbone-config llama31-8b=configs/backbones/llama31-8b.json \
  --backbone-config qwen25-7b=configs/backbones/qwen25-7b.json \
  --backbone-config default=configs/backbones/default.json

python scripts/run_queue.py --queue results/all-rqs/queue.json --gpus 0,1 --cpu-workers 0 --cpu-threads 8
python scripts/aggregate.py --queue results/all-rqs/queue.json --output results/all-rqs/summary.json
```

Each worker runs one configuration at a time. Start the queue inside the site's allocated GPU session; this runner does not request scheduler allocations. Multiple independent queues need distinct output directories and device allocations. A run can also be launched directly with `python -m rabr.run --config JOB.json --data DATA --domain electronic --output RUN --seed 42`.

Completed runs include label-free predictions, per-session evaluation, metrics, cost records, selection/training metadata, environment information, and hashed receipts. Seeds 42–46 are aggregated separately from sessions. A failed RABR certification produces `no_certified_pair`, without a replacement threshold or test score.

| Suite | Purpose |
|---|---|
| RQ1 | All 11 methods; two backbones where applicable; three domains; five seeds |
| RQ2 | RABR empirical risk across target risk levels |
| RQ3 | Score components and bundle construction ablations |
| RQ4 | Risk level, confidence level, F-beta weight, and sampling budget |
| Optional sensitivity | Separate calibration-size extension |

## Layout

```text
configs/        method, backbone, protocol, and RQ specifications
docs/           method scope, sources, and reproduction commands
requirements/  dependency groups
scripts/        data preparation, suite creation, queueing, aggregation
src/rabr/       methods, calibration, metrics, and execution
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source acknowledgements and applicable notices.
