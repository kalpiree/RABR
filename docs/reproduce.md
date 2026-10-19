# Reproducing the experiments

The repository provides RABR, the eight manuscript baselines, and two recent baselines: KD4BG and DDBC. The suite configurations cover RQ1–RQ4. They define a fresh reconstruction of the paper's protocol; the original partition IDs, complete prompt templates, and trained checkpoints are not available here.

## Installation

Use Python 3.11 on Linux for the complete method suite. Install the package and model dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[models,diffusion,distillation,quantization]'
```

Learned baselines use PyTorch; DDBC also uses FAISS. Configure local generation and sentence-encoder checkpoints in the backbone configuration files. Experiment configurations disable automatic downloads by default. KD4BG's teacher-student configuration also requires a local student checkpoint and its fine-tuning dependencies.

## Data

Clone the BundleRec repository into `data/raw/BundleRec`. Its CSV files are under `dataset/<domain>/`, where the domains are `electronic`, `clothing`, and `food`.

```bash
git clone https://github.com/BundleRec/bundle_recommendation.git data/raw/BundleRec
git -C data/raw/BundleRec checkout 087da2e2b4d8738ef23f3f6f2005505e8685edde
```

Each domain needs:

```text
item_titles.csv
item_categories.csv
user_session.csv
session_bundle.csv
bundle_intent.csv
bundle_item.csv
user_item.csv
```

Prepare the data with the declared selection seed:

```bash
python scripts/prepare_data.py \
  --source data/raw/BundleRec/dataset --output data/processed --mode raw \
  --seed 42 --validation-fraction 0.1 --history-limit 20 \
  --include-pretrain-history --require-history
```

An existing prepared dataset can instead supply `catalog.jsonl`, `train.jsonl`, `calibration.jsonl`, and `test.jsonl` per domain. Session records must contain `session_id`, `domain`, a numeric `timestamp`, `input.intent`, `input.history`, and `target.gold_items` with string catalog IDs. Optional `user_id` values are preserved. Importing preserves calibration and test target membership and takes validation from the supplied training split:

```bash
python scripts/prepare_data.py \
  --source data/existing --output data/processed --mode import \
  --validation-fraction 0.1 --history-limit 20
```

The raw conversion chooses one annotated bundle per session with a deterministic hash. It places whole timestamp groups near the cumulative 60% and 90% boundaries, then reserves the latest 10% of the original training partition for validation. Ties never cross partitions. The resulting proportions are approximate. Validation selects learned-model checkpoints; calibration labels are used only by RABR. Pattern mining and fixed-K selection use the final training partition.

Histories contain at most 20 distinct items strictly earlier than the session. `--include-pretrain-history` adds same-user, catalog-known events from `user_item_pretrain.csv`, with source hashes and strict timestamp filtering. `--require-history` selects nonempty-history sessions before splitting and records the selected cohort; this optional filter supports Equation 8 without inventing an empty user profile. Missing categories remain null. Bundle-intent annotations serve as assumed user queries; this does not establish that the annotations were available before purchase. The catalog is a static set of item metadata. Without `--require-history`, empty-history sessions are retained. RABR then requires either a documented neutral cold-start policy or a history-eligible imported cohort. POG and other history-dependent models retain their native prerequisites.

Each output domain contains a manifest, catalog, training patterns, and separate input and target JSONL files for all four splits. Source and processed files are hashed. The loader checks split disjointness, time boundaries, training-derived patterns, and the training-derived median output size. An existing output directory is never silently replaced.

## Experiment suites

Configure `configs/backbones/qwen25-7b.json`, `llama31-8b.json`, and `default.json` with the required local checkpoints. The default backend supplies the sentence encoder for methods without a generative backbone. Build all research questions together, or pass a subset of the suite files:

```bash
python scripts/build_suite.py \
  --suite configs/suites/rq1.json configs/suites/rq2.json \
          configs/suites/rq3.json configs/suites/rq4.json \
  --data data/processed --output runs/experiments \
  --backbone-config qwen25-7b=configs/backbones/qwen25-7b.json \
  --backbone-config llama31-8b=configs/backbones/llama31-8b.json \
  --backbone-config default=configs/backbones/default.json
python scripts/run_queue.py \
  --queue runs/experiments/queue.json --gpus 0,1 --cpu-threads 4
```

Each worker runs one job at a time. GPU IDs are local device indices. The worker masks its selected GPU, so experiment devices use `cuda:0`. There are no cluster credentials, remote-login commands, or host-specific paths in the queue.

| Suite | Methods and parameters |
| --- | --- |
| RQ1 | POG, BUNT, BGN, BYOB, Zero-Shot, LLM4B, Text2Bundle, AICL, KD4BG, DDBC, RABR; two language backbones where applicable |
| RQ2 | RABR risk targets 0.1, 0.2, 0.3, 0.4, 0.5 |
| RQ3 | Equation (8), without-grounding, without-relevance, zero-shot, independent fixed-K sampling, and RABR fixed-K |
| RQ4 | Risk target, confidence, F-beta weight, and sampling budget |

Experiments use seeds 42–46 and all three domains. RQ3 uses Llama3.1-8B, as in the manuscript's ablation setup. RQ4 uses budgets 1–10; each budget experiment has the single candidate prefix `[s]`. The calibration fractions 0.25, 0.5, 0.75, and 1.0 are an additional declared grid: the manuscript raises calibration size in the research question but does not provide its exact grid.

The main score and component ablations follow Equation (8); frequency interpretations are removed. RQ3 fixed K uses the uncapped rounded final-training mean. RQ1 POG/BGN/BYOB explicitly use `paper_rabr_mean`, which creates dependencies on the matching domain, seed and reference backbone's calibrated mean output size. Two reference-backbone conditions keep this choice explicit. BUNT's native multiround feedback protocol has no artificial size dependency and is reported separately. Test predictions never determine K.

LLM4B, AICL and KD4BG match final recommendation sampling to the selected RABR budget under `paper_rabr_selected`; training feedback/teacher calls are separate costs. Zero-shot keeps one completion. Diffusion steps and DQN actions are not LLM samples. RQ1 explicitly selects DDBC's CLHE-based query-seed adapter for full-bundle evaluation. Its supplied-seed completion configuration requires a separately prepared input/target protocol. The full-catalog sample aggregation convention is documented in [the protocol](rabr-protocol.md).

Set the model and native-asset environment variables in [traditional methods](traditional-methods.md), [Text2Bundle](text2bundle.md), [generative methods](generative-methods.md), and [DDBC](ddbc.md). Missing assets produce explicit deterministic prerequisite failures. The original proprietary POG collaborative features are not distributed here. Generic feature replacements cannot satisfy that native profile.

Method defaults are loaded from `configs/methods/<method>.json`; suite defaults and individual experiment settings override them. Explicit backbone files override backend settings. Every expanded configuration is saved separately and hashed. Change the output directory whenever data, source code, or configuration changes.

## Resume and results

Rerun the same queue command to resume. The coordinator uses a local lock, preserves per-job state, refuses to duplicate a live child, validates completed artifacts, and retries transient failures at most three times. Deterministic errors stop immediately. Timeouts terminate the worker process group. Logs are stored separately from hashed run artifacts. Stop the coordinator normally so its children are cleaned up; after an abrupt coordinator crash, resolve any still-live child before resuming.

Every terminal run writes `result.json` with configuration, dataset, source, and artifact hashes. `no_certified_pair` is a terminal calibration outcome. It is not an empty recommendation with a quality score. Resumed prediction records are reused only under the same run identity.

Aggregate only after all selected jobs have terminal result receipts:

```bash
python scripts/aggregate.py \
  --queue runs/experiments/queue.json --output runs/experiments/summary.json
```

Aggregation verifies prediction coverage and recomputes metrics from the committed predictions and test targets. It reports per-configuration seed means and sample standard deviations, preserving no-certificate outcomes separately. RQ1 paired two-sided tests require matched seeds, complete results and compatible evaluation protocols; nominal and Holm-adjusted p-values are reported. The chosen paired t-test family is declared because the manuscript does not name one. Five seeds reuse the same test sessions and are not counted as five independent test datasets. Recorded costs are retained per run; interrupted work not committed to a checkpoint is not assumed to be measured.

The calibration calculation implements its stated test algebra. Chronological splits and repeated users do not establish exchangeability or a verified distributional guarantee. Fresh-protocol outputs should be identified by their manifests and settings rather than presented as exact reproductions of the manuscript's tables.
