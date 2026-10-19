# RABR and evaluation protocol

## Generation and Equation 8

Each round receives the intent, strictly earlier interactions, and accumulated grounded candidate titles. It requests additional items without repeating that pool. Checkpoint recovery rebuilds the pool in the same order of rounds and verifies the input, history-profile and prompt hashes. Independent multi-sample generation deliberately omits this feedback.

A response contains an `items` list of descriptions. Invalid JSON contributes an empty generation. At most `v_max` descriptions are assigned to their nearest catalog title if raw cosine is at least `tau`. Accepted similarities become `(1+cosine)/2`. Multiple descriptions for an item are averaged within a round, then across the rounds containing that item. Missing appearances do not insert zeros into that mean.

Equation 8 is authoritative:

`score(i) = gamma * mean_grounding(i) + (1-gamma) * history_relevance(i)`

The user profile contains catalog titles and categories from prior interactions only. Intent is available to generation but is excluded from this profile. Relevance is rescaled cosine similarity to the catalog title. `without_grounding` sets gamma to zero; `without_relevance` sets it to one. There is no frequency term or frequency ablation.

The manuscript does not define a profile for empty histories. The default `empty_history_policy=require` rejects such inputs before generation. Explicit `neutral` assigns relevance 0.5 and is a cold-start adaptation, not a recovered manuscript formula. Data preparation can include genuine earlier pretraining interactions; it never fills histories with target items. Declare any filtering or cold-start policy before calibration and retain the changed dataset identity.

## Calibration and construction ablations

Budgets and thresholds are finite grids fixed before calibration. Every calibration session receives the largest budget; smaller budgets reuse prefixes. For every pair, set loss `1-F_beta` is converted to a one-sided Hoeffding/KL p-value and tested at delta divided by the full grid size. Selection takes the smallest certified budget, then its largest certified threshold. Threshold membership is inclusive.

No passing pair produces `no_certified_pair`, with no test predictions or substitute threshold. Certification remains conditional on the theorem's sampling and fixed-grid assumptions; a chronological split does not establish them.

The primary defaults are alpha=delta=0.05, beta=1, gamma=0.5, budgets 1–10, thresholds 0–1 in increments of 0.05, tau=0, and `v_max=8`. Encoder, exact prompt text, gamma, tau and finite grid resolution are declared choices where the manuscript leaves implementation details open.

RQ3 fixed-K constructions use the uncapped half-up rounded mean of final-training bundle sizes. Their test generation budget comes from the matching adaptive Equation 8 run's calibration selection. `fixed_k` retains candidate feedback and Equation 8 scoring. `multi_sample_fixed_k` uses independent prompts and matching-sample mean grounding similarity. These top-K constructions are marked `uncertified_ablation`; their aggregation rule is explicit because the manuscript does not fully specify it.

RQ1 POG/BGN/BYOB use the rounded RABR calibration mean output size under an explicit `rabr_fixed_k` evaluation adaptation. Separate reference-backbone conditions avoid choosing or pooling the two RABR backbones implicitly. BUNT retains its native interaction protocol and has no artificial RABR size dependency. A missing certificate blocks dependent comparisons.

Stochastic LLM4B/AICL/KD4BG inference can use `sampling_budget_policy=paper_rabr_selected`. Each sample is a final recommendation completion; AICL's training-demonstration feedback and KD4BG's teacher training are accounted for separately. Description samples use matching-sample mean grounding; exact-ID samples use selection frequency. Zero-shot has exactly one completion. DQN transitions and diffusion denoising steps are not counted as LLM samples.

## Data and evaluation boundaries

Training labels support model fitting, pattern mining, teacher knowledge, and training reward simulators. Validation selects checkpoints when specified by the method; KD4BG's released training profile instead selects the final epoch and records validation diagnostically. RABR calibration labels select only its budget and threshold.

The common prediction input is an allowlist of identity, timestamp, optional user ID, intent, history, and explicitly supplied seed items. Test targets are excluded. Query-only and supplied-seed predictions are committed before scoring.

BUNT is a separate feedback-assisted protocol: a held-out simulator can respond only to proposed items or tag questions. It does not expose the target bundle to the policy. Evaluation includes rejected proposals in the recommendation denominator, rather than scoring accepted items alone. Its results must not be pooled with query-only significance comparisons.

Macro Precision, Recall, HitRatio, F-beta, `1-F_beta`, and bundle size retain every test session. Parse failures are empty predictions. Unknown exact-ID selections invalidate their entire sample. Empty predictions have zero Precision/Recall/F-beta and risk one.

Paired tests operate on matched seed-level scores, not on sessions multiplied by seeds. The primary analysis uses a two-sided paired t-test and Holm adjustment across planned baseline comparisons, with optional exact sign-flip results. The manuscript does not identify its paired-test family; this choice is declared. Incomplete seeds, failed certificates and incompatible evaluation protocols do not enter a valid paired comparison.

## Persistence

Per-run locks prevent concurrent writers. Append-only checkpoints bind records to data, config, source, seed and model inventory. Completed receipts hash predictions, selections and fitted models. Changes require a new output directory. External manifests and input assets are verified; local model inventories record file names, sizes and modification times, not weight-level cryptographic identity.

Cost records distinguish the current attempt, durable generation/prediction costs, and auxiliary evaluator/teacher calls. Lost uncommitted work is not reconstructed. The local queue uses bounded retries and one worker per requested GPU; scheduler allocation and cross-host access remain deployment responsibilities.
