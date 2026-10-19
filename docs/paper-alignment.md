# Alignment with the manuscript

This repository implements the method formulation and explicitly separates evaluation adaptations from original method components. Original manuscript split IDs, complete prompts, author checkpoints and several native feature assets are unavailable. Running the code does not recover those artifacts or establish numerical reproduction.

## RABR and the four research questions

RABR now supplies accumulated grounded candidates to every subsequent generation, derives relevance only from prior interactions, and implements Equation 8 without an invented frequency term. The grounding averages, bounded F-beta loss, KL test, Bonferroni correction, threshold inclusion and minimum-budget/maximum-threshold choice follow the numbered formulation. Empty-history handling is explicit; the default requires genuine history.

| Workflow | Implemented rule |
|---|---|
| RQ1 | RABR calibration dependencies for fixed-size conventional adapters and stochastic LLM recommendation samples; matched-seed paired two-sided tests and explicit reference-backbone conditions |
| RQ2 | Risk-target sweep using the same corrected generation and calibration paths; uncertified outcomes remain terminal |
| RQ3 | Equation 8 reference, grounding removal, relevance removal, independent fixed-K sampling, RABR fixed-K, and zero-shot; fixed K is uncapped rounded training mean |
| RQ4 | Alpha, delta, beta and generation-budget sweeps; calibration-size extension lives separately in `sensitivity.json` |

The manuscript's frequency discussion is inconsistent with its equation. The implementation follows the method formulation, so the component names and resulting ablation columns should be grounding and relevance. No numerical frequency-ablation claim is made.

## Baseline components and protocol boundaries

| Method | Corrected implementation | Remaining input or evaluation boundary |
|---|---|---|
| POG | Native-dimensional image/TextCNN/CF inputs, category-triplet fusion, separate Per/FOM/Gen computation and masked training | Original Behemoth CF features are proprietary; the native profile requires them. RABR fixed-K decoding is explicitly distinct from native EOS decoding. |
| BGN | ID/category feature-aware CNN/LSTM, sampled output loss, native EOS penalty and full bundle-list DPP | Fixed-K RABR evaluation is an explicit decoder adaptation; the original bundle-list task differs from one-bundle recommendation. |
| BUNT | Hierarchical cloze pretraining, separate item/category/attribute policies, manager PPO and genuine repeated feedback with slot replenishment | Requires real tags and historical bundle groups. Native feedback-assisted results are reported separately from query-only methods. |
| BYOB | Pinned original pointer/NCF/SkipGram definitions, compatibility threshold, item-to-bundle reward curriculum, undiscounted PPO | Requires persistent users, training interactions/click sequences and target-independent candidate pools. Fixed-K RABR training is a declared evaluation adaptation. |
| Zero-shot | One completion, strict parsing and catalog grounding | Exact manuscript prompt and encoder snapshot are not available. |
| LLM4B | Retrieval followed by constrained LLM selection | Query/history full-catalog evaluation and repeated-sample aggregation are declared. |
| Text2Bundle | External RoBERTa/LightGCN features, full sparse FM interactions, state/action Transformers, BPR and double DQN | Requires original-style assets. The source paper has unresolved dimensional descriptions; see its method document. |
| AICL | Training-only correction/feedback, two genuinely distinct evaluator models with the source rating scales | Local model identities differ from the released proprietary evaluators. Query/history recommendation remains an adaptation of observed-session grouping. |
| KD4BG | Apriori category sets, teacher prediction/error review/rules/thoughts, response-only LoRA with batch/accumulation/warmup/linear schedule | Fresh teacher corpus, Transformers/PEFT runtime, semantic knowledge retrieval and query/history task differ from the released observed-session experiment. |
| DDBC | Verified CLHE features, RVQ, level-wise legal-token diffusion and explicit supplied seeds | Full-catalog retrieval avoids target-built candidate pools. Variable-length training batches and the optional query-seed adapter are declared separately. |

See [traditional methods](traditional-methods.md), [generative methods](generative-methods.md), [Text2Bundle](text2bundle.md) and [DDBC](ddbc.md) for asset formats and pinned primary sources. Asset requirements are operational prerequisites; synthetic arrays used in small verification checks are not research features.

## Data and claims

The raw protocol selects one annotated bundle per session, splits whole timestamp groups near 60/30/10 and reserves the final part of training for validation. Annotated intents are treated as assumed queries. Optional pretraining histories use only genuine earlier same-user events. Optional history eligibility is applied before splitting and recorded in the manifest. These choices define a fresh benchmark, not the missing original partitions.

Tests and small integration runs check equations, gradient flow, response-only labels, interaction boundaries, source/asset identity, calibration dependencies and checkpoint recovery. They do not establish convergence, original checkpoint equivalence or theorem sampling assumptions. Earlier experiment results have different source/protocol identities and cannot be relabeled as outputs of this corrected implementation.
