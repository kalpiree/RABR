# Alignment with the manuscript

This version provides executable reconstructions and experiment workflows. It is not a verified reproduction of every original method or of the manuscript's four experimental sections. Some differences are task adaptations; others are missing implementation steps. They should not be treated interchangeably.

## RABR

The calibration module implements nearest-title grounding, raw-cosine acceptance, within-generation averaging, matching-generation averaging, Equation 8 weighted scoring, inclusive threshold selection, set F-beta loss, one-sided KL p-values, Bonferroni correction, and minimum-budget / maximum-threshold selection. A failed certificate remains `no_certified_pair`.

Two implementation gaps remain in the generation and relevance path:

- Section 3.1 requires each generation prompt to include accumulated candidate titles and request additional, nonrepeated items. The current runner reuses the same prompt across rounds. It accumulates grounded scores but does not supply that pool to subsequent prompts.
- Equation 8 derives the user-profile embedding from interaction history. The current encoder input combines intent and history. A faithful history-only profile, including an explicit empty-history policy, has not been implemented.

Gamma, grounding threshold, sentence encoder, description limit, threshold grid, and exact prompts are declared reconstruction choices where the manuscript does not provide a complete executable specification.

Equation 8 does not include generation frequency, while RQ3 and Appendix B.3 describe a frequency component. The separate `frequency_similarity` scorer is a chosen interpretation, not an author-specified formula. This ambiguity must be resolved before claiming an exact RQ3 reproduction.

## Baselines

| Method | Difference in this repository |
|---|---|
| POG | Reconstructed Transformer/FOM training. Original image/CF features and multimodal category-triplet pretraining are absent; title/category hashing and an intent token replace those inputs. Fixed-K inference suppresses EOS. |
| BGN | Reconstructed text-CNN/LSTM/feature-softmax and bundle-level DPP. Full-catalog cross-entropy and fixed-K decoding replace the original sampled-softmax/variable-length setup. |
| BUNT | Learned cloze and policy components are present, but the standard experiment profile has no respondent and uses one-shot prediction. It does not reproduce the original 20-turn interaction experiment. |
| BYOB | Reconstructed B-M reward/pointer/PPO pipeline. Original NCF/item2vec assets, persistent-user representation, optional candidate encoders, and distributed optimizer are not reproduced. |
| Zero-shot | Direct prompting with shared JSON parsing and full-catalog grounding. Exact manuscript prompts are unavailable. |
| LLM4B | Retains retrieval followed by LLM selection, using the shared query/history protocol and sentence encoder. |
| Text2Bundle | Retains learned graph/FM/Transformer/BPR/DQN components. Training-only interactions and the shared encoder replace the external Amazon pretraining graph and RoBERTa features. Hyperparameters are repository defaults. |
| AICL | Original observed-session grouping is adapted to query/history recommendation. Different evaluator models are replaced by independent seeds of one backbone. |
| KD4BG | Teacher-rule extraction and LoRA student fitting are reconstructed for the shared task with a new category miner. PatternICL is a separately labeled partial configuration, not another baseline or the complete distilled method. |
| DDBC | Retains RVQ item coding and masked diffusion. Shared text features replace CLHE; inference uses a query-derived seed and stricter full-catalog distinct-item constraints. SDPA replaces FlashAttention. |

The detailed method documents include primary sources, checkpoint choices, objectives, and omitted original assets.

## Research questions

| Question | Implemented workflow | Remaining alignment issues |
|---|---|---|
| RQ1 | All eleven method entries, two backbones where applicable, three domains, five seeds, shared metrics | Adapted methods and fresh splits; training-median K instead of the manuscript's RABR-average rule by default; no shared RABR-selected sampling-budget dependency; no paired two-sided significance test |
| RQ2 | Calibration and empirical-risk sweep over alpha | Inherits the RABR generation/profile gaps and reconstructed data protocol; an empirical curve does not establish the theorem's sampling assumptions |
| RQ3 | Score-component and bundle-construction ablations | Frequency formula is unresolved; fixed-K uses capped training median whereas the manuscript's construction comparison specifies training mean; inherits the generation gap |
| RQ4 | Alpha, delta, beta, and sampling-budget sweeps; an additional calibration-size sweep | Inherits RABR/protocol gaps; calibration-size grid is an explicit extension rather than a recovered manuscript setting |

The raw-data protocol chooses one bundle per session, splits whole timestamp groups near 60/30/10, reserves a final portion of training for validation, uses strictly earlier histories, and treats annotated intents as assumed queries. Original split IDs are unavailable. These decisions change the experiment and must be retained in any result description.

The optional `paper_rabr_mean` policy derives K from a completed matching RABR run's calibration outputs. It is safer than choosing K from test predictions, but is not enabled by the default suites. Aggregation currently reports seed means and standard deviations; significance testing remains separate work.
