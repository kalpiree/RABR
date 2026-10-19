# Generative methods

All methods implement `fit(train, validation, catalog)` and `predict(context, k, seed)`. Calibration targets do not enter their fitting. The query/history adapter and the original observed-session or bundle-completion tasks are different evaluation conditions; results must identify which one was used.

## Zero-shot and LLM4B

Zero-shot makes one recommendation completion. LLM4B retrieves 30 candidates by query/title cosine, then selects exact IDs using query and history, following the baseline in [Text2Bundle §5.2](https://arxiv.org/html/2310.18004). The full-catalog encoder and exact prompt serialization are declared benchmark choices. Invalid JSON is empty; an invalid exact-ID selection invalidates its sample. Configured repeated final completions have explicit cost and aggregation receipts.

## Text2Bundle

The learned Text2Bundle path uses RoBERTa text representations, supplied Amazon-domain LightGCN features, a sparse factorization machine, state/action Transformers, BPR pretraining and double DQN. The old shared-encoder and train-only graph replacements are removed. See [Text2Bundle inputs and implementation](text2bundle.md) for asset manifests, native constants, the EOS policy and ambiguities in the source paper.

## AICL

Sources: [paper](https://arxiv.org/html/2312.16262v2), [official code](https://github.com/BundleRec/bundle_generation/tree/604d9c6d774dd057bee1ece3b68e8c406ac97e9d).

Training neighbors supply demonstrations. Generation and inferred intent are mutually corrected; training labels supply bundle feedback. Two distinct evaluator models rate intent naturalness and coverage on 1–3 scales and motivation on a 1–2 scale. Each evaluator repeats its rating three times. An aspect is revised if either evaluator's average prefers the reference intent. Invalid ratings fail explicitly. Model paths are expanded and resolved before enforcing evaluator distinctness; separate seeds of one checkpoint do not satisfy this requirement.

Set `RABR_AICL_RATER1_PATH` and `RABR_AICL_RATER2_PATH` to two local chat snapshots. Their identities and call costs are retained. Choosing local evaluator models replaces the released ChatGPT/Claude model identities, not the two-evaluator procedure. Exact historical proprietary endpoints cannot be reproduced from local weights.

The RABR adapter recommends from query/history over the full catalog. The original AICL task groups already observed products into multiple bundles. That task change remains explicit; no held-out bundle labels enter demonstration construction, retrieval or prompt feedback.

## KD4BG

Sources: [paper](https://arxiv.org/abs/2504.17220), [official code](https://github.com/KaiDF/KD4BG/tree/f9a713ee8ef8c435d78c65710ba794e9e0e3ad0a).

The default teacher/student path mines training-only category sets with Apriori support pruning. It generates an initial training prediction, compares it with training labels, reviews its errors, extracts reusable rules, and generates deeper explanatory knowledge. The teacher cache records each stage and is bound to input/config hashes. Teacher knowledge is retrieved only from training examples.

Set `RABR_KD_TEACHER_PATH` and `RABR_STUDENT_MODEL_PATH` explicitly. Student training uses response-only causal loss, LoRA rank/alpha16 on attention and MLP projections, batch2 with four-step accumulation, five warmup updates, linear decay, learning rate0.0002, weight decay0.01 and three epochs. The released-style profile selects the final epoch. Validation losses are diagnostic. Model, optimizer, scheduler and RNG state are checkpointed each epoch. A 4-bit CUDA profile uses PEFT/bitsandbytes; CPU verification explicitly disables quantization.

The original release uses Unsloth and proprietary teacher outputs for observed-session bundle detection. This implementation uses Transformers/PEFT and freshly produced teacher knowledge for the RABR query/history task. Semantic retrieval of knowledge in that adapter is not the release's category-similar observed-session lookup. These task, teacher and runtime differences remain declared; this is not a claim to recover the authors' SFT corpus or exact model scores.

The optional `kd4bg_pattern.json` is a partial PatternICL configuration of the same method, not a second baseline. It is excluded from the main comparison suite.

## DDBC

Sources: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/fec8e70600149b3c496e94d1f76ee0a7-Paper-Conference.pdf), [official code](https://github.com/LiAi16/DDBC/tree/e1f4afc59121662e626ffefde79316a5e0dde044).

The native feature path requires verified CLHE features and explicitly supplied seed items. It uses residual quantization, masked diffusion and level-wise illegal-token filtering. Full-catalog retrieval does not construct candidate sets from test targets. The separate query-seed adapter is a distinct evaluation protocol for the same method. See [DDBC](ddbc.md) for feature provenance, decoder, training and evaluation details.

## Dependencies and scope

Torch, Transformers, FAISS and PEFT load only for the methods requiring them. The `quantization` extra supplies Linux bitsandbytes for 4-bit student training. No model weights or proprietary feature files are bundled. Source acknowledgements are in `THIRD_PARTY_NOTICES.md`.
