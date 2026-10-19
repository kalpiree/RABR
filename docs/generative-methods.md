# Generative methods

All methods implement `fit(train, validation, catalog)` and `predict(context, k, seed)`. The common runner passes only the intent, strict-past history, session identifier, timestamp, and optional user identifier to prediction. Training targets are available during fitting; validation selects model checkpoints. RABR calibration sessions do not enter these methods' fitting procedures.

Generated product descriptions are grounded against the full static title catalog using the configured shared encoder. Exact catalog IDs are returned. Invalid JSON is recorded and produces an empty recommendation; no hidden retrieval fallback repairs malformed bundle outputs. Prediction metadata identifies the implementation variant and language-model calls.

| Method | Implementation | Source |
| --- | --- | --- |
| Zero-shot | One intent/history prompt, product-description grounding | RABR baseline protocol |
| LLM4B | Query cosine recall of 30 catalog items, then LLM selection using query and history | Text2Bundle §5.2 |
| Text2Bundle | Intent decomposition, LightGCN and FM pretraining, state/action Transformers, BPR pretraining, double DQN and EOS action | Text2Bundle §§3–4 |
| AICL | Nearest training demonstrations, mutual correction, labeled bundle feedback, intent feedback, rule summarization | SIGIR 2024 AICL |
| KD4BG | Teacher-derived category rules plus response-only LoRA student training | KD4BG 2025/2026 |
| KD4BG PatternICL | Training category patterns supplied at inference | Released KD4BG PatternICL configuration |
| DDBC | Residual item codes, masked denoising Transformer, constrained reverse diffusion | ICLR 2026 DDBC |

## LLM4B and Text2Bundle

The [Text2Bundle paper](https://arxiv.org/html/2310.18004) defines LLM4B in §5.2. Both names therefore legitimately reference the same paper. LLM4B is its retrieval-and-selection baseline. Text2Bundle is the learned reinforcement-learning model, implemented separately here.

Text2Bundle extracts four or five product interests with the language model and retains the original query. Train-only user–item edges pretrain LightGCN embeddings; a two-field factorization model ranks items retrieved by query/title cosine. Unknown users use their available history embeddings, or the mean training-user vector when history is empty. A six-type Transformer state represents query text, history text, selected-bundle text, user ID, history IDs and selected IDs. A second Transformer encodes candidate actions. A learned EOS action terminates generation.

Cosine BPR pretraining samples a training bundle prefix, an unselected positive and a negative outside that bundle. Double DQN then uses replay, a target network, epsilon-greedy exploration and Huber TD loss. The reward combines training-bundle precision, the pretrained personalization score, selected-item text similarity and entropy across inferred interests. Validation F1 selects the checkpoint; prediction stops at EOS or the requested size limit.

This is a paper-based reconstruction. The paper's external Amazon pretraining graph and RoBERTa features are replaced with training-only interactions and the shared configured encoder. Two-field FM on user/item IDs has the usual factor-dot-product form. Hyperparameters in `configs/methods/text2bundle.json` are explicit repository defaults; they are not a recovered author configuration. The accessible paper states that code and data will be released, but no official runtime was identified from its linked materials. The implementation retains the learned architecture rather than replacing Text2Bundle with a dialogue prompt.

## AICL

Sources: [paper](https://arxiv.org/html/2312.16262v2), [official source](https://github.com/BundleRec/bundle_generation/tree/604d9c6d774dd057bee1ece3b68e8c406ac97e9d), DOI [10.1145/3626772.3657808](https://doi.org/10.1145/3626772.3657808).

The source task groups products in an observed purchase session. This repository adapts that workflow to a query and strict-past history over the full catalog. Nearest demonstrations come only from training. Their labeled targets support automatic keep/remove/missing feedback; neither validation nor target-test labels are used for demonstrations. Intent ratings average repeated evaluations within each rater and request revision if any rater scores a reference aspect higher. The original heterogeneous ChatGPT/Claude evaluators are replaced with independent seeds of the configured backbone, recorded as `same_backbone_independent_seeds`. Demonstrations are cached in memory by training session and seed.

## KD4BG

Sources: [paper](https://arxiv.org/abs/2504.17220), [official source](https://github.com/KaiDF/KD4BG/tree/f9a713ee8ef8c435d78c65710ba794e9e0e3ad0a).

`configs/methods/kd4bg.json` enables `teacher_student`. The teacher extracts reusable rules from labeled training bundles. The student receives intent/history, mined category patterns and retrieved training rules, and learns the training bundle through response-only causal loss. Validation chooses the adapter after each epoch. LoRA targets attention and MLP projections with rank 16, alpha 16, learning rate 0.0002 and three epochs. Set `RABR_STUDENT_MODEL_PATH` to a local chat model; the teacher uses the common configured backbone. The `distillation` extra supplies PEFT.

The teacher cache is bound to input/config hashes and saved after each training example. Adapter, optimizer, RNG and selection state are saved after each epoch. A restart repeats at most the interrupted epoch. The frozen teacher rules can be regenerated from training data; no private teacher assets are required.

`configs/methods/kd4bg_pattern.json` explicitly selects `pattern_icl`. It mines category multisets of length 2–4 with training-session support at least two, retrieves patterns by semantic similarity, and makes one generation call. This is a partial configuration of KD4BG. It must be labeled **KD4BG PatternICL**, not the complete distilled system.

Both variants use the new query/history task and category miner. These are documented task adaptations, not reproductions of the authors' observed-session benchmark scores.

## DDBC

Sources: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/fec8e70600149b3c496e94d1f76ee0a7-Paper-Conference.pdf), [official source](https://github.com/LiAi16/DDBC/tree/e1f4afc59121662e626ffefde79316a5e0dde044).

Catalog text embeddings are whitened to 64 dimensions and quantized with three FAISS residual codebooks. A fourth occurrence digit resolves collisions in stable catalog order. The default codebook size is 128. Six 64-dimensional Transformer blocks use eight attention heads, AdaLN-zero, noninterleaved RoPE, 128-dimensional conditioning and the released default of disabled timestep conditioning. PyTorch SDPA replaces the upstream FlashAttention dependency.

Only item-code positions are corrupted. At time `t`, the masking probability is `0.999*t`; continuous SUBS loss weights masked-token cross entropy by `1/t`, normalized by code positions per example and then averaged across the batch. Same-length batches avoid padding and preserve all training bundle lengths. Length groups are sampled in proportion to their frequency. Adjacent swaps augment item order. AdamW, warmup and EMA follow the released settings; fixed validation corruption streams select the EMA checkpoint.

Inference retrieves one seed from query/history text without consulting the target. Reverse diffusion reveals masked codes over 25 steps, clamps structural tokens and seed codes, and constrains revealed assignments to existing, distinct catalog items. Matching feasibility prevents partial assignments that would make a distinct full bundle impossible. These full-catalog constraints are stricter than the released decoder. The paper's Appendix D.2 discusses seed-then-complete personalization; shared text embeddings replace its CLHE representation. The output-size policy is supplied by the common runner.

Checkpoints include optimizer, EMA, batch/corruption/global RNG state and input identity. PCA, FAISS index, catalog codes and selected-checkpoint metadata are written under the run's model directory. No imports refer to an older workspace or a cluster release.

## Runtime

The model modules import Torch, Transformers, FAISS and PEFT only when their corresponding learned methods are fitted. Base package import and prompt methods require NumPy only. See `THIRD_PARTY_NOTICES.md` for attribution and the inspected source-license status.
