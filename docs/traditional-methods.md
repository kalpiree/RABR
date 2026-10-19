# Traditional bundle-generation methods

The `pog`, `bgn`, `bunt`, and `byob` entries are independent PyTorch reconstructions for the repository's intent/history-to-catalog task. Their architecture, objective, and decoder are separate implementations. The profiles use the available catalog and split schema; they are not packaged reproductions of the original datasets or published scores. Source URLs, audited upstream commits, and implementation scope are recorded in [traditional-upstream.json](traditional-upstream.json).

## Interface and data

```python
from rabr.baselines.traditional import create_traditional

predictor = create_traditional("bgn", algorithm_config)
predictor.fit(train_rows, validation_rows, catalog_rows)
result = predictor.predict({"input": {"intent": "camera accessories", "history": []}}, k=3, seed=42)
```

Training rows use `target.gold_items` (or `target_items`) and `input.intent`/`input.history`. Catalog rows require unique string `item_id` values and may include `title` and `category`. Prediction receives no target field; passing one raises an error. Only training data supplies fitting examples and simulated rewards. Validation selects checkpoints and early stopping; calibration and test targets are never passed into these classes.

Text uses deterministic signed hashing of title/category words, projected jointly with trainable item and category embeddings. Intent supplies a projected context token. Known history items supply additional context; absent history remains valid. This feature profile replaces unavailable image, collaborative-filtering, and persistent-user assets. `item_features_path` can supply a JSON mapping from every catalog ID to a finite `feature_dim` vector. Precomputing those vectors is the caller's responsibility; the file hash is included in the fit identity.

Install PyTorch to create a traditional predictor. Importing the factory alone does not import PyTorch. All methods support CPU and CUDA through `algorithm.device`. Configuration files default to CPU; use `"device": "cuda"` for GPU jobs. No pretrained weights or network access are required by these implementations.

## POG

[POG](https://arxiv.org/abs/1905.01866) separates an item-context compatibility model from personalized autoregressive generation. This implementation pretrains a bidirectional masked-item Transformer (FOM), transfers its self-attention/feed-forward weights into the generation decoder, and trains a Transformer history encoder with a causal cross-attention decoder. The default objective uses the target plus three sampled negatives. Output scores use transition embeddings, and no positional embeddings are added. Training includes an end token.

The default six layers and width 512 follow the supplied RABR appendix; the original POG paper reports different hidden-width settings. Category-triplet multimodal pretraining, image/CF feature extractors, and fashion-specific category rules are not implemented. The default text/category feature profile and intent token are adaptations. Fixed-K inference suppresses the learned end token and masks already selected catalog IDs. `fom_epochs` controls compatibility pretraining; `epochs` controls generation training. Both stages select their lowest validation objective.

## BGN

[BGN](https://arxiv.org/abs/1904.01933) models bundle quality with a text-CNN context encoder, attention LSTM decoder, and feature-aware softmax. Here the softmax weight vectors are the current learned item/category/text representations. Training uses full-catalog sequence cross-entropy, including an end token, rather than sampled softmax.

At each decoding expansion, repeated items are masked; the highest-quality candidate prefixes are retained; greedy DPP selection maximizes log sequence quality plus `diversity × logdet(Jaccard kernel)`. The default diversity coefficient is 0.1. The DPP operates **between bundles**, as in the original task. It does not impose a within-bundle item diversity objective.

`predict_list` exposes the diversified bundle list and sequence log probabilities. `predict` returns its first bundle and includes the entire list in metadata. A one-bundle evaluation cannot measure list diversity; the candidate list still affects prefix survival during decoding. The fixed-K profile suppresses the end token instead of using the original variable-length end-token penalty. Validation selects minimum sequence cross-entropy.

## BUNT

[BUNT / Bundle MCR](https://arxiv.org/abs/2207.12628) uses hierarchical history Transformers, cross-attentive bundle slots, item/category/attribute heads, and a gated conversation manager. Offline cloze training masks bundle items and their category/title-token attributes. Its losses train item reconstruction, category/attribute prediction, and successful ask-versus-recommend decisions. Online fitting freezes the representation backbone and updates the decision heads with clipped PPO and value losses using feedback simulated **only from training targets**. The manager receives terminal bundle-F1 returns; item/category policies receive immediate acceptance rewards.

The reconstruction uses catalog categories and a bounded title-token attribute vocabulary. Historical bundle IDs are used when available; otherwise each known historical item forms a singleton history bundle. Category consultation is the online question policy; the attribute head is trained in the cloze stage. Fixed output slots persist after acceptance rather than creating new slots indefinitely. These choices differ from the original interaction environment and its richer metadata. Validation uses a fixed-seed simulated conversation reward, without updates.

A genuine conversation needs a respondent. The Python API accepts `context["feedback_provider"]`, a callback receiving either an `ask` request with a category or a `recommend` request with proposed item IDs. It returns `{"accepted": true}` / `{"rejected": true}` for a category, or `{"accepted_items": [...], "rejected_items": [...]}` for proposed items. Returning `None` ends feedback collection. A finite `context["feedback"]` list can provide recorded replies. At most 20 turns are supported; the actual number and transcript are returned in metadata. Responses must not be fabricated from evaluation targets.

Without a respondent, prediction uses the trained item policy once and records `feedback_mode=one_shot_no_respondent` and zero conversation turns. The standard fresh-data RQ1 profile therefore evaluates a BUNT one-shot adaptation, not a 20-turn target-oracle simulation. The runner's input-only record does not add a respondent automatically.

## BYOB

[BYOB](https://github.com/fuxiAIlab/BYOB) builds a bundle sequentially with an item-pointer policy. This implementation first fits item-preference and pair-compatibility scorers using positive training interactions/co-membership and sampled negatives. It then freezes these reward models and learns the composition policy with clipped PPO and a value baseline.

The three reward components implement the **B-M branch**: item preference, summed compatibility with previously chosen items, and terminal bundle F1. Only the training simulator accesses a target bundle. The policy state concatenates the intent, mean known-history representation, and mean partial-bundle representation. It replaces the upstream persistent user embedding and uses the full catalog rather than a target-injected candidate pool. A 0.2 uniform exploration mixture is used during training; its actual mixture probability is used in the PPO ratio. Prediction is deterministic greedy pointer decoding with selected-item masking and does not call a reward model or access targets.

The original code supports multiple reward/model variants and Ray/RLlib optimizers. This implementation selects a trainable B-M profile with compact PyTorch reward scorers; it does not reproduce the original NCF/item2vec checkpoints, optional candidate-pool Transformers, or distributed optimizer. Validation selects maximum greedy bundle F1. `reward_epochs` and `epochs` control reward-model fitting and policy learning respectively.

## Training persistence

When `work_dir` is supplied, a completed fit writes `traditional-model.pt` and `traditional-fit.json`. The receipt hashes the weights, implementation, configuration, catalog, training rows, validation rows, and any external feature file. An identical retry restores the completed fit; changed inputs or corrupted weights fail validation. Incomplete fits restart training: optimizer/rollout state is not saved inside an unfinished epoch. The experiment runner separately checkpoints completed predictions.

Model-selection randomness is reset for validation. Seeds control initialization, training order, masking, sampled negatives, and policy rollouts. Fixed-model inference uses greedy decoding, so its `seed` argument is accepted for the shared interface but does not add sampling.
