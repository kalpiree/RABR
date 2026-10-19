# Traditional bundle-generation methods

The implementations use method-specific inputs and objectives. Missing modalities, user identities, histories, or candidate pools raise errors. There is no hash-text feature fallback, invented user embedding, or automatic replacement of missing interactive feedback.

```python
from rabr.baselines.traditional import create_traditional

predictor = create_traditional("bgn", algorithm_config)
predictor.fit(train_rows, validation_rows, catalog_rows)
result = predictor.predict(input_only_context, k=3, seed=42)
```

Rows use `target.gold_items`, `input.history`, and a stable `session_id`; BYOB additionally requires `user_id`. Timestamped histories are sorted chronologically before keeping the most recent events; histories without timestamps retain their declared order. Prediction rejects target-bearing contexts. Calibration/test targets are never used to fit these models. Method defaults retain native stopping and input requirements. The RABR suite selects explicit evaluation adapters where its protocol differs.

Install the `models` optional dependencies for PyTorch. Importing the factory alone does not import PyTorch. `algorithm.device` controls CPU/CUDA placement. BYOB also needs the external pinned source checkout described below. See [traditional-upstream.json](traditional-upstream.json) for source scope.

## POG

[POG §§4.1–4.3 and 5.1–5.2](https://arxiv.org/html/1905.01866) specifies multimodal item embedding, masked FOM pretraining, and personalized generation. The implementation now requires image, category-trained text, and collaborative vectors; it trains category triplets, then every-position FOM masking, then causal generation. Per and FOM/Gen have separate transition layers, and FOM transition/attention weights initialize Gen. Sampled prediction uses three negatives outside the outfit. Defaults are 128-dimensional multimodal embeddings, six layers, width 64 and eight heads. No intent token is inserted into historical clicks.

The RABR appendix profile overrides both multimodal embedding width and Transformer width to 512, with six layers and eight heads; the method defaults retain the original paper widths. Native decoding honors END. `decoding="fixed_k"` explicitly implements RABR's fixed-size evaluation by suppressing END; that change is recorded in prediction metadata. `maximum_bundle_size` is only a safety cap for native decoding. Dida's production category rules and original online experiment are outside this implementation.

The [official POG release](https://github.com/wenyuer/POG) contains data, not the model checkpoints. Original Behemoth collaborative features remain an external prerequisite. Training the three implemented stages does not reconstruct that proprietary platform or certify original-paper scores.

Set `POG_ASSETS_PATH` to a manifest with this structure:

```json
{
  "schema_version": 1,
  "item_ids": ["catalog-id-1", "catalog-id-2"],
  "image": {"path": "image.npy", "sha256": "<sha256>", "extractor": "InceptionResNetV2", "provenance": "<extractor checkpoint and image source>"},
  "text": {"path": "text.npy", "sha256": "<sha256>", "extractor": "TextCNN-category", "training_scope": "training_only", "provenance": "<category-training data and checkpoint>"},
  "cf": {"path": "cf.npy", "sha256": "<sha256>", "extractor": "Behemoth", "training_scope": "training_only", "provenance": "<click-training data and checkpoint>"}
}
```

The arrays must have shapes `[catalog_size,1536]`, `[catalog_size,300]`, and `[catalog_size,160]`; all catalog IDs must appear exactly once. Paths are relative to the manifest. Files are checked against SHA256; provenance and training-scope declarations are retained, but declarations alone do not independently prove the origin of a feature file. Features must be computed for the intended catalog using permitted training data. Category labels must support same-category positives and different-category negatives. The default requires at least 11 real historical clicks and keeps the latest 50. Epoch counts, learning rates, and fusion activation are explicit reconstruction choices where the paper does not completely specify training details.

## BGN

[BGN §§4–5](https://arxiv.org/html/1904.01933) is implemented with item/category embeddings, a history CNN using widths `{1,2,4,8,12,16,32,64}` and 12 channels each, a two-layer attention LSTM, and a nonlinear feature-aware output matrix. Training uses sampled feature-aware cross-entropy with 1,024 negatives, Adam and the reported L2 coefficient. The one-negative objective is checked against its BPR equivalent.

Inference uses duplicate-item masking, the END penalty `max(C-t,0)`, beam pruning, and bundle-level Jaccard DPP selection at every expansion. Native defaults preserve variable length. `decoding="fixed_k"` suppresses END for the RABR comparison and records that evaluation adaptation. `predict_list` exposes the entire diversified list; `predict` returns its first bundle with the list in metadata. Single-bundle metrics do not measure the original list-diversity objective.

The Amazon profile requires finite catalog prices for descending-price target order, categories, and real purchase history. `target_order="provided_bundle_order"` is available for datasets with a declared native bundle order; `use_category=false` supports category-free datasets. No official implementation was found. The exact nonlinear feature map, attention initialization, learning rate and stopping budgets remain specified reconstruction choices, not recovered author checkpoints.

## BUNT

[BUNT / Bundle MCR](https://arxiv.org/abs/2207.12628) is implemented as an interactive model. The inspected official source is [Bundle-MCR at 7de7358](https://github.com/AaronHeee/Bundle-MCR/tree/7de735848729e847efd7cc89a8f3c61fe7f695e0). It uses separate category and attribute policies alongside item recommendation and a gated ask/recommend manager. Historical bundle and item Transformers have no positional encoding; the decoder combines accepted items and tags with historical bundle representations. Offline cloze training fits item, category, attribute, and successful decision objectives. Online PPO freezes the recommendation backbone, updates all four policies and their value heads, and uses binary side-policy feedback plus terminal conversation F1.

Accepted item slots remain in the conversation state and are replaced by new active slots. Category and attribute feedback is slot-specific; strong rejection excludes matching items globally. The default uses two active slots and at most 20 rounds. The `predict` call requires a `feedback_provider`; it never silently switches to one-shot inference. The fixed-K argument does not cap interactive output size.

Native item tags and historical bundle membership are mandatory. `RABR_BUNT_METADATA_PATH` names a JSON mapping each catalog ID to `{"categories": ["category-id"], "attributes": ["attribute-id"]}`. `RABR_BUNT_HISTORY_PATH` names a JSON mapping every session ID to a list of `{"timestamp": 123, "item_ids": ["item-id", "item-id"]}` records. Every history timestamp must precede the session timestamp; all IDs must belong to the supplied catalog. Missing history can be represented by an empty list and uses a learned empty-history token. No title-derived substitute tags, singleton history fabrication, or intent projection is used. Original clothing style metadata or corresponding real metadata for the target domain must be supplied; the raw BundleRec export alone does not contain every required asset.

A feedback provider receives either `{"turn": 1, "action": "recommend", "proposals": [{"slot": 0, "item_id": "..."}]}` or `{"turn": 1, "action": "ask", "queries": [{"slot": 0, "kind": "category", "tag": "..."}]}`. It returns `{"feedback": [{...same proposal/query..., "decision": "accept"}], "done": false}`; decisions can be `accept`, `reject`, or `ignore`. Responses may only refer to the submitted proposals or questions. Returning `None` ends the conversation.

`TargetFeedback` lives in a separate simulator module. Its protocol must explicitly be `training_simulation`, `validation_simulation`, or `heldout_evaluation`. The last option is a feedback-assisted evaluation: a separate evaluator sees held-out targets, but the model receives only the responses to the items and tags it actually asked about. This must be distinguished from target-free query/history evaluation. Predictions contain every proposed item, including rejected proposals. Reporting only accepted items would incorrectly force precision to one; the accepted subset is preserved separately as conversation metadata.

This is a new implementation of the published architecture and interactive algorithm, not the original authors' checkpoint or dataset split. Shared chronological train/validation partitions replace the native per-user offline/online split. Configurations specify offline epochs, online episodes and PPO updates; these units differ from the upstream command's online environment-step budget. A completed fit saves identity-bound weights. An interrupted unfinished fit restarts training. Native data and assets, full training and convergence validation are still required before claiming original-paper reproduction.

## BYOB

[BYOB](https://crazynote.v.netease.com/2022/0704/a0bd31b6913c14c50bd641d4f4ced703.pdf) uses item preference and compatibility rewards before bundle-reward fine-tuning. The implementation now follows that curriculum with undiscounted returns. In native BYOB, the neighbor threshold limits feasible actions; it is not a random-exploration probability. The RABR appendix instead calls its 0.2 setting an exploration rate. Only the explicit `rabr_fixed_k` adapter enables this distinct option: its policy mixes 20% uniform probability over feasible actions, and PPO uses that same mixed distribution for both rollout and update log probabilities. The neighbor threshold remains a separate parameter. Terminal reward uses the selected native bundle metric, defaulting to Recall, rather than F1.

`ItemPointer`, `ItemNCFModel`, and `SkipGramModel` are loaded unchanged from the [audited official revision](https://github.com/fuxiAIlab/BYOB/tree/538fb8de57fbc4f92dfb456b261a5150aab27121). The loader checks both commit and file contents, then loads only their definitions, avoiding obsolete Ray imports. NCF receives genuine user/item IDs and labeled training interactions. SGNS receives actual training click sequences, context-window pairs, and frequency-to-the-3/4 negative sampling. Frozen reward models train the pointer through clipped PPO. Policy embeddings receive gradients because observations are recomputed during updates.

The released NCF preference network differs from the paper's illustrated feature network; this integration selects the released implementation. The PyTorch PPO loop replaces RLlib, with explicit epoch/minibatch budgets and validation selection. It is not a claim that unspecified original optimizer settings or published performance have been recovered.

Obtain the external source without modifying it:

```sh
git clone https://github.com/fuxiAIlab/BYOB "$BYOB_SOURCE_PATH"
git -C "$BYOB_SOURCE_PATH" checkout 538fb8de57fbc4f92dfb456b261a5150aab27121
```

Set `BYOB_ASSETS_PATH` to an asset manifest:

```json
{
  "schema_version": 1,
  "training_session_ids": ["train-session-id"],
  "item_interactions": {"path": "item-feedback.json", "sha256": "<sha256>", "training_scope": "training_only", "provenance": "<feedback source>"},
  "click_sequences": {"path": "click-sequences.json", "sha256": "<sha256>", "training_scope": "training_only", "provenance": "<click source>"},
  "candidate_pools": {"path": "candidate-pools.json", "sha256": "<sha256>", "target_independent": true, "provenance": "<upstream candidate-generation procedure>"}
}
```

`item-feedback.json` is a list of `{session_id,user_id,item_id,label}` records, with binary positive/negative feedback and only fitting-session IDs. `click-sequences.json` contains `{session_id,items}` records from training histories. The candidate file maps each validation/prediction session ID to a distinct list of `pool_size` catalog IDs. It contains no labels. Candidate provenance must explain how held-out targets were excluded from pool construction; the declaration is checked but cannot establish that fact independently.

The default requires known persistent users, 20 observed history items, 20 candidates, and a three-item action horizon. Native mode requires equally sized training/validation targets. `evaluation_protocol="rabr_fixed_k"` allows variable target cardinality while retaining a fixed action horizon; the runner sets that horizon from the certified RABR calibration mean before fitting. Its terminal Recall divides by the full actual target size. The three reward signals remain organized into the original item-to-bundle curriculum because the RABR appendix does not specify a replacement combination formula. Defaults allow 50 epochs per policy stage, with validation early stopping. Singleton item/history dimensions are preserved at the released pointer input boundary to avoid the upstream `squeeze` shape failure for a one-item horizon. Native training pools include training positives plus sampled unpurchased negatives; prediction uses only the independently supplied pool. Unknown users, missing histories, missing pools, and empty feasible-neighbor sets fail explicitly. There is no full-catalog or intent-derived fallback.

## Persistence and scope

Completed fits save identity-bound weights and a receipt containing configuration, training/validation/catalog identities, source hashes and external-asset hashes. Identical retries restore those weights; changed inputs or files fail validation. An interrupted unfinished fit restarts training. The outer runner separately checkpoints completed predictions.

The supplied fresh BundleRec export does not by itself provide POG's original modalities, BGN's Amazon prices, BYOB's reward feedback/candidate pools, or BUNT's full interactive metadata. These are explicit prerequisites. Scratch checks verify equations, real gradient updates, valid decoding, failure gates and persistence; they do not establish native data equivalence, convergence or published scores.
