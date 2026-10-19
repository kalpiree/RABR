# Text2Bundle

The implementation follows the learned components of [Text2Bundle](https://arxiv.org/html/2310.18004): intention decomposition, pretrained LightGCN user/item representations, RoBERTa text features, a Transformer state encoder, FM candidate ranking, an action-relation Transformer, BPR initialization and double DQN. It is not a dialogue-only generator. The state has six input types and uses position and type embeddings. Candidate actions include a learned termination action. Training rewards combine bundle precision, pretrained user/item compatibility, text compatibility and intention coverage.

The original paper uses the full Amazon domain to pretrain LightGCN. The repository now requires these external model assets instead of training a substitute graph from the small bundle training split. User/item FM features include linear and pairwise terms; a latent dot product alone is no longer substituted for the FM. The text path requires a RoBERTa checkpoint and never calls the common MiniLM encoder. Raw text vectors are retained for model inputs and reward dot products; only recall uses cosine normalization.

## Required assets

Set these variables before constructing the suite:

```bash
export RABR_T2B_LIGHTGCN_PATH=/path/to/lightgcn.npz
export RABR_T2B_FM_PATH=/path/to/fm.npz
export RABR_T2B_ROBERTA_PATH=/path/to/roberta-snapshot
export RABR_T2B_TEXT_PROJECTION_PATH=/path/to/text-projection.npz
```

The LightGCN NPZ must contain string vectors `user_ids`, `item_ids` and finite matrices `user_vectors`, `item_vectors`. Vector width must equal `hidden_size`. The item map may include additional Amazon items; it must cover the entire bundle catalog. Every evaluation user must have a pretrained representation. Missing users cause an error rather than a mean-user substitution.

The FM NPZ contains:

- String vectors `user_ids` and `item_ids`.
- Integer matrices `user_feature_indices` and `item_feature_indices`.
- Corresponding finite matrices `user_feature_values` and `item_feature_values`.
- Parameters `linear` of shape `[features]`, `factors` of shape `[features, factors]`, and scalar `bias`.

Each user/item row lists its nonzero features. Pad unused positions with index `-1` and value zero. User and item rows share one feature vocabulary. Duplicate indices are coalesced before computing the FM, including features present in both rows. Sparse rows avoid a dense user-by-feature matrix. The ranking score is the FM bias, linear term and all second-order feature interactions. Asset hashes are included in checkpoint identity. Keep training-corpus and feature-generation provenance alongside the assets; an NPZ file alone cannot prove where the model was trained.

The text checkpoint must load locally through Transformers with `model_type=roberta`. `text_pooling` accepts `mean` or `cls`. An optional pretrained projection NPZ contains `weight` with shape `[roberta_hidden_size, text_embedding_size]` and `bias` with shape `[text_embedding_size]`. If the RoBERTa snapshot already has the intended output width, remove `text_projection_path` from the method configuration. No projection is generated or randomly substituted to satisfy an expected dimension.

## Configuration and limits

The default profile uses ID width 320, text width 384, candidate count 50, recall cosine threshold 0.3, maximum generated size 5, main reward weight 3, one Transformer layer with two heads, learning rate 0.00005, replay capacity 20,000 and minibatch size 256. Native EOS generation uses its own maximum size; the ranking-baseline K argument does not truncate it.

The paper does not identify the exact 384-dimensional RoBERTa snapshot/projection. It also reports MLP widths `[64,64]` while specifying 320-dimensional ID embeddings; a literal 64-dimensional output cannot be added to those embeddings. This implementation makes that unresolved detail explicit: the projection uses one configurable hidden width and an output of `hidden_size`. Pooling, dropout, RL update count, target-update interval and checkpoint selection are recorded settings, not recovered author configuration.

The configured backbone performs intention decomposition with four or five pipe-separated entities and retains the original query. Malformed decomposition retains only that original query and records the raw response and parse status. The RABR experiment backbones differ from the original Text2Bundle ChatGLM setup. The shared chronological data partitions also differ from its random 8:1:1 split. Validation F1 selects checkpoints; no test labels enter reward construction or model selection.

Full Amazon pretraining assets, the intended RoBERTa feature transform and the original implementation are not bundled. Supplying validated assets enables the implemented training/inference path; matching the authors' published numbers still requires their original data/configuration and a full convergence study.
