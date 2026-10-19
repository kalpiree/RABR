# DDBC

The default DDBC configuration uses CLHE catalog features, residual-quantized item codes, masked diffusion, and supplied seed items. It follows the released implementation at [LiAi16/DDBC](https://github.com/LiAi16/DDBC/tree/e1f4afc59121662e626ffefde79316a5e0dde044). CLHE features must be trained or obtained for the target catalog with compatible training provenance; sentence embeddings are not silently substituted.

## Feature assets

Set `RABR_DDBC_CLHE_MANIFEST` to a JSON manifest with the following structure:

```json
{
  "schema_version": 1,
  "feature_type": "clhe",
  "training_records_sha256": "SHA256_OF_CANONICAL_TRAIN_ROWS",
  "source": {
    "repository": "https://github.com/Xiaohao-Liu/CLHE",
    "commit": "CLHE_SOURCE_REVISION",
    "training_configuration_sha256": "SHA256_OF_TRAINING_CONFIGURATION"
  },
  "item_ids": {"path": "item_ids.json", "sha256": "FILE_SHA256"},
  "features": {"path": "clhe.npy", "sha256": "FILE_SHA256"}
}
```

`item_ids.json` contains string catalog IDs in feature-row order. `clhe.npy` is a finite floating-point matrix with one nonzero row for every catalog item. Files are relative to the manifest and must remain within its directory. Loading verifies file hashes, exact catalog coverage, and the canonical training-record hash, then reorders rows to catalog order. The training-record digest is `rabr.io.digest(dataset.train)` for the loaded prepared dataset. The manifest records upstream feature-training provenance; an identifier alone does not independently establish how an external feature file was trained.

The released small model uses 64-dimensional codebook weights. By default, features are already 64-dimensional and no PCA is applied, matching `sent_emb_pca: -1` in the released configuration. Explicit `pca_dimensions: 64` enables the released optional whitening transformation for higher-dimensional CLHE assets. Reduced dimensions below 64 are padded only for small software checks and are not the production default.

The released configuration uses three FAISS residual codebooks of size 256 and an occurrence-index fourth digit. The final paper describes size 128; this discrepancy is exposed through `codebook_size`, with the released-code value as the default. FAISS uses inner-product residual quantization. Collisions exceeding the fourth digit's capacity fail rather than changing the vocabulary silently.

## Seed inputs and outputs

Native completion takes an explicit nonempty `input.seed_items` list of distinct catalog IDs. The predictor accepts only this input field and never obtains seeds from target labels. `k` denotes the number of completion items under the default `output_mode: completion_only`. The `supplied_seed_completion` evaluation protocol requires withheld completion targets that exclude every supplied seed; overlap is rejected before fitting or prediction. Data preparation does not invent seed inputs from bundle targets or remove seeds from evaluation targets. This condition is reported separately from query-only full-bundle methods and is excluded from their paired significance tests.

`configs/methods/ddbc_query_adapter.json` instead declares a query-to-seed adapter: the shared sentence encoder retrieves one catalog seed from the query/history, CLHE remains the diffusion feature source, and the output contains the seed plus generated completion items. RQ1 explicitly selects this query-seed adapter so it can use the same query/history inputs and full-bundle targets as RABR. This changes the input protocol and must be labeled separately from supplied-seed construction. It is one configuration of DDBC, not a second baseline.

The earlier shared-text tokenizer and catalog-constrained sampler remain available only through explicit `feature_mode: shared_text` and `decoder: catalog_constrained` settings. They are adaptations and are not the native default.

## Training and decoding

Training uses the provided training bundles only. Adjacent swaps preserve bundle membership; length-weighted sampling gives examples equal marginal probability and avoids padding. Validation uses fixed masking randomness and selects the EMA checkpoint. This batching protocol supports variable-length recommendation data and differs from the upstream fixed-length crop loader. Training budgets, validation history, feature hashes, tokenizer hashes, and checkpoint identity are recorded.

The native decoder concatenates a supplied seed segment and masked completion segment. Structural tokens, seed codes, and revealed codes remain fixed. Each generated code is restricted only to its codebook level. It may form a tuple absent from the catalog; the decoder does not apply the stronger catalog-prefix feasibility constraints of the optional adapter. The reverse process preserves the released transition weights, caches unchanged denoiser outputs, and stops before negative transition times. Illegal categories receive zero probability, and code ranges are determined from segment positions rather than a fixed global offset.

The released evaluator constructs a candidate set containing withheld target items. This repository does not use that target-dependent candidate pool. It retrieves each decoded code against the entire catalog: known tuples use their catalog feature, and unknown tuples use the sum of the three residual codebook vectors. Retrieval uses cosine similarity, excludes supplied seeds and previously selected completions, and resolves ties by catalog order. This is an explicit evaluation adaptation; it must not be described as the original sampled-candidate evaluation.

No official CLHE features or trained DDBC checkpoints are bundled. Missing or inconsistent assets fail with an error before training. Producing new CLHE features requires the original CLHE implementation and training-only collaborative inputs for the intended catalog.
