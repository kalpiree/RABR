"""PCA whitening and residual-quantized catalog item identifiers."""

def pca_whiten(vectors, components=64):
    """Deterministic full-SVD PCA, sklearn's whiten=True scaling and sign rule."""
    import numpy as np
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) <= components or not np.isfinite(vectors).all():
        raise ValueError("PCA requires finite [N,D] input with N,D greater than components")
    mean = vectors.mean(axis=0)
    u, singular, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    signs = np.sign(u[np.argmax(abs(u), axis=0), np.arange(u.shape[1])])
    signs[signs == 0] = 1
    u *= signs
    vt *= signs[:, None]
    if singular[components - 1] <= np.finfo(float).eps * singular[0] * max(vectors.shape):
        raise ValueError("Catalog embedding rank is insufficient for declared PCA dimension")
    scale = singular[:components] / np.sqrt(len(vectors) - 1)
    reduced = ((vectors - mean) @ vt[:components].T) / scale
    return reduced.astype(np.float32), {"mean": mean, "components": vt[:components], "scale": scale}

def extend_codes(raw_codes, codebook_size):
    """Official occurrence-index fourth digit, deterministic catalog order."""
    import numpy as np
    raw = np.asarray(raw_codes)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Expected integer residual codes [catalog_size,3]")
    if (raw < 0).any() or (raw >= codebook_size).any():
        raise ValueError("Residual code outside book range")
    seen, result = {}, []
    for row in raw:
        key = tuple(int(value) for value in row)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > codebook_size:
            raise ValueError("RVQ collision multiplicity exceeds fourth digit capacity")
        result.append([value + digit * codebook_size + 1 for digit, value in enumerate(key)]
                      + [3 * codebook_size + seen[key]])
    return np.asarray(result, dtype=np.int64), max(seen.values(), default=0)

def faiss_quantize(vectors, codebook_size, seed, threads):
    import faiss
    import numpy as np
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("FAISS codebook size must be a power of two")
    if len(vectors) < codebook_size:
        raise ValueError("Catalog has fewer items than RVQ centroids")
    faiss.omp_set_num_threads(threads)
    bits = codebook_size.bit_length() - 1
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexResidualQuantizer(vectors.shape[1], 3, bits, faiss.METRIC_INNER_PRODUCT)
    index.rq.cp.seed = int(seed)
    index.train(vectors)
    index.add(vectors)
    packed = index.rq.compute_codes(vectors)
    raw = []
    for row in packed:
        reader = faiss.BitstringReader(faiss.swig_ptr(row), len(row))
        raw.append([reader.read(bits) for _ in range(3)])
    weights = faiss.vector_to_array(index.rq.codebooks).reshape(3 * codebook_size, vectors.shape[1])
    return np.asarray(raw, dtype=np.int64), weights, faiss.serialize_index(index).tobytes()


def load_clhe_features(manifest_path, catalog_ids, training_rows):
    import json
    import os
    from pathlib import Path
    import numpy as np
    from rabr.io import digest, file_hash
    value = os.path.expandvars(os.path.expanduser(str(manifest_path or '')))
    if not value or '$' in value:
        raise ValueError('Native DDBC requires a configured CLHE asset manifest')
    path = Path(value).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('feature_type') != 'clhe':
        raise ValueError('Expected a version-one CLHE feature manifest')
    if manifest.get('training_records_sha256') != digest(training_rows):
        raise ValueError('CLHE training provenance does not match the supplied training partition')
    source = manifest.get('source', {})
    if not source.get('repository') or not source.get('commit') or not source.get('training_configuration_sha256'):
        raise ValueError('CLHE features require source revision and training configuration provenance')
    loaded = {}
    for key in ('item_ids', 'features'):
        descriptor = manifest[key]
        file = (path.parent / descriptor['path']).resolve()
        if not file.is_relative_to(path.parent) or file_hash(file) != descriptor['sha256']:
            raise ValueError('CLHE asset path or hash mismatch: ' + key)
        loaded[key] = file
    ids = json.loads(loaded['item_ids'].read_text())
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(catalog_ids):
        raise ValueError('CLHE item IDs must cover the exact catalog once')
    vectors = np.load(loaded['features'], allow_pickle=False)
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
        raise ValueError('Invalid CLHE feature matrix')
    lookup = {item: index for index, item in enumerate(ids)}
    vectors = np.asarray(vectors[[lookup[item] for item in catalog_ids]], dtype=np.float32)
    if (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError('CLHE feature rows must be nonzero')
    return vectors, {'manifest_sha256': file_hash(path), 'features_sha256': manifest['features']['sha256'],
                     'item_ids_sha256': manifest['item_ids']['sha256'], 'source': source,
                     'training_records_sha256': manifest['training_records_sha256']}


def retrieve_generated_codes(generated_codes, catalog_codes, codebook_weights, catalog_features, codebook_size, excluded=()):
    import numpy as np
    codes = np.asarray(generated_codes)
    catalog_codes = np.asarray(catalog_codes)
    features = np.asarray(catalog_features, dtype=np.float64)
    weights = np.asarray(codebook_weights, dtype=np.float64)
    if codes.ndim != 2 or codes.shape[1] != 4 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Expected generated four-level integer codes')
    if features.ndim != 2 or len(features) != len(catalog_codes) or weights.shape != (3 * codebook_size, features.shape[1]):
        raise ValueError('Codebook and catalog feature dimensions differ')
    known = {tuple(row): index for index, row in enumerate(catalog_codes.tolist())}
    unavailable = set(int(index) for index in excluded)
    if len(codes) > len(catalog_codes) - len(unavailable):
        raise ValueError('Insufficient catalog items for distinct completion retrieval')
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or (norms == 0).any():
        raise ValueError('Catalog features must be finite and nonzero')
    normalized = features / norms[:, None]
    selected, unmatched = [], 0
    for row in codes:
        if any(not level * codebook_size < int(token) <= (level + 1) * codebook_size for level, token in enumerate(row)):
            raise ValueError('Generated code violates its level range')
        index = known.get(tuple(row.tolist()))
        if index is None:
            vector = weights[row[:3] - 1].sum(axis=0)
            unmatched += 1
        else:
            vector = features[index]
        norm = np.linalg.norm(vector)
        scores = normalized @ (vector / norm) if norm > 0 else np.zeros(len(features))
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        chosen = next(index for index in order if index not in unavailable)
        selected.append(chosen)
        unavailable.add(chosen)
    return selected, {'unmatched_code_tuples': unmatched, 'retrieval': 'full_catalog_cosine_distinct_excluding_supplied_seeds'}
