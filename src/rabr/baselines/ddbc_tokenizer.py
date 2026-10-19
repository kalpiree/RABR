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
