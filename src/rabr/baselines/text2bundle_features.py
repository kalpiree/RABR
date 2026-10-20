from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from rabr.io import file_hash


def asset_path(config, key):
    value = config.get(key)
    if not value:
        raise ValueError(f"Text2Bundle requires {key}; shared-encoder and train-only substitutes are not supported")
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.exists():
        raise FileNotFoundError(f"Text2Bundle asset missing: {key}: {path}")
    return path


def _strings(values, name):
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(name + " must be a string vector")
    result = [str(value) for value in values.astype(str)]
    if len(result) != len(set(result)) or not all(result):
        raise ValueError(name + " must contain distinct nonempty IDs")
    return result


def factorization_machine(user_features, item_features, linear, factors, bias):
    features = np.asarray(item_features) + np.asarray(user_features)
    interactions = .5 * ((features @ factors) ** 2 - features ** 2 @ (factors ** 2)).sum(-1)
    return np.asarray(bias).item() + features @ linear + interactions


class NativeInteractions:
    def __init__(self, config, catalog_ids):
        graph_path, fm_path = asset_path(config, "lightgcn_assets_path"), asset_path(config, "fm_assets_path")
        with np.load(graph_path, allow_pickle=False) as graph:
            if set(graph.files) != {"user_ids", "item_ids", "user_vectors", "item_vectors"}:
                raise ValueError("LightGCN assets require user_ids,item_ids,user_vectors,item_vectors")
            user_ids, item_ids = _strings(graph["user_ids"], "user_ids"), _strings(graph["item_ids"], "item_ids")
            self.users = {user: index for index, user in enumerate(user_ids)}
            lookup = {item: index for index, item in enumerate(item_ids)}
            if not set(catalog_ids) <= lookup.keys():
                raise ValueError("LightGCN assets do not cover the bundle catalog")
            self.user_vectors = np.asarray(graph["user_vectors"], dtype=np.float32)
            all_items = np.asarray(graph["item_vectors"], dtype=np.float32)
            if (self.user_vectors.ndim != 2 or all_items.ndim != 2 or len(self.user_vectors) != len(user_ids)
                    or len(all_items) != len(item_ids) or self.user_vectors.shape[1] != all_items.shape[1]
                    or self.user_vectors.shape[1] != int(config.get("hidden_size", 320))):
                raise ValueError("LightGCN vector dimensions differ from IDs/hidden_size")
            self.item_vectors = all_items[[lookup[item] for item in catalog_ids]]
        with np.load(fm_path, allow_pickle=False) as fm:
            if set(fm.files) != {"user_ids", "item_ids", "user_feature_indices", "user_feature_values", "item_feature_indices", "item_feature_values", "linear", "factors", "bias"}:
                raise ValueError("FM assets require IDs, sparse feature indices/values, linear weights, factors and bias")
            fm_users, fm_items = _strings(fm["user_ids"], "fm user_ids"), _strings(fm["item_ids"], "fm item_ids")
            user_lookup, item_lookup = {user: i for i, user in enumerate(fm_users)}, {item: i for i, item in enumerate(fm_items)}
            if not set(user_ids) <= user_lookup.keys() or not set(catalog_ids) <= item_lookup.keys():
                raise ValueError("FM assets do not cover LightGCN users and bundle catalog")
            self.linear, self.factors, self.bias = (np.asarray(fm[key], dtype=np.float32) for key in ("linear", "factors", "bias"))
            if self.linear.ndim != 1 or self.factors.ndim != 2 or self.factors.shape[0] != len(self.linear) or self.bias.size != 1:
                raise ValueError("FM parameter dimensions differ")
            for kind, ids, lookup, requested in (("user", fm_users, user_lookup, user_ids), ("item", fm_items, item_lookup, catalog_ids)):
                indices, values = fm[kind + "_feature_indices"], np.asarray(fm[kind + "_feature_values"], dtype=np.float32)
                if (indices.ndim != 2 or len(indices) != len(ids) or indices.dtype.kind not in {"i", "u"}
                        or indices.shape != values.shape or (indices < -1).any() or (indices >= len(self.linear)).any()
                        or ((indices == -1) & (values != 0)).any() or not np.isfinite(values).all()):
                    raise ValueError("Invalid sparse FM feature rows")
                order = [lookup[value] for value in requested]
                setattr(self, kind + "_indices", indices[order].astype(np.int64))
                setattr(self, kind + "_values", values[order])
        for value in (self.user_vectors, self.item_vectors, self.linear, self.factors, self.bias):
            if not np.isfinite(value).all():
                raise ValueError("Native Text2Bundle assets must be finite")
        self.hashes = {"lightgcn": file_hash(graph_path), "fm": file_hash(fm_path)}

    def rank(self, user, items):
        if user < 0 or user >= len(self.user_indices):
            raise ValueError("Text2Bundle requires the user's pretrained graph and FM features")
        scores = []
        for item in items:
            indices = np.concatenate((self.user_indices[user], self.item_indices[item]))
            values = np.concatenate((self.user_values[user], self.item_values[item]))
            keep = indices >= 0
            unique, inverse = np.unique(indices[keep], return_inverse=True)
            summed = np.zeros(len(unique), dtype=np.float32)
            np.add.at(summed, inverse, values[keep])
            vectors = self.factors[unique]
            interaction = .5 * (((summed[:, None] * vectors).sum(0) ** 2) - ((summed[:, None] * vectors) ** 2).sum(0)).sum()
            scores.append(float(self.bias.item() + self.linear[unique] @ summed + interaction))
        return np.asarray(scores, dtype=np.float32)


class RobertaEncoder:
    def __init__(self, config):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        path = asset_path(config, "text_encoder_path")
        model_config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if model_config.model_type != "roberta":
            raise ValueError("Text2Bundle requires a RoBERTa text checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(config.get("text_device", config.get("device", "cpu"))).eval()
        self.config, self.projection = dict(config), None
        self.pooling = config.get("text_pooling", "mean")
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("text_pooling must be mean or cls")
        if config.get("text_projection_path"):
            projection_path = asset_path(config, "text_projection_path")
            with np.load(projection_path, allow_pickle=False) as projection:
                if set(projection.files) != {"weight", "bias"}:
                    raise ValueError("Text projection requires weight and bias")
                weight, bias = np.asarray(projection["weight"], dtype=np.float32), np.asarray(projection["bias"], dtype=np.float32)
                if weight.ndim != 2 or weight.shape[0] != model_config.hidden_size or bias.shape != (weight.shape[1],):
                    raise ValueError("Text projection dimensions differ")
                if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                    raise ValueError("Text projection must be finite")
                self.projection = weight, bias
        dimension = self.projection[0].shape[1] if self.projection else model_config.hidden_size
        if dimension != int(config.get("text_embedding_size", 384)):
            raise ValueError("RoBERTa output dimension differs from text_embedding_size; supply the intended pretrained projection")
        self.dimension = dimension
        self.identity = {"model_config_sha256": file_hash(path / "config.json"), "pooling": self.pooling,
                         "dimension": dimension, "projection_sha256": file_hash(projection_path) if self.projection else None,
                         "snapshot": {str(p.relative_to(path)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                      for p in sorted(path.rglob("*")) if p.is_file()}}

    def encode(self, texts):
        import torch
        values = []
        for start in range(0, len(texts), int(self.config.get("text_batch_size", 64))):
            tokens = self.tokenizer(list(texts[start:start + int(self.config.get("text_batch_size", 64))]), padding=True,
                                    truncation=True, max_length=int(self.config.get("text_max_length", 256)), return_tensors="pt")
            tokens = {key: value.to(next(self.model.parameters()).device) for key, value in tokens.items()}
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                array = pooled.float().cpu().numpy()
            if self.projection:
                array = array @ self.projection[0] + self.projection[1]
            values.append(array)
        output = np.concatenate(values) if values else np.empty((0, self.dimension), dtype=np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite RoBERTa embeddings")
        return output
