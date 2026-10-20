from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


class MissingMethodAsset(ValueError):
    pass


def read_manifest(config, key):
    value = os.path.expandvars(os.path.expanduser(str(config.get(key, ""))))
    if not value or "$" in value or not Path(value).is_file():
        raise MissingMethodAsset(f"Set algorithm.{key} to the required asset manifest; no synthetic feature fallback is available")
    path = Path(value).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported method asset manifest version")
    return path, manifest


def verified_file(manifest_path, entry):
    path = (manifest_path.parent / entry["path"]).resolve()
    if not path.is_file():
        raise MissingMethodAsset(f"Missing method asset: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry.get("sha256"):
        raise ValueError(f"Method asset SHA256 mismatch: {path.name}")
    return path


def load_pog_features(config, catalog):
    path, manifest = read_manifest(config, "pog_assets_path")
    ids = manifest.get("item_ids", [])
    if len(ids) != len(set(ids)) or set(ids) != {row["item_id"] for row in catalog}:
        raise ValueError("POG modality assets must cover exactly the supplied catalog IDs")
    order = [ids.index(row["item_id"]) for row in catalog]
    sources = {"image": (1536, "InceptionResNetV2"), "text": (300, "TextCNN-category"), "cf": (160, "Behemoth")}
    arrays, hashes = [], {"manifest": hashlib.sha256(path.read_bytes()).hexdigest()}
    for key, (dimension, extractor) in sources.items():
        entry = manifest.get(key, {})
        if entry.get("extractor") != extractor or not entry.get("provenance"):
            raise MissingMethodAsset(f"POG requires documented {extractor} {key} features; alternate encoders are not this profile")
        if key in {"text", "cf"} and entry.get("training_scope") != "training_only":
            raise ValueError(f"POG {key} feature fitting must be declared training_only")
        source = verified_file(path, entry)
        values = np.load(source, allow_pickle=False)
        if values.shape != (len(ids), dimension) or not np.isfinite(values).all():
            raise ValueError(f"POG {key} requires finite [{len(ids)},{dimension}] vectors")
        arrays.append(values[order].astype(np.float32))
        hashes[key] = entry["sha256"]
    return np.concatenate(arrays, axis=1), hashes


def historical_ids(context, index, *, maximum=None, minimum=1):
    data = context.get("input", context)
    history = data.get("history")
    if not isinstance(history, list):
        raise MissingMethodAsset("This model requires a genuine item-interaction history")
    timestamped = [isinstance(entry, dict) and "timestamp" in entry for entry in history]
    if any(timestamped):
        if not all(timestamped) or any(isinstance(entry["timestamp"], bool) or not isinstance(entry["timestamp"], (int, float)) or not math.isfinite(entry["timestamp"]) for entry in history):
            raise ValueError("Timestamped histories require finite numeric timestamps for every event")
        history = sorted(history, key=lambda entry: entry["timestamp"])
    ids = [entry.get("item_id") if isinstance(entry, dict) else entry for entry in history]
    if any(item not in index for item in ids):
        raise ValueError("Historical item is outside the catalog")
    if len(ids) < minimum:
        raise MissingMethodAsset(f"This model requires at least {minimum} historical items; intent is not a replacement")
    return [index[item] for item in (ids[-maximum:] if maximum else ids)]
