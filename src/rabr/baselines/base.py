from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def targets(row):
    items = row.get("target_items", row.get("target", {}).get("gold_items"))
    if not isinstance(items, list) or not items:
        raise ValueError("Training/validation rows require a nonempty target.gold_items list")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("Item IDs must be nonempty strings")
    return list(dict.fromkeys(items))


def inputs(context):
    if any(key in context for key in ("target", "target_items", "gold_items")):
        raise ValueError("Prediction context must not contain target labels")
    return context.get("input", context)


def category(row):
    value = row.get("category", "unknown")
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


class Base:
    name = "base"

    def __init__(self, config):
        self.config = config
        self.training_history = []
        self.fitted = False

    def setup(self, train, validation, catalog):
        if not train or not validation or not catalog:
            raise ValueError("Training, validation, and catalog cannot be empty")
        self.catalog = list(catalog)
        self.item_ids = [r["item_id"] for r in catalog]
        if len(set(self.item_ids)) != len(self.item_ids) or any(not isinstance(i, str) for i in self.item_ids):
            raise ValueError("Catalog needs unique string item IDs")
        self.index = {item: i for i, item in enumerate(self.item_ids)}
        train_ids, validation_ids = [row["session_id"] for row in train], [row["session_id"] for row in validation]
        if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids) or set(train_ids) & set(validation_ids):
            raise ValueError("Training and validation session IDs must be unique and disjoint")
        for row in list(train) + list(validation):
            if not set(targets(row)) <= self.index.keys():
                raise ValueError("Target item is outside the supplied catalog")
        self.seed = int(self.config.get("seed", 42))
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        if "cpu_threads" in self.config:
            threads = int(self.config["cpu_threads"])
            if threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(threads)
        self.device = torch.device(self.config.get("device", "cpu"))
        dim, heads = int(self.config.get("hidden_dim", 64)), int(self.config.get("heads", 4))
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if int(self.config.get("epochs", 100)) < 1 or int(self.config.get("batch_size", 16)) < 1:
            raise ValueError("epochs and batch_size must be positive")
        self.features = self.build_features(catalog)
        if self.features is not None:
            self.features.to(self.device)
        self.fit_counts = {"train": len(train), "validation": len(validation), "catalog": len(catalog)}
        identity = {"version": 2, "method": self.name, "config": {k: v for k, v in self.config.items() if k not in {"work_dir", "backend"}},
                    "train": train, "validation": validation, "catalog": catalog,
                    "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__).with_name(name) for name in ("traditional.py", "base.py", "pog.py", "bgn.py", "byob.py", "features.py", "byob_models.py"))}}
        identity["method_assets"] = getattr(self, "asset_hashes", {})
        self.fit_identity = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def build_features(self, catalog):
        from .bgn import BGNFeatures
        return BGNFeatures(catalog, self.config)

    def restore_fit(self):
        if not self.config.get("work_dir"):
            return False
        directory = Path(self.config["work_dir"])
        metadata = directory / "traditional-fit.json"
        if not metadata.exists():
            return False
        state = json.loads(metadata.read_text())
        weights = directory / "traditional-model.pt"
        if state["identity_sha256"] != self.fit_identity:
            raise ValueError("Completed fit identity differs from requested training inputs/configuration")
        if hashlib.sha256(weights.read_bytes()).hexdigest() != state["weights_sha256"]:
            raise ValueError("Completed fit checkpoint hash mismatch")
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.training_history = state["training_history"]
        self.fit_metadata = self.fit_summary(state["weights_sha256"])
        self.model.eval()
        self.fitted = True
        return True

    def ids(self, row):
        return [self.index[i] for i in targets(row)]

    def row_context(self, row):
        return {"session_id": row["session_id"], "input": row.get("input", {}), **({"user_id": row["user_id"]} if "user_id" in row else {})}

    def train_stage(self, train, validation, objective, epochs, stage):
        if epochs <= 0:
            return
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                     lr=float(self.config.get("learning_rate", 3e-4)))
        batch_size = int(self.config.get("batch_size", 16))
        best, stale, selected = math.inf, 0, None
        for epoch in range(epochs):
            order = list(range(len(train)))
            self.rng.shuffle(order)
            self.model.train()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [objective(train[i], self.rng) for i in order[start:start+batch_size]]
                loss = torch.stack(batch).mean()
                if self.config.get("l2_regularization", 0):
                    loss = loss + float(self.config["l2_regularization"]) * sum(parameter.square().sum() for parameter in self.model.parameters() if parameter.requires_grad)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip", 1.)))
                optimizer.step()
                losses.extend(float(x.detach()) for x in batch)
            self.model.eval()
            with torch.no_grad():
                checks = [float(objective(row, random.Random(self.seed + i))) for i, row in enumerate(validation)]
            score = sum(checks) / len(checks) if checks else sum(losses) / len(losses)
            self.training_history.append({"stage": stage, "epoch": epoch+1, "train_loss": sum(losses)/len(losses),
                                          "validation_loss": score if checks else None})
            if score < best:
                best, stale, selected = score, 0, copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if validation and stale >= int(self.config.get("patience", 10)):
                    break
        self.model.load_state_dict(selected)
        self.model.eval()

    def finish(self):
        self.fitted = True
        self.model.eval()
        weight_hash = None
        if self.config.get("work_dir"):
            directory = Path(self.config["work_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "traditional-model.pt.tmp"
            torch.save(self.model.state_dict(), temporary)
            weight_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(directory / "traditional-model.pt")
            metadata = directory / "traditional-fit.json.tmp"
            metadata.write_text(json.dumps({"identity_sha256": self.fit_identity, "weights_sha256": weight_hash,
                                           "training_history": self.training_history}, indent=2, allow_nan=False) + "\n")
            metadata.replace(directory / "traditional-fit.json")
        self.fit_metadata = self.fit_summary(weight_hash)
        return self

    def fit_summary(self, weight_hash):
        return {"method": self.name, "implementation": self.config.get("implementation", "paper_equations"), "fit_identity_sha256": self.fit_identity,
                "counts": self.fit_counts, "model_class": type(self.model).__name__,
                "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
                "training_history": self.training_history, "weights_sha256": weight_hash,
                "feature_profile": self.config.get("feature_profile", "native_inputs"), "asset_hashes": getattr(self, "asset_hashes", {})}

    def validate_predict(self, context, k):
        inputs(context)
        if not self.fitted:
            raise RuntimeError("fit must finish before predict")
        if not isinstance(k, int) or not 1 <= k <= len(self.catalog):
            raise ValueError("k must be within catalog size")
        self.model.eval()

    def result(self, indices, **metadata):
        from rabr.types import Prediction
        return Prediction(items=[self.item_ids[i] for i in indices], metadata={"method": self.name,
            "implementation": self.config.get("implementation", "paper_equations"), "training_epochs": len(self.training_history),
            "feature_profile": self.config.get("feature_profile", "native_inputs"), "asset_hashes": getattr(self, "asset_hashes", {}),
            "llm_calls": 0, **metadata})


def sampled_ce(logits, target, excluded, count, rng):
    if count < 1:
        raise ValueError("Sampled softmax requires a positive negative sample count")
    negatives = [index for index in range(logits.shape[-1]) if index != target and index not in excluded]
    if not negatives:
        raise ValueError("No negative item is available for sampled softmax")
    choices = [target] + rng.sample(negatives, min(count, len(negatives)))
    return F.cross_entropy(logits[choices][None], torch.zeros(1, dtype=torch.long, device=logits.device))
