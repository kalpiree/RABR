"""POG multimodal pretraining and BGN feature-aware generation."""
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
from .traditional_assets import MissingMethodAsset, historical_ids, load_pog_features


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


def transformer_encoder(dim, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class BGNFeatures(nn.Module):
    def __init__(self, catalog, config):
        super().__init__()
        self.n = len(catalog)
        self.dim = int(config.get("hidden_dim", 64))
        self.ids = {row["item_id"]: index for index, row in enumerate(catalog)}
        self.use_category = bool(config.get("use_category", True))
        self.item = nn.Embedding(self.n + 2, self.dim)
        if self.use_category:
            if any(row.get("category") is None for row in catalog):
                raise MissingMethodAsset("BGN's category profile requires catalog category IDs")
            self.categories = sorted({category(row) for row in catalog})
            self.category = nn.Embedding(len(self.categories) + 1, self.dim)
            self.register_buffer("category_ids", torch.tensor([self.categories.index(category(row)) for row in catalog] + [len(self.categories)] * 2))
        self.raw_dim = self.dim * (2 if self.use_category else 1)
        self.output_features = nn.Sequential(nn.Linear(self.raw_dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))

    def embed(self, indices):
        value = self.item(indices)
        return torch.cat((value, self.category(self.category_ids[indices])), -1) if self.use_category else value

    def all_items(self):
        indices = torch.arange(self.n + 2, device=self.item.weight.device)
        return self.output_features(self.embed(indices))

    def context(self, context):
        indices = historical_ids(inputs(context), self.ids)
        return self.embed(torch.tensor(indices, device=self.item.weight.device))[None]


class POGFeatures(nn.Module):
    def __init__(self, catalog, config, values):
        super().__init__()
        self.n = len(catalog)
        self.dim = int(config.get("hidden_dim", 64))
        self.embedding_dim = int(config.get("embedding_dim", 128))
        self.ids = {row["item_id"]: index for index, row in enumerate(catalog)}
        self.register_buffer("modalities", torch.as_tensor(values, dtype=torch.float32))
        self.fusion = nn.Sequential(nn.Linear(1996, self.embedding_dim), nn.ReLU())
        self.special = nn.Embedding(3, self.embedding_dim)
        self.minimum_history = int(config.get("minimum_history", 11))

    def all_items(self):
        return self.fusion(self.modalities)

    def embed(self, indices):
        return torch.cat((self.all_items(), self.special.weight), 0)[indices]

    def context(self, context):
        history = historical_ids(inputs(context), self.ids, maximum=50, minimum=self.minimum_history)
        return self.embed(torch.tensor(history, device=self.modalities.device))[None]


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
                        for path in Path(__file__).parent.glob("traditional*.py")}}
        identity["method_assets"] = getattr(self, "asset_hashes", {})
        self.fit_identity = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def build_features(self, catalog):
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


def transition(source, target):
    return nn.Sequential(nn.Linear(source, target), nn.ReLU(), nn.Linear(target, target))


class POGNetwork(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.features = features
        dim, heads = features.dim, int(config.get("heads", 8))
        layers, dropout = int(config.get("layers", 6)), float(config.get("dropout", .1))
        self.per_transition = transition(features.embedding_dim, dim)
        self.fom_transition = transition(features.embedding_dim, dim)
        self.gen_transition = transition(features.embedding_dim, dim)
        self.per = transformer_encoder(dim, heads, int(config.get("per_layers", layers)), dropout)
        self.fom = transformer_encoder(dim, heads, layers, dropout)
        self.gen = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True), layers)

    def candidates(self, stage="gen"):
        function = self.fom_transition if stage == "fom" else self.gen_transition
        ids = torch.arange(self.features.n + 1, device=self.features.modalities.device)
        return function(self.features.embed(ids))

    def next_logits(self, context, prefix):
        memory = self.per(self.per_transition(self.features.context(context)))
        ids = torch.tensor([self.features.n + 1] + list(prefix), device=memory.device)
        sequence = self.gen_transition(self.features.embed(ids))[None]
        mask = torch.triu(torch.ones(len(ids), len(ids), device=memory.device, dtype=torch.bool), diagonal=1)
        return self.gen(sequence, memory, tgt_mask=mask)[0] @ self.candidates().T

    def initialize_generator(self):
        self.gen_transition.load_state_dict(self.fom_transition.state_dict())
        for encoder, decoder in zip(self.fom.layers, self.gen.layers):
            decoder.self_attn.load_state_dict(encoder.self_attn.state_dict())
            decoder.linear1.load_state_dict(encoder.linear1.state_dict())
            decoder.linear2.load_state_dict(encoder.linear2.state_dict())
            decoder.norm1.load_state_dict(encoder.norm1.state_dict())
            decoder.norm3.load_state_dict(encoder.norm2.state_dict())


def sampled_ce(logits, target, excluded, count, rng):
    if count < 1:
        raise ValueError("Sampled softmax requires a positive negative sample count")
    negatives = [index for index in range(logits.shape[-1]) if index != target and index not in excluded]
    if not negatives:
        raise ValueError("No negative item is available for sampled softmax")
    choices = [target] + rng.sample(negatives, min(count, len(negatives)))
    return F.cross_entropy(logits[choices][None], torch.zeros(1, dtype=torch.long, device=logits.device))


class POG(Base):
    name = "pog"

    def build_features(self, catalog):
        values, self.asset_hashes = load_pog_features(self.config, catalog)
        return POGFeatures(catalog, self.config, values)

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        for row in list(train) + list(validation):
            historical_ids(row, self.index, maximum=50, minimum=self.features.minimum_history)
        self.model = POGNetwork(self.features, self.config).to(self.device)
        if self.restore_fit():
            self.features.fusion.requires_grad_(False)
            return self
        self.train_embedding()
        self.features.fusion.requires_grad_(False)
        self.train_stage(train, validation, self.fom_loss, int(self.config.get("fom_epochs", 10)), "all_position_masked_FOM")
        self.model.initialize_generator()
        self.model.fom.requires_grad_(False)
        self.model.fom_transition.requires_grad_(False)
        self.train_stage(train, validation, self.objective, int(self.config.get("epochs", 100)), "personalized_generation")
        return self.finish()

    def train_embedding(self):
        groups = {}
        for index, row in enumerate(self.catalog):
            if row.get("category") is None:
                raise MissingMethodAsset("POG triplet pretraining requires leaf category labels")
            groups.setdefault(category(row), []).append(index)
        anchors = [index for group in groups.values() if len(group) > 1 for index in group]
        if not anchors or len(groups) < 2:
            raise MissingMethodAsset("POG triplets require a same-category positive and a different-category negative")
        if int(self.config.get("embedding_epochs", 10)) < 1 or int(self.config.get("fom_epochs", 10)) < 1:
            raise ValueError("POG requires positive multimodal and FOM pretraining budgets")
        optimizer = torch.optim.Adam(self.features.fusion.parameters(), lr=float(self.config.get("embedding_learning_rate", .001)))
        for epoch in range(int(self.config.get("embedding_epochs", 10))):
            self.rng.shuffle(anchors)
            losses = []
            for start in range(0, len(anchors), int(self.config.get("batch_size", 16))):
                batch = anchors[start:start + int(self.config.get("batch_size", 16))]
                positive = [self.rng.choice([j for j in groups[category(self.catalog[i])] if j != i]) for i in batch]
                negative = [self.rng.choice([j for j in range(len(self.catalog)) if category(self.catalog[j]) != category(self.catalog[i])]) for i in batch]
                embedded = self.features.all_items()
                loss = F.triplet_margin_loss(embedded[batch], embedded[positive], embedded[negative], margin=.1, p=2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            self.training_history.append({"stage": "multimodal_category_triplet", "epoch": epoch + 1, "train_loss": sum(losses) / len(losses)})

    def fom_loss(self, row, rng):
        gold = self.ids(row)
        losses = []
        for position in range(len(gold)):
            masked = list(gold)
            masked[position] = len(self.catalog) + 2
            value = self.model.fom_transition(self.features.embed(torch.tensor(masked, device=self.device)))[None]
            output = self.model.fom(value)[0, position]
            logits = output @ self.model.candidates("fom")[:len(self.catalog)].T
            losses.append(sampled_ce(logits, gold[position], gold, 3, rng))
        return torch.stack(losses).mean()

    def objective(self, row, rng):
        gold = self.ids(row)
        logits = self.model.next_logits(self.row_context(row), gold)
        return torch.stack([sampled_ce(logits[t], item, gold, 3, rng) for t, item in enumerate(gold + [len(self.catalog)])]).mean()

    @torch.no_grad()
    def predict(self, context, k, seed):
        self.validate_predict(context, k)
        selected = []
        mode = self.config.get("decoding", "native_end")
        if mode not in {"native_end", "fixed_k"}:
            raise ValueError("POG decoding must be native_end or fixed_k")
        limit = k if mode == "fixed_k" else min(int(self.config.get("maximum_bundle_size", k)), len(self.catalog))
        if limit < 1:
            raise ValueError("maximum_bundle_size must be positive")
        terminated = False
        for _ in range(limit):
            scores = self.model.next_logits(context, selected)[-1].clone()
            scores[selected] = -torch.inf
            if mode == "fixed_k":
                scores[len(self.catalog)] = -torch.inf
            item = int(scores.argmax())
            if item == len(self.catalog):
                terminated = True
                break
            selected.append(item)
        return self.result(selected, decoding=mode, ended=terminated, maximum_size=limit, intent_used=False,
                           evaluation_adaptation="RABR_fixed_K_END_suppression" if mode == "fixed_k" else None)


def dpp_select(bundles, log_quality, count, diversity=.1):
    if len(bundles) != len(log_quality) or diversity < 0:
        raise ValueError("Invalid DPP inputs")
    selected, sets = [], [set(bundle) for bundle in bundles]
    for _ in range(min(count, len(bundles))):
        best = None
        for index in range(len(bundles)):
            if index in selected:
                continue
            if diversity == 0:
                score = log_quality[index]
            else:
                indices = selected + [index]
                matrix = torch.tensor([[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) if sets[a] | sets[b] else float(a == b) for b in indices] for a in indices], dtype=torch.float64)
                sign, determinant = torch.linalg.slogdet(matrix)
                score = log_quality[index] + diversity * float(determinant) if sign > 0 else -math.inf
            if best is None or score > best[0]:
                best = score, index
        if best is None or not math.isfinite(best[0]):
            break
        selected.append(best[1])
    return selected


class BGNNetwork(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.features = features
        dim = features.dim
        widths = (1, 2, 4, 8, 12, 16, 32, 64)
        self.convolutions = nn.ModuleList([nn.Conv1d(features.raw_dim, 12, width) for width in widths])
        self.context_projection = nn.Linear(96, dim)
        self.memory_projection = nn.Linear(features.raw_dim, dim, bias=False)
        self.lstm = nn.LSTM(features.raw_dim, dim, num_layers=2, batch_first=True)
        self.combine = nn.Linear(2 * dim, dim)

    def logits(self, context, prefix):
        memory = self.features.context(context)
        channels = memory.transpose(1, 2)
        pooled = torch.cat([F.relu(conv(F.pad(channels, (0, max(0, conv.kernel_size[0] - channels.shape[-1]))))).amax(-1) for conv in self.convolutions], -1)
        initial = torch.tanh(self.context_projection(pooled))[None].repeat(2, 1, 1)
        ids = torch.tensor([self.features.n + 1] + list(prefix), device=memory.device)
        outputs, _ = self.lstm(self.features.embed(ids)[None], (initial, torch.zeros_like(initial)))
        values = self.memory_projection(memory)
        attention = (outputs @ values.transpose(1, 2)).softmax(-1) @ values
        hidden = torch.tanh(self.combine(torch.cat((outputs, attention), -1)))[0]
        return hidden @ self.features.all_items()[:self.features.n + 1].T


class BGN(Base):
    name = "bgn"

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        for row in list(train) + list(validation):
            historical_ids(row, self.index)
        if self.config.get("target_order", "price_desc") == "price_desc":
            if any(type(row.get("price")) not in (int, float) or not math.isfinite(row["price"]) for row in catalog):
                raise MissingMethodAsset("BGN's Amazon profile requires catalog prices for descending-price target order")
        elif self.config.get("target_order") != "provided_bundle_order":
            raise ValueError("BGN target_order must be price_desc or provided_bundle_order")
        self.model = BGNNetwork(self.features, self.config).to(self.device)
        if self.restore_fit():
            return self
        self.train_stage(train, validation, self.objective, int(self.config.get("epochs", 100)), "sampled_feature_aware_sequence_MLE")
        return self.finish()

    def objective(self, row, rng):
        gold = self.ids(row)
        if self.config.get("target_order", "price_desc") == "price_desc":
            gold.sort(key=lambda index: -self.catalog[index]["price"])
        scores = self.model.logits(self.row_context(row), gold)
        return torch.stack([sampled_ce(scores[t], item, {item}, int(self.config.get("negative_samples", 1024)), rng) for t, item in enumerate(gold + [len(self.catalog)])]).mean()

    def masked_logits(self, logits, prefix, step):
        values = logits.clone()
        values[prefix] = -torch.inf
        values[len(self.catalog)] -= max(float(self.config.get("end_shift", 0)) - step, 0)
        if self.config.get("decoding", "native_end") == "fixed_k":
            values[len(self.catalog)] = -torch.inf
        return values

    @torch.no_grad()
    def predict_list(self, context, k, seed, list_size=None):
        self.validate_predict(context, k)
        width = int(self.config.get("beam_width", 50))
        count = int(list_size or self.config.get("bundle_list_size", 10))
        if width < count or count < 1:
            raise ValueError("beam_width must cover bundle_list_size")
        candidates = [([], 0., False)]
        mode = self.config.get("decoding", "native_end")
        if mode not in {"native_end", "fixed_k"}:
            raise ValueError("BGN decoding must be native_end or fixed_k")
        limit = k if mode == "fixed_k" else min(int(self.config.get("maximum_bundle_size", k)), len(self.catalog))
        if limit < 1:
            raise ValueError("maximum_bundle_size must be positive")
        for step in range(1, limit + 1):
            expanded = {}
            for prefix, quality, ended in candidates:
                if ended:
                    expanded[(tuple(sorted(prefix)), True)] = (prefix, quality, True)
                    continue
                scores = self.masked_logits(self.model.logits(context, prefix)[-1], prefix, step).log_softmax(-1)
                for item in torch.topk(scores, min(width, int(torch.isfinite(scores).sum()))).indices.tolist():
                    ended = item == len(self.catalog)
                    sequence = prefix if ended else prefix + [item]
                    score = quality + float(scores[item])
                    key = tuple(sorted(sequence)), ended
                    if key not in expanded or score > expanded[key][1]:
                        expanded[key] = sequence, score, ended
            top = sorted(expanded.values(), key=lambda value: (-value[1], value[0]))[:width]
            chosen = dpp_select([value[0] for value in top], [value[1] for value in top], count, float(self.config.get("diversity", .1)))
            candidates = [top[index] for index in chosen]
            if all(value[2] for value in candidates):
                break
        return [(items, score) for items, score, ended in candidates]

    def predict(self, context, k, seed):
        candidates = self.predict_list(context, k, seed)
        mode = self.config.get("decoding", "native_end")
        return self.result(candidates[0][0], bundle_list=[[self.item_ids[index] for index in bundle] for bundle, _ in candidates],
                           list_log_quality=[score for _, score in candidates], decoding=mode, intent_used=False,
                           evaluation_adaptation="RABR_fixed_K_END_suppression" if mode == "fixed_k" else None)
