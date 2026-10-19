"""Transformer POG and recurrent feature-aware BGN reconstructions."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import re
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


def words(text):
    return re.findall(r"[\w]+", str(text).lower())


def hash_features(text, size):
    values = [0.] * size
    for token in words(text):
        digest = hashlib.sha256(token.encode()).digest()
        values[int.from_bytes(digest[:4], "big") % size] += 1. if digest[4] & 1 else -1.
    norm = math.sqrt(sum(x*x for x in values)) or 1.
    return [x / norm for x in values]


def f1(prediction, gold):
    return 2 * len(set(prediction) & set(gold)) / (len(set(prediction)) + len(set(gold))) if prediction else 0.


def transformer_encoder(dim, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Features(nn.Module):
    def __init__(self, catalog, config):
        super().__init__()
        self.n = len(catalog)
        self.dim = int(config.get("hidden_dim", 64))
        self.feature_dim = int(config.get("feature_dim", 128))
        self.ids = {r["item_id"]: i for i, r in enumerate(catalog)}
        self.categories = sorted({category(r) for r in catalog})
        self.register_buffer("category_ids", torch.tensor([self.categories.index(category(r)) for r in catalog]))
        self.item = nn.Embedding(self.n + 3, self.dim)
        self.category = nn.Embedding(len(self.categories), self.dim)
        self.text_projection = nn.Linear(self.feature_dim, self.dim, bias=False)
        values = [hash_features(str(r.get("title", "")) + " " + category(r), self.feature_dim) for r in catalog]
        if config.get("item_features_path"):
            supplied = json.loads(Path(config["item_features_path"]).read_text())
            values = [supplied[r["item_id"]] for r in catalog]
            if any(len(v) != self.feature_dim or not all(math.isfinite(float(x)) for x in v) for v in values):
                raise ValueError("External item features must be finite feature_dim vectors for every catalog ID")
        self.register_buffer("text", torch.tensor(values, dtype=torch.float32))
        self.transition = nn.Sequential(nn.Linear(3 * self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))

    def all_items(self):
        indices = torch.arange(self.n, device=self.item.weight.device)
        return self.transition(torch.cat((self.item(indices), self.category(self.category_ids), self.text_projection(self.text)), -1))

    def embed(self, indices):
        all_values = torch.cat((self.all_items(), self.item.weight[self.n:]), 0)
        return all_values[indices]

    def context(self, context):
        data = inputs(context)
        history = [h.get("item_id") if isinstance(h, dict) else h for h in data.get("history", [])]
        history = [self.ids[i] for i in history if i in self.ids][-100:]
        intent = torch.tensor(hash_features(data.get("intent", ""), self.feature_dim), device=self.text.device)
        query = self.text_projection(intent)[None]
        if history:
            query = torch.cat((query, self.embed(torch.tensor(history, device=self.text.device))), 0)
        return query[None]


class Base:
    name = "base"

    def __init__(self, config):
        self.config = config
        self.training_history = []
        self.fitted = False

    def setup(self, train, validation, catalog):
        if not train or not catalog:
            raise ValueError("Training data and catalog cannot be empty")
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
        self.features = Features(catalog, self.config).to(self.device)
        self.fit_counts = {"train": len(train), "validation": len(validation), "catalog": len(catalog)}
        identity = {"version": 1, "method": self.name, "config": {k: v for k, v in self.config.items() if k not in {"work_dir", "backend"}},
                    "train": train, "validation": validation, "catalog": catalog,
                    "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in Path(__file__).parent.glob("traditional*.py")}}
        if self.config.get("item_features_path"):
            identity["external_features_sha256"] = hashlib.sha256(Path(self.config["item_features_path"]).read_bytes()).hexdigest()
        self.fit_identity = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

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
        return {"input": row.get("input", {})}

    def train_stage(self, train, validation, objective, epochs, stage):
        if epochs <= 0:
            return
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=float(self.config.get("learning_rate", 3e-4)), weight_decay=float(self.config.get("weight_decay", 1e-5)))
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
        return {"method": self.name, "implementation": "paper_based_reconstruction", "fit_identity_sha256": self.fit_identity,
                "counts": self.fit_counts, "model_class": type(self.model).__name__,
                "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
                "training_history": self.training_history, "weights_sha256": weight_hash,
                "feature_profile": "external_item_features" if self.config.get("item_features_path") else "catalog_hash_text_category"}

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
            "implementation": "paper_based_reconstruction", "training_epochs": len(self.training_history),
            "feature_profile": "external_item_features" if self.config.get("item_features_path") else "catalog_hash_text_category",
            "llm_calls": 0, **metadata})


class POGNetwork(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.features = features
        dim, heads = features.dim, int(config.get("heads", 8))
        layers, dropout = int(config.get("layers", 6)), float(config.get("dropout", .1))
        self.per = transformer_encoder(dim, heads, layers, dropout)
        self.fom = transformer_encoder(dim, heads, layers, dropout)
        self.gen = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True), layers)
        self.end = nn.Parameter(torch.randn(dim) * .02)

    def candidates(self):
        return torch.cat((self.features.all_items(), self.end[None]), 0)

    def next_logits(self, context, prefix):
        memory = self.per(self.features.context(context))
        ids = torch.tensor([self.features.n] + list(prefix), device=self.end.device)
        sequence = self.features.embed(ids)[None]
        mask = torch.triu(torch.ones(len(ids), len(ids), device=self.end.device, dtype=torch.bool), diagonal=1)
        output = self.gen(sequence, memory, tgt_mask=mask)
        return output[0] @ self.candidates().T

    def initialize_generator(self):
        for encoder, decoder in zip(self.fom.layers, self.gen.layers):
            decoder.self_attn.load_state_dict(encoder.self_attn.state_dict())
            decoder.linear1.load_state_dict(encoder.linear1.state_dict())
            decoder.linear2.load_state_dict(encoder.linear2.state_dict())
            decoder.norm1.load_state_dict(encoder.norm1.state_dict())
            decoder.norm3.load_state_dict(encoder.norm2.state_dict())


class POG(Base):
    name = "pog"

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        self.model = POGNetwork(self.features, self.config).to(self.device)
        if self.restore_fit():
            return self
        self.train_stage(train, validation, self.fom_loss, int(self.config.get("fom_epochs", 10)), "masked_item_pretraining")
        self.model.initialize_generator()
        self.train_stage(train, validation, self.objective, int(self.config.get("epochs", 100)), "autoregressive_generation")
        return self.finish()

    def sampled_loss(self, logits, target, excluded, rng):
        negatives = [i for i in range(logits.shape[-1]) if i not in excluded and i != target]
        count = int(self.config.get("negative_samples", 3))
        choices = [target] + rng.sample(negatives, min(count, len(negatives))) if count else list(range(logits.shape[-1]))
        label = choices.index(target)
        return F.cross_entropy(logits[choices][None], torch.tensor([label], device=self.device))

    def fom_loss(self, row, rng):
        gold = self.ids(row)
        position = rng.randrange(len(gold))
        masked = list(gold)
        masked[position] = len(self.catalog) + 1
        output = self.model.fom(self.features.embed(torch.tensor(masked, device=self.device))[None])[0, position]
        logits = output @ self.features.all_items().T
        return self.sampled_loss(logits, gold[position], gold, rng)

    def objective(self, row, rng):
        gold = self.ids(row)
        logits = self.model.next_logits(self.row_context(row), gold)
        expected = gold + [len(self.catalog)]
        return torch.stack([self.sampled_loss(logits[t], item, gold, rng) for t, item in enumerate(expected)]).mean()

    @torch.no_grad()
    def predict(self, context, k, seed):
        self.validate_predict(context, k)
        selected = []
        for _ in range(k):
            scores = self.model.next_logits(context, selected)[-1, :len(self.catalog)].clone()
            scores[selected] = -torch.inf
            selected.append(int(scores.argmax()))
        return self.result(selected, decoding="greedy_fixed_k", native_end_token_suppressed=True)


def dpp_select(bundles, log_quality, count, diversity=.1):
    """Greedy log-quality + lambda log-det selection, using bundle Jaccard."""
    if len(bundles) != len(log_quality) or not 0 <= diversity:
        raise ValueError("Invalid DPP inputs")
    selected = []
    sets = [set(x) for x in bundles]
    for _ in range(min(count, len(bundles))):
        best = None
        for i in range(len(bundles)):
            if i in selected:
                continue
            indices = selected + [i]
            kernel = torch.tensor([[len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
                                    for b in indices] for a in indices], dtype=torch.float64)
            sign, determinant = torch.linalg.slogdet(kernel)
            score = log_quality[i] + diversity * float(determinant) if sign > 0 else -math.inf
            if best is None or score > best[0]:
                best = (score, i)
        if best is None or not math.isfinite(best[0]):
            break
        selected.append(best[1])
    return selected


class BGNNetwork(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.features = features
        dim = features.dim
        self.convolutions = nn.ModuleList([nn.Conv1d(dim, dim, width, padding=width//2) for width in (1, 3, 5)])
        self.context_projection = nn.Linear(3*dim, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=int(config.get("layers", 2)), batch_first=True)
        self.combine = nn.Linear(2*dim, dim)
        self.end = nn.Parameter(torch.randn(dim) * .02)

    def logits(self, context, prefix):
        memory = self.features.context(context)
        pooled = torch.cat([F.relu(conv(memory.transpose(1, 2))).amax(-1) for conv in self.convolutions], -1)
        initial = torch.tanh(self.context_projection(pooled))[None].repeat(self.lstm.num_layers, 1, 1)
        ids = torch.tensor([self.features.n] + list(prefix), device=memory.device)
        outputs, _ = self.lstm(self.features.embed(ids)[None], (initial, torch.zeros_like(initial)))
        attention = (outputs @ memory.transpose(1, 2)).softmax(-1) @ memory
        hidden = torch.tanh(self.combine(torch.cat((outputs, attention), -1)))[0]
        candidate_features = torch.cat((self.features.all_items(), self.end[None]), 0)
        return hidden @ candidate_features.T


class BGN(Base):
    name = "bgn"

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        self.model = BGNNetwork(self.features, self.config).to(self.device)
        if self.restore_fit():
            return self
        self.train_stage(train, validation, self.objective, int(self.config.get("epochs", 100)), "feature_aware_sequence_mle")
        return self.finish()

    def objective(self, row, rng):
        gold = self.ids(row)
        scores = self.model.logits(self.row_context(row), gold)
        return F.cross_entropy(scores, torch.tensor(gold + [len(self.catalog)], device=self.device))

    @torch.no_grad()
    def predict_list(self, context, k, seed, list_size=None):
        self.validate_predict(context, k)
        width = int(self.config.get("beam_width", 20))
        output_count = int(list_size or self.config.get("bundle_list_size", 5))
        if width < output_count or output_count < 1:
            raise ValueError("beam_width must cover the positive bundle_list_size")
        candidates = [([], 0.)]
        for _ in range(k):
            expanded = {}
            for prefix, quality in candidates:
                scores = self.model.logits(context, prefix)[-1, :len(self.catalog)].clone()
                scores[prefix] = -torch.inf
                log_prob = scores.log_softmax(-1)
                for item in torch.topk(log_prob, min(width, len(self.catalog) - len(prefix))).indices.tolist():
                    sequence, score = prefix + [item], quality + float(log_prob[item])
                    key = tuple(sorted(sequence))
                    if key not in expanded or score > expanded[key][1]:
                        expanded[key] = (sequence, score)
            top = sorted(expanded.values(), key=lambda x: (-x[1], x[0]))[:width]
            selected = dpp_select([x[0] for x in top], [x[1] for x in top], output_count, float(self.config.get("diversity", .1)))
            candidates = [top[i] for i in selected]
        return candidates

    def predict(self, context, k, seed):
        candidates = self.predict_list(context, k, seed)
        return self.result(candidates[0][0], bundle_list=[[self.item_ids[i] for i in bundle] for bundle, _ in candidates],
                           list_log_quality=[score for _, score in candidates], decoding="masked_beam_dpp_fixed_k")
