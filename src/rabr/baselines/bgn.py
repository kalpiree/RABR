from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

from .base import Base, category, inputs, sampled_ce
from .features import MissingMethodAsset, historical_ids


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
