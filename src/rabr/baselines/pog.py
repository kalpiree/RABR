from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from .base import Base, category, inputs, sampled_ce
from .features import MissingMethodAsset, historical_ids, load_pog_features


def transformer_encoder(dim, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


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
