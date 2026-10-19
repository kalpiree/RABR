"""BYOB released models with item-to-bundle policy curriculum."""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .traditional_models import Base, inputs
from .traditional_assets import MissingMethodAsset, historical_ids, read_manifest, verified_file
from .traditional_upstream import load_byob


def ppo_loss(log_probability, old_log_probability, advantage, clip=.2):
    ratio = (log_probability - old_log_probability).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def discounted_returns(rewards, gamma=1.):
    total, output = 0., []
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
        output.append(total)
    return list(reversed(output))


def bundle_metric(prediction, gold, metric, *, allow_variable_targets=False):
    if not gold or (len(prediction) != len(gold) and not allow_variable_targets):
        raise ValueError("BYOB native bundle metrics require equal target and generated sizes")
    common = set(prediction) & set(gold)
    if metric == "recall":
        return len(common) / len(gold)
    if metric == "precision":
        return float(prediction[0] in common)
    if metric == "precision_plus":
        return float(bool(common))
    raise ValueError("Unknown BYOB bundle metric")


class BYOB(Base):
    name = "byob"

    def build_features(self, catalog):
        return None

    def load_assets(self, train, catalog):
        path, manifest = read_manifest(self.config, "byob_assets_path")
        train_ids = {row["session_id"] for row in train}
        if set(manifest.get("training_session_ids", [])) != train_ids:
            raise ValueError("BYOB asset training session identities differ from fit inputs")
        self.asset_hashes = {"manifest": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assets = {}
        for key in ("item_interactions", "click_sequences", "candidate_pools"):
            entry = manifest.get(key, {})
            if not entry.get("provenance"):
                raise MissingMethodAsset("BYOB requires documented " + key)
            if key == "candidate_pools":
                if entry.get("target_independent") is not True:
                    raise ValueError("Prediction candidate pools must be built independently of held-out targets")
            elif entry.get("training_scope") != "training_only":
                raise ValueError("BYOB reward assets must be training_only")
            file = verified_file(path, entry)
            self.assets[key] = json.loads(file.read_text())
            self.asset_hashes[key] = entry["sha256"]
        for key in ("item_interactions", "click_sequences"):
            if not self.assets[key] or any(row.get("session_id") not in train_ids for row in self.assets[key]):
                raise ValueError("Reward training assets contain missing or non-training session identities")
        self.candidate_pools = self.assets["candidate_pools"]
        if not isinstance(self.candidate_pools, dict):
            raise ValueError("candidate_pools must map session IDs to item-ID lists")

    def fit(self, train, validation, catalog):
        self.load_assets(train, catalog)
        definitions, source = load_byob(self.config)
        self.asset_hashes.update(source)
        self.setup(train, validation, catalog)
        self.users = {user: index + 1 for index, user in enumerate(sorted({str(row["user_id"]) for row in train if row.get("user_id") is not None}))}
        if not self.users:
            raise MissingMethodAsset("BYOB requires persistent user_id values, not session or intent embeddings")
        self.bundle_size = int(self.config.get("training_bundle_size", 3))
        self.pool_size = int(self.config.get("pool_size", 20))
        self.training_rows = train
        self.evaluation_protocol = self.config.get("evaluation_protocol", "native")
        if self.evaluation_protocol not in {"native", "rabr_fixed_k"}:
            raise ValueError("Unknown BYOB evaluation protocol")
        self.exploration_rate = float(self.config.get("exploration_rate", 0.))
        if not 0 <= self.exploration_rate < 1:
            raise ValueError("BYOB exploration_rate must lie in [0,1)")
        if self.evaluation_protocol == "native" and self.exploration_rate:
            raise ValueError("Exploration-rate mixing requires the explicit rabr_fixed_k adapter")
        if not 1 <= self.bundle_size <= self.pool_size <= len(catalog):
            raise ValueError("BYOB requires 1 <= bundle size <= pool size <= catalog size")
        for row in list(train) + list(validation):
            self.observation(self.row_context(row), [], self.training_pool(row, random.Random(self.seed)) if row in train else self.pool(self.row_context(row)))
            if self.evaluation_protocol == "native" and len(self.ids(row)) != self.bundle_size:
                raise MissingMethodAsset("BYOB's native profile requires fixed-size training/validation bundles")
        conf = {"n_user": len(self.users) + 1, "n_item": len(catalog) + 1, "vocab_size": len(catalog) + 1,
                "bundle_size": self.bundle_size, "embed_dim": int(self.config.get("embed_dim", 32)),
                "hidden_dim": int(self.config.get("pointer_hidden_dim", 128)), "num_heads": 2, "num_layers": 1,
                "dropout": float(self.config.get("dropout", .1)), "encoder": bool(self.config.get("pool_transformer", False)),
                "concat": bool(self.config.get("concat_bundle", False)), "clip": 10}
        preference = definitions["ItemNCFModel"]({**conf, "hidden_dim": 128})
        compatibility = definitions["SkipGramModel"]({**conf, "hidden_dim": 64})
        self.model = nn.ModuleDict({"policy": definitions["ItemPointer"](conf), "value": nn.Linear(3 * conf["embed_dim"], 1),
                                    "preference": preference, "compatibility": compatibility}).to(self.device)
        if self.restore_fit():
            return self
        self.train_rewards()
        self.model["preference"].requires_grad_(False)
        self.model["compatibility"].requires_grad_(False)
        self.train_policy(train, validation, "item", int(self.config.get("item_policy_epochs", 50)))
        self.train_policy(train, validation, "bundle", int(self.config.get("bundle_policy_epochs", 50)))
        return self.finish()

    def user(self, context):
        user = context.get("user_id")
        if user is None or str(user) not in self.users:
            raise MissingMethodAsset("BYOB needs a persistent user present in the reward/policy training data")
        return self.users[str(user)]

    def metric(self, prediction, gold):
        return bundle_metric(prediction, gold, self.config.get("bundle_metric", "recall"),
                             allow_variable_targets=self.evaluation_protocol == "rabr_fixed_k")

    def pool(self, context):
        items = self.candidate_pools.get(context.get("session_id"))
        if not isinstance(items, list) or len(items) != self.pool_size or len(set(items)) != len(items) or not set(items) <= self.index.keys():
            raise MissingMethodAsset("BYOB requires a distinct, target-independent candidate pool of declared pool_size for this session")
        return [self.index[item] + 1 for item in items]

    def training_pool(self, row, rng):
        positive = [item + 1 for item in self.ids(row)]
        if len(positive) > self.pool_size:
            raise MissingMethodAsset("BYOB training target exceeds the declared candidate pool size")
        excluded = {item for example in getattr(self, "training_rows", [row]) if example.get("user_id") == row.get("user_id") for item in self.ids(example)}
        negative = [item + 1 for item in range(len(self.catalog)) if item not in excluded]
        if len(negative) < self.pool_size - len(positive):
            raise MissingMethodAsset("BYOB training needs enough unpurchased candidate negatives")
        pool = positive + rng.sample(negative, self.pool_size - len(positive))
        rng.shuffle(pool)
        return pool

    def observation(self, context, prefix, pool):
        inputs(context)
        length = int(self.config.get("sequence_length", 20))
        history = [index + 1 for index in historical_ids(context, self.index, maximum=length, minimum=length)]
        return {"user": self.user(context), "history": history, "pool": [item if item not in prefix else 0 for item in pool],
                "bundle": list(prefix) + [0] * (self.bundle_size - len(prefix))}

    def compatibility(self, actions, previous):
        model = self.model["compatibility"]
        left = model.inp_embed(torch.as_tensor(actions, device=self.device))
        if model.hidden_dim:
            left = F.relu(model.fc1(left))
            left = F.relu(model.fc2(left))
        right = model.out_embed(torch.as_tensor(previous, device=self.device))
        return (left * right).sum(-1).sigmoid()

    def policy(self, observation):
        tensors = [torch.tensor([[observation["user"]]], device=self.device),
                   torch.tensor([observation["history"]], device=self.device),
                   torch.tensor([observation["pool"]], device=self.device),
                   torch.tensor([observation["bundle"]], device=self.device)]
        for index in (1, 2, 3):
            if tensors[index].shape[1] == 1:
                tensors[index] = tensors[index].unsqueeze(1)
        logits, state = self.model["policy"](tensors)
        allowed = torch.tensor(observation["pool"], device=self.device) > 0
        previous = [item for item in observation["bundle"] if item]
        threshold = float(self.config.get("neighbor_threshold", 0))
        if previous and threshold:
            with torch.no_grad():
                allowed &= self.compatibility(observation["pool"], [previous[-1]] * len(observation["pool"])) >= threshold
        if not allowed.any():
            raise ValueError("BYOB neighbor constraint leaves no feasible action")
        probability = logits[0].masked_fill(~allowed, -torch.inf).softmax(-1)
        rate = self.exploration_rate
        probability = (1 - rate) * probability + rate * allowed.to(probability.dtype) / allowed.sum()
        return probability, self.model["value"](state)[0, 0]

    def train_rewards(self):
        records = self.assets["item_interactions"]
        training_users = {row["session_id"]: str(row["user_id"]) for row in self.training_rows}
        for row in records:
            if str(row.get("user_id")) not in self.users or row.get("item_id") not in self.index or row.get("label") not in (0, 1):
                raise ValueError("Invalid BYOB item interaction")
            if str(row["user_id"]) != training_users[row["session_id"]]:
                raise ValueError("Item interaction user differs from its training session")
        if {row["label"] for row in records} != {0, 1}:
            raise MissingMethodAsset("BYOB preference fitting requires positive and negative training feedback")
        preference = self.model["preference"]
        optimizer = torch.optim.Adam(preference.parameters(), lr=float(self.config.get("reward_learning_rate", .001)))
        size = int(self.config.get("reward_batch_size", 256))
        if size < 2 or int(self.config.get("reward_epochs", 10)) < 1:
            raise ValueError("BYOB reward fitting requires batch size >= 2 and positive epochs")
        for epoch in range(int(self.config.get("reward_epochs", 10))):
            self.rng.shuffle(records)
            losses = []
            for start in range(0, len(records), size):
                batch = records[start:start + size]
                user = torch.tensor([[self.users[str(row["user_id"])]] for row in batch], device=self.device)
                item = torch.tensor([[self.index[row["item_id"]] + 1] for row in batch], device=self.device)
                label = torch.tensor([[float(row["label"])] for row in batch], device=self.device)
                loss = F.binary_cross_entropy(preference((user, item, None)), label)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            self.training_history.append({"stage": "released_ItemNCF_BCE", "epoch": epoch + 1, "train_loss": sum(losses) / len(losses)})
        sequences = []
        for row in self.assets["click_sequences"]:
            if not set(row.get("items", [])) <= self.index.keys():
                raise ValueError("Unknown item in BYOB click sequence")
            sequences.append([self.index[item] + 1 for item in row["items"]])
        counts = collections.Counter(item for sequence in sequences for item in sequence)
        population, probabilities = list(counts), [counts[item] ** .75 for item in counts]
        window = int(self.config.get("context_window", 5))
        if window < 1:
            raise ValueError("BYOB SGNS context window must be positive")
        pairs = [(sequence[i], sequence[j]) for sequence in sequences for i in range(window, len(sequence) - window)
                 for j in range(max(i - window, 0), min(i + window, len(sequence))) if j != i]
        if len(pairs) < 2:
            raise MissingMethodAsset("BYOB SGNS requires actual training click sequences longer than twice its context window")
        model = self.model["compatibility"]
        optimizer = torch.optim.SGD(model.parameters(), lr=float(self.config.get("compatibility_learning_rate", .001)))
        negatives = int(self.config.get("compatibility_negatives", 15))
        if negatives < 2:
            raise ValueError("The released SkipGram forward requires at least two negative samples")
        for epoch in range(int(self.config.get("reward_epochs", 10))):
            self.rng.shuffle(pairs)
            losses = []
            batches = [pairs[start:start + size] for start in range(0, len(pairs), size)]
            if len(batches[-1]) == 1:
                batches[-2].extend(batches.pop())
            for batch in batches:
                left = torch.tensor([[item] for item, _ in batch], device=self.device)
                right = torch.tensor([[item] for _, item in batch], device=self.device)
                negative = torch.tensor([self.rng.choices(population, weights=probabilities, k=negatives) for _ in batch], device=self.device)
                loss = model((left, right, negative))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            self.training_history.append({"stage": "released_SkipGram_SGNS", "epoch": epoch + 1, "train_loss": sum(losses) / len(losses)})

    @torch.no_grad()
    def reward_terms(self, context, prefix, action, gold, final):
        user = torch.tensor([[self.user(context)]], device=self.device)
        item = torch.tensor([[action]], device=self.device)
        preference = float(self.model["preference"]((user, item, None))[0, 0])
        compatibility = float(self.compatibility([action] * len(prefix), prefix).sum()) if prefix else 0.
        terminal = self.metric(prefix + [action], gold) if final else 0.
        return {"item_preference": preference, "item_compatibility": compatibility, "bundle_metric": terminal}

    @torch.no_grad()
    def rollout(self, row, rng, stage):
        context = self.row_context(row)
        pool = self.training_pool(row, rng)
        gold = [index + 1 for index in self.ids(row)]
        prefix, records, rewards = [], [], []
        for step in range(self.bundle_size):
            observation = self.observation(context, prefix, pool)
            probabilities, value = self.policy(observation)
            choice = rng.choices(range(len(pool)), weights=probabilities.cpu().tolist())[0]
            action = pool[choice]
            terms = self.reward_terms(context, prefix, action, gold, step + 1 == self.bundle_size)
            rewards.append(terms["item_preference"] + terms["item_compatibility"] if stage == "item" else terms["bundle_metric"])
            records.append({"observation": observation, "action": choice, "old_log_probability": float(probabilities[choice].log()), "old_value": float(value), "terms": terms})
            prefix.append(action)
        returns = discounted_returns(rewards, 1.)
        for record, value in zip(records, returns):
            record["return"] = value
            record["advantage"] = value - record["old_value"]
        return records

    def train_policy(self, train, validation, stage, epochs):
        if epochs < 1:
            raise ValueError("Both BYOB curriculum stages require positive epochs")
        self.training_rows = train
        if int(self.config.get("ppo_updates", 4)) < 1 or int(self.config.get("rl_batch_size", 128)) < 1:
            raise ValueError("BYOB PPO budgets must be positive")
        parameters = list(self.model["policy"].parameters()) + list(self.model["value"].parameters())
        optimizer = torch.optim.Adam(parameters, lr=float(self.config.get("rl_learning_rate", .0003)))
        best, stale, selected = -math.inf, 0, None
        self.model.eval()
        for epoch in range(epochs):
            losses = []
            for start in range(0, len(train), int(self.config.get("batch_size", 16))):
                records = [record for row in train[start:start + int(self.config.get("batch_size", 16))] for record in self.rollout(row, self.rng, stage)]
                for _ in range(int(self.config.get("ppo_updates", 4))):
                    self.rng.shuffle(records)
                    size = int(self.config.get("rl_batch_size", 128))
                    for offset in range(0, len(records), size):
                        objectives = []
                        for record in records[offset:offset + size]:
                            probability, value = self.policy(record["observation"])
                            policy_loss = ppo_loss(probability[record["action"]].log(), value.new_tensor(record["old_log_probability"]), value.new_tensor(record["advantage"]), float(self.config.get("ppo_clip", .2)))
                            objectives.append(policy_loss + .5 * (value - record["return"]).square())
                        loss = torch.stack(objectives).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite BYOB policy objective")
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(parameters, 1.)
                        optimizer.step()
                        losses.append(float(loss.detach()))
            with torch.no_grad():
                scores = [self.metric(self.greedy(self.row_context(row)), [i + 1 for i in self.ids(row)]) for row in validation]
            score = sum(scores) / len(scores)
            self.training_history.append({"stage": stage + "_reward_PPO", "epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "validation_bundle_metric": score})
            if score > best:
                best, stale, selected = score, 0, copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= int(self.config.get("patience", 10)):
                    break
        self.model.load_state_dict(selected)

    @torch.no_grad()
    def greedy(self, context):
        pool = self.pool(context)
        prefix = []
        for _ in range(self.bundle_size):
            probability, _ = self.policy(self.observation(context, prefix, pool))
            prefix.append(pool[int(probability.argmax())])
        return prefix

    def predict(self, context, k, seed):
        self.validate_predict(context, k)
        if k != self.bundle_size:
            raise ValueError("BYOB was trained for a fixed bundle_size; inference k must match")
        return self.result([item - 1 for item in self.greedy(context)], decoding="released_ItemPointer", curriculum=["item_preference_plus_compatibility", "terminal_" + self.config.get("bundle_metric", "recall")], gamma=1., candidate_pool_size=self.pool_size, intent_used=False,
                           evaluation_protocol=self.evaluation_protocol,
                           exploration_rate=self.exploration_rate,
                           evaluation_adaptation="fixed_action_horizon_variable_target_recall" if self.evaluation_protocol == "rabr_fixed_k" else None)
