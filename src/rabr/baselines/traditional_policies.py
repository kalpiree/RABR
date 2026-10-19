"""Bundle BERT conversation policies and BYOB sequential composition policies."""
from __future__ import annotations
import collections
import copy
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .traditional_models import Base, Features, category, f1, inputs, targets, transformer_encoder, words


def ppo_loss(log_probability, old_log_probability, advantage, clip=.2):
    ratio = (log_probability - old_log_probability).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()


def discounted_returns(rewards, gamma):
    total, output = 0., []
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
        output.append(total)
    return list(reversed(output))


def update_minibatches(model, records, optimizer, objective, config):
    size, updates = int(config.get("rl_batch_size", 128)), int(config.get("ppo_updates", 2))
    if size < 1 or updates < 1:
        raise ValueError("rl_batch_size and ppo_updates must be positive")
    losses = []
    for _ in range(updates):
        for start in range(0, len(records), size):
            loss = torch.stack([objective(record) for record in records[start:start+size]]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO objective")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def head(dim, out):
    return nn.Sequential(nn.Linear(dim, 2*dim), nn.ReLU(), nn.Linear(2*dim, out))


class BUNTNetwork(nn.Module):
    def __init__(self, features, catalog, config):
        super().__init__()
        self.features = features
        dim, heads = features.dim, int(config.get("heads", 4))
        layers, dropout = int(config.get("layers", 2)), float(config.get("dropout", .1))
        self.low = transformer_encoder(dim, heads, 1, dropout)
        self.high = transformer_encoder(dim, heads, layers, dropout)
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(dim, heads, 4*dim, dropout, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        frequencies = collections.Counter(token for row in catalog for token in set(words(row.get("title", ""))) if len(token) > 2)
        self.attributes = [token for token, _ in sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))[:int(config.get("max_attributes", 128))]] or ["unknown"]
        self.attribute_ids = {x: i for i, x in enumerate(self.attributes)}
        target_attributes = torch.zeros(len(catalog), len(self.attributes))
        for i, row in enumerate(catalog):
            for token in words(row.get("title", "")):
                if token in self.attribute_ids:
                    target_attributes[i, self.attribute_ids[token]] = 1
        self.register_buffer("attribute_targets", target_attributes)
        self.register_buffer("attribute_weights", len(catalog) / (len(self.attributes) * target_attributes.sum(0).clamp_min(1)))
        self.category_state = nn.Embedding(len(features.categories), dim)
        self.attribute_state = nn.Embedding(len(self.attributes), dim)
        self.results = nn.Embedding(6, dim)
        self.actors = nn.ModuleDict({"item": head(dim, features.n), "category": head(dim, len(features.categories)),
                                    "attribute": head(dim, len(self.attributes))})
        self.manager_context = head(dim, 2)
        self.manager_history = head(dim, 2)
        self.manager_gate = nn.Linear(2*dim, 1)
        self.values = nn.ModuleDict({name: nn.Linear(dim if name != "manager" else 2*dim, 1)
                                     for name in ("item", "category", "attribute", "manager")})

    def memory(self, context):
        history = inputs(context).get("history", [])
        groups = collections.defaultdict(list)
        for position, row in enumerate(history):
            item = row.get("item_id") if isinstance(row, dict) else row
            if item in self.features.ids:
                key = str(row.get("bundle_id", position)) if isinstance(row, dict) else str(position)
                groups[key].append(self.features.ids[item])
        representations = [self.features.context(context)[:, :1]]
        for group in list(groups.values())[-100:]:
            values = self.features.embed(torch.tensor(group, device=self.features.text.device))[None]
            representations.append(self.low(values).mean(1, keepdim=True))
        return self.high(torch.cat(representations, 1))

    def state(self, context, slots, categories, attributes):
        memory = self.memory(context)
        ids = torch.tensor(slots, device=memory.device)
        values = self.features.embed(ids)[None]
        extra = torch.zeros_like(values)
        for i in range(len(slots)):
            if categories[i]:
                extra[0, i] += self.category_state(torch.tensor(categories[i], device=memory.device)).mean(0)
            if attributes[i]:
                extra[0, i] += self.attribute_state(torch.tensor(attributes[i], device=memory.device)).mean(0)
        for layer, norm in zip(self.decoder, self.norms):
            values = layer(norm(values + extra), memory)
        return values[0]

    def manager_features(self, state, history):
        codes = torch.tensor(history or [0], device=state.device)
        return torch.cat((state.mean(0), self.results(codes).mean(0)), -1)

    def policy(self, name, features):
        if name != "manager":
            return self.actors[name](features)
        context, history = features.chunk(2, -1)
        gate = self.manager_gate(features).sigmoid()
        probability = (1-gate) * self.manager_context(context).softmax(-1) + gate * self.manager_history(history).softmax(-1)
        return probability.clamp_min(1e-12).log()


class BUNT(Base):
    name = "bunt"

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        self.max_rounds = int(self.config.get("max_rounds", 20))
        if not 1 <= self.max_rounds <= 20:
            raise ValueError("BUNT max_rounds must be between one and twenty")
        self.model = BUNTNetwork(self.features, catalog, self.config).to(self.device)
        if self.restore_fit():
            return self
        self.train_stage(train, validation, self.objective, int(self.config.get("epochs", 100)), "bundle_cloze_pretraining")
        self.online_fit(train, validation)
        return self.finish()

    def objective(self, row, rng):
        gold = self.ids(row)
        mask = self.features.n + 1
        slots = list(gold)
        chosen = rng.sample(range(len(gold)), rng.randint(1, len(gold)))
        categories, attributes = [[] for _ in gold], [[] for _ in gold]
        for i, item in enumerate(gold):
            if rng.random() >= float(self.config.get("attribute_mask_probability", .5)):
                categories[i] = [int(self.features.category_ids[item])]
                attributes[i] = self.model.attribute_targets[item].nonzero().flatten().tolist()
        for i in chosen:
            slots[i] = mask
        state = self.model.state(self.row_context(row), slots, categories, attributes)[chosen]
        item_logits = self.model.policy("item", state)
        category_logits = self.model.policy("category", state)
        attr_logits = self.model.policy("attribute", state)
        target = torch.tensor([gold[i] for i in chosen], device=self.device)
        item_loss = F.cross_entropy(item_logits, target)
        category_loss = F.cross_entropy(category_logits, self.features.category_ids[target])
        attr_loss = -(self.model.attribute_targets[target] * self.model.attribute_weights * attr_logits.log_softmax(-1)).sum(-1).mean()
        item_hit = item_logits.detach().argmax(-1) == target
        category_hit = category_logits.detach().argmax(-1) == self.features.category_ids[target]
        valid = item_hit | category_hit
        manager_loss = state.sum() * 0
        if valid.any():
            manager_loss = F.cross_entropy(self.model.manager_context(state[valid]), item_hit[valid].long())
        return item_loss + float(self.config.get("ask_loss_weight", 1.)) * (category_loss + attr_loss + manager_loss)

    def policy_record(self, name, feature, allowed, rng):
        scores = self.model.policy(name, feature)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[allowed] = True
        probability = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        action = rng.choices(range(len(scores)), weights=probability.detach().cpu().tolist())[0]
        return {"name": name, "feature": feature.detach(), "allowed": allowed, "action": action,
                "old_log_probability": probability[action].detach().log(),
                "old_value": float(self.model.values[name](feature).detach().squeeze()), "reward": 0.}

    def train_episode(self, row, rng):
        gold = self.ids(row)
        k = min(int(self.config.get("training_slots", 3)), len(self.catalog))
        slots = [self.features.n + 1] * k
        categories, attributes = [[] for _ in slots], [[] for _ in slots]
        item_pool = [set(range(len(self.catalog))) for _ in slots]
        category_pool = [set(range(len(self.features.categories))) for _ in slots]
        records, managers, accepted, history = [], [], set(), []
        with torch.no_grad():
            for turn in range(self.max_rounds):
                state = self.model.state(self.row_context(row), slots, categories, attributes)
                manager = self.policy_record("manager", self.model.manager_features(state, history), [0, 1], rng)
                managers.append(manager)
                successes = 0
                for position in range(k):
                    if slots[position] < self.features.n:
                        continue
                    name = "item" if manager["action"] else "category"
                    pool = item_pool[position] if name == "item" else category_pool[position]
                    allowed = sorted(pool - accepted) if name == "item" else sorted(pool)
                    if not allowed:
                        continue
                    record = self.policy_record(name, state[position], allowed, rng)
                    action = record["action"]
                    if name == "item":
                        success = action in gold
                        for pool_ in item_pool:
                            pool_.discard(action)
                        if success:
                            accepted.add(action)
                            slots[position] = action
                    else:
                        success = any(int(self.features.category_ids[item]) == action for item in gold if item not in accepted)
                        category_pool[position].discard(action)
                        if success:
                            categories[position].append(action)
                    record["reward"] = float(success)
                    records.append(record)
                    successes += int(success)
                history.append((1 if successes else 2) if manager["action"] else (3 if successes else 4))
                if len(accepted) >= min(k, len(gold)):
                    break
        terminal = f1(list(accepted), gold)
        for record, reward in zip(managers, discounted_returns([0.] * (len(managers)-1) + [terminal], float(self.config.get("gamma", .99)))):
            record["reward"] = reward
        return records + managers, terminal

    def online_fit(self, train, validation):
        epochs = int(self.config.get("online_epochs", 10))
        if epochs < 1:
            return
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (self.model.actors, self.model.manager_context, self.model.manager_history, self.model.manager_gate, self.model.values):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=float(self.config.get("rl_learning_rate", 3e-4)))
        best, selected, stale = -math.inf, None, 0
        for epoch in range(epochs):
            self.model.eval()
            rewards, losses = [], []
            rollout_batch = int(self.config.get("batch_size", 16))
            for start in range(0, len(train), rollout_batch):
                records = []
                for row in train[start:start+rollout_batch]:
                    episode, reward = self.train_episode(row, self.rng)
                    records.extend(episode)
                    rewards.append(reward)
                losses.extend(self.update_records(records, optimizer))
            check = []
            for i, row in enumerate(validation):
                _, reward = self.train_episode(row, random.Random(self.seed + i))
                check.append(reward)
            score = sum(check or rewards)/len(check or rewards)
            self.training_history.append({"stage": "conversation_ppo", "epoch": epoch+1, "train_loss": sum(losses)/len(losses), "validation_reward": score if check else None})
            if score > best:
                best, selected, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if validation and stale >= int(self.config.get("patience", 10)):
                    break
        self.model.load_state_dict(selected)

    def update_records(self, records, optimizer):
        def objective(record):
            feature, name = record["feature"], record["name"]
            scores = self.model.policy(name, feature)
            invalid = torch.ones_like(scores, dtype=torch.bool)
            invalid[record["allowed"]] = False
            log_probability = scores.masked_fill(invalid, -torch.inf).log_softmax(-1)[record["action"]]
            value = self.model.values[name](feature).squeeze(-1)
            advantage = value.new_tensor(record["reward"] - record["old_value"])
            return ppo_loss(log_probability, record["old_log_probability"], advantage, float(self.config.get("ppo_clip", .2))) + .5 * (value - record["reward"]).square()
        return update_minibatches(self.model, records, optimizer, objective, self.config)

    @torch.no_grad()
    def fill(self, context, slots, categories, attributes, excluded=None):
        selected = [item for item in slots if item < self.features.n]
        output = list(slots)
        for position, item in enumerate(slots):
            if item < self.features.n:
                continue
            state = self.model.state(context, output, categories, attributes)
            scores = self.model.policy("item", state[position]).clone()
            blocked = set(selected) | set(excluded or [])
            if len(blocked) >= len(self.catalog):
                break
            scores[list(blocked)] = -torch.inf
            output[position] = int(scores.argmax())
            selected.append(output[position])
        return [item for item in output if item < self.features.n]

    @torch.no_grad()
    def predict(self, context, k, seed):
        self.validate_predict(context, k)
        data = inputs(context)
        provider = context.get("feedback_provider", data.get("feedback_provider"))
        feedback = context.get("feedback", data.get("feedback", []))
        slots = [self.features.n + 1] * k
        categories, attributes = [[] for _ in slots], [[] for _ in slots]
        history, transcript, rejected, asked = [], [], set(), set()
        if provider is None and not feedback:
            return self.result(self.fill(context, slots, categories, attributes), conversation_turns=0,
                               feedback_mode="one_shot_no_respondent", max_rounds=self.max_rounds)
        for turn in range(self.max_rounds):
            state = self.model.state(context, slots, categories, attributes)
            action = int(self.model.policy("manager", self.model.manager_features(state, history)).argmax())
            open_positions = [i for i, item in enumerate(slots) if item >= self.features.n]
            if not open_positions:
                break
            if action == 1:
                proposal = self.fill(context, slots, categories, attributes, rejected)
                proposed_new = [i for i in proposal if i not in slots]
                request = {"turn": turn+1, "action": "recommend", "items": [self.item_ids[i] for i in proposed_new]}
            else:
                logits = self.model.policy("category", state[open_positions].mean(0)).clone()
                if len(asked) >= len(self.features.categories):
                    break
                logits[list(asked)] = -torch.inf
                choice = int(logits.argmax())
                asked.add(choice)
                request = {"turn": turn+1, "action": "ask", "category": self.features.categories[choice]}
            response = provider(request) if provider is not None else (feedback[turn] if turn < len(feedback) else None)
            if response is None:
                break
            if not isinstance(response, dict):
                raise ValueError("Feedback must be a dictionary or None")
            if action == 1:
                accepted_ids = response.get("accepted_items", [])
                rejected_ids = response.get("rejected_items", [])
                if not set(accepted_ids + rejected_ids) <= set(request["items"]):
                    raise ValueError("Item feedback must refer only to proposed items")
                if len(set(accepted_ids)) != len(accepted_ids):
                    raise ValueError("Accepted item feedback cannot contain duplicates")
                if set(accepted_ids) & set(rejected_ids):
                    raise ValueError("Feedback cannot accept and reject the same item")
                for position, item_id in zip(open_positions, accepted_ids):
                    slots[position] = self.index[item_id]
                rejected.update(self.index[i] for i in rejected_ids)
                success = bool(accepted_ids)
                history.append(1 if success else 2)
            else:
                success = bool(response.get("accepted", False))
                if success:
                    for position in open_positions:
                        categories[position].append(choice)
                elif response.get("rejected", False):
                    rejected.update(i for i, cat in enumerate(self.features.category_ids.tolist()) if cat == choice)
                history.append(3 if success else 4)
            transcript.append({"request": request, "response": response})
        selected = self.fill(context, slots, categories, attributes, rejected)
        return self.result(selected, conversation_turns=len(transcript), feedback_mode="external", transcript=transcript, max_rounds=self.max_rounds)


class BYOBNetwork(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.features = features
        dim = features.dim
        self.preference_query = nn.Linear(dim, dim)
        self.compatibility = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(3*dim, dim)
        self.reference = nn.Linear(dim, dim)
        self.pointer = nn.Linear(dim, 1, bias=False)
        self.value = nn.Linear(3*dim, 1)
        self.clip = float(config.get("pointer_clip", 10.))

    def state(self, context, prefix):
        memory = self.features.context(context)[0]
        query = memory[0]
        history = memory[1:].mean(0) if len(memory) > 1 else torch.zeros_like(query)
        bundle = self.features.all_items()[prefix].mean(0) if prefix else torch.zeros_like(query)
        return torch.cat((query, history, bundle), -1)

    def logits(self, state):
        logits = self.pointer(torch.tanh(self.query(state)[None] + self.reference(self.features.all_items()))).squeeze(-1)
        return self.clip * torch.tanh(logits)

    def preference(self, context):
        user = self.preference_query(self.features.context(context).mean(1)[0])
        return self.features.all_items() @ user / math.sqrt(self.features.dim)

    def compatibility_scores(self, item, previous):
        values = self.features.all_items()
        return (self.compatibility(values[item])[None] * values[previous]).sum(-1) / math.sqrt(self.features.dim)


class BYOB(Base):
    name = "byob"

    def fit(self, train, validation, catalog):
        self.setup(train, validation, catalog)
        self.epsilon = float(self.config.get("epsilon", .2))
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be in [0,1)")
        self.model = BYOBNetwork(self.features, self.config).to(self.device)
        if self.restore_fit():
            return self
        self.train_stage(train, validation, self.reward_objective, int(self.config.get("reward_epochs", 10)), "item_preference_compatibility")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (self.model.query, self.model.reference, self.model.pointer, self.model.value):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.train_policy(train, validation)
        return self.finish()

    def reward_objective(self, row, rng):
        gold = self.ids(row)
        negative = [i for i in range(len(self.catalog)) if i not in gold]
        positive = rng.choice(gold)
        candidates = [positive] + ([rng.choice(negative)] if negative else [])
        labels = torch.tensor([1.] + ([0.] if negative else []), device=self.device)
        preference = F.binary_cross_entropy_with_logits(self.model.preference(self.row_context(row))[candidates], labels)
        compatible = [i for i in gold if i != positive]
        if not compatible:
            return preference
        previous = [rng.choice(compatible)] + ([rng.choice(negative)] if negative else [])
        return preference + F.binary_cross_entropy_with_logits(self.model.compatibility_scores(positive, previous), labels)

    @torch.no_grad()
    def reward_terms(self, context, prefix, action, gold, final):
        preference = float(self.model.preference(context)[action].sigmoid())
        compatibility = float(self.model.compatibility_scores(action, prefix).sigmoid().sum()) if prefix else 0.
        terminal = f1(prefix + [action], gold) if final else 0.
        return {"item_preference": preference, "item_compatibility": compatibility, "bundle_metric": terminal}

    def probabilities(self, state, prefix):
        logits = self.model.logits(state)
        allowed = torch.ones(len(self.catalog), device=self.device, dtype=torch.bool)
        allowed[prefix] = False
        probability = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
        return (1-self.epsilon) * probability + self.epsilon * allowed.float() / allowed.sum()

    @torch.no_grad()
    def rollout(self, row, rng):
        gold = self.ids(row)
        k = min(int(self.config.get("training_bundle_size", 3)), len(self.catalog))
        prefix, records, rewards = [], [], []
        context = self.row_context(row)
        weights = self.config.get("reward_weights", [1., 1., 1.])
        if len(weights) != 3:
            raise ValueError("BYOB requires three reward weights")
        for step in range(k):
            state = self.model.state(context, prefix)
            probability = self.probabilities(state, prefix)
            action = rng.choices(range(len(self.catalog)), weights=probability.cpu().tolist())[0]
            terms = self.reward_terms(context, prefix, action, gold, step == k-1)
            reward = sum(float(w) * value for w, value in zip(weights, terms.values()))
            records.append({"state": state.detach(), "prefix": list(prefix), "action": action,
                            "old_log_probability": probability[action].log().detach(), "reward_terms": terms,
                            "old_value": float(self.model.value(state).squeeze())})
            rewards.append(reward)
            prefix.append(action)
        returns = discounted_returns(rewards, float(self.config.get("gamma", .99)))
        for record, value in zip(records, returns):
            record["return"] = value
        return records, f1(prefix, gold)

    def train_policy(self, train, validation):
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=float(self.config.get("rl_learning_rate", 3e-4)))
        best, selected, stale = -math.inf, None, 0
        self.model.eval()
        for epoch in range(int(self.config.get("epochs", 100))):
            rewards, losses = [], []
            rollout_batch = int(self.config.get("batch_size", 16))
            for start in range(0, len(train), rollout_batch):
                records = []
                for row in train[start:start+rollout_batch]:
                    episode, reward = self.rollout(row, self.rng)
                    records.extend(episode)
                    rewards.append(reward)
                losses.extend(self.update_records(records, optimizer))
            with torch.no_grad():
                check = [f1(self.greedy(self.row_context(row), min(int(self.config.get("training_bundle_size", 3)), len(self.catalog))), self.ids(row)) for row in validation]
            score = sum(check or rewards)/len(check or rewards)
            self.training_history.append({"stage": "composition_ppo", "epoch": epoch+1, "train_loss": sum(losses)/len(losses), "validation_f1": score if check else None})
            if score > best:
                best, selected, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if validation and stale >= int(self.config.get("patience", 10)):
                    break
        self.model.load_state_dict(selected)

    def update_records(self, records, optimizer):
        def objective(record):
            state = record["state"]
            probability = self.probabilities(state, record["prefix"])
            value = self.model.value(state).squeeze(-1)
            advantage = value.new_tensor(record["return"] - record["old_value"])
            return ppo_loss(probability[record["action"]].log(), record["old_log_probability"], advantage, float(self.config.get("ppo_clip", .2))) + .5 * (value - record["return"]).square()
        return update_minibatches(self.model, records, optimizer, objective, self.config)

    @torch.no_grad()
    def greedy(self, context, k):
        selected = []
        for _ in range(k):
            scores = self.model.logits(self.model.state(context, selected)).clone()
            scores[selected] = -torch.inf
            selected.append(int(scores.argmax()))
        return selected

    def predict(self, context, k, seed):
        self.validate_predict(context, k)
        return self.result(self.greedy(context, k), decoding="sequential_pointer_greedy", reward_variant="B-M",
                           reward_terms=["item_preference", "item_compatibility", "terminal_bundle_f1"], exploration_epsilon_training=self.epsilon)
