"""Bundle BERT with hierarchical policies and multiround feedback."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from rabr.io import atomic_json, digest, file_hash
from rabr.types import Prediction, gold_items
from .bunt_feedback import TargetFeedback


def _context(row):
    source = row.get("input", row)
    return {"session_id": str(row["session_id"]), "timestamp": row.get("timestamp"),
            "input": {"history": copy.deepcopy(source.get("history", []))}}


def _head(dim, size):
    return nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, size))


def _encoder(dim, heads, layers, dropout):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                                            batch_first=True), layers,
                                 enable_nested_tensor=False)


def _f1(proposed, accepted, truth):
    denominator = len(set(proposed)) + len(set(truth))
    return 2 * len(set(accepted)) / denominator if denominator else 0.0


class BUNTNetwork(nn.Module):
    def __init__(self, item_count, tag_names, config):
        super().__init__()
        dim, heads = int(config.get("hidden_dim", 32)), int(config.get("heads", 1))
        layers, dropout = int(config.get("layers", 1)), float(config.get("dropout", .2))
        if dim < 1 or heads < 1 or dim % heads or layers < 1:
            raise ValueError("Invalid BUNT Transformer dimensions")
        self.mask = item_count
        self.item = nn.Embedding(item_count + 2, dim)
        self.low = _encoder(dim, heads, 1, dropout)
        self.high = _encoder(dim, heads, layers, dropout)
        self.tags = nn.ModuleDict({kind: nn.Embedding(len(names) + 1, dim, padding_idx=len(names))
                                  for kind, names in tag_names.items()})
        self.projections = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers + 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers + 1)])
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout,
                                                                batch_first=True) for _ in range(layers)])
        self.actors = nn.ModuleDict({"item": _head(dim, item_count),
                                    **{kind: _head(dim, len(names)) for kind, names in tag_names.items()}})
        self.results = nn.Embedding(7, dim, padding_idx=0)
        self.manager_slot, self.manager_history = _head(dim, 2), _head(dim, 2)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.values = nn.ModuleDict({name: _head(dim * 2 if name == "manager" else dim, 1)
                                    for name in ("manager", "item", "category", "attribute")})

    def memory(self, bundles):
        device = self.item.weight.device
        if not bundles:
            bundles = [[self.mask + 1]]
        pooled = [self.low(self.item(torch.tensor(items, device=device))[None] * math.sqrt(self.item.embedding_dim)).mean(1)
                  for items in bundles]
        return self.high(torch.stack(pooled, dim=1))

    def state(self, bundles, slots, accepted_tags):
        device = self.item.weight.device
        values = self.item(torch.tensor(slots, device=device))[None]
        tags = torch.zeros_like(values)
        for position, assignments in enumerate(accepted_tags):
            for kind, indices in assignments.items():
                if indices:
                    tags[0, position] += self.tags[kind](torch.tensor(sorted(indices), device=device)).mean(0)
        memory = self.memory(bundles)
        for layer, norm, projection in zip(self.decoder, self.norms, self.projections):
            values = layer(norm(values + projection(tags)), memory)
        return self.norms[-1](values + self.projections[-1](tags))[0]

    def manager_feature(self, states, results):
        codes = torch.tensor(results or [0], device=states.device)
        return torch.cat((states.mean(0), self.results(codes).mean(0)), -1)

    def policy(self, name, feature):
        if name != "manager":
            return self.actors[name](feature)
        slot, history = feature.chunk(2, dim=-1)
        gate = self.gate(feature)
        probability = (1 - gate) * self.manager_slot(slot).softmax(-1) + gate * self.manager_history(history).softmax(-1)
        return probability.clamp_min(1e-12).log()


class BUNT:
    name = "bunt"

    def __init__(self, config):
        self.config = dict(config)
        self.training_history, self.fit_metadata, self.fitted = [], {}, False

    def _path(self, name):
        value = self.config.get(name)
        if not value:
            raise ValueError(f"BUNT requires {name}; native tags/history cannot be inferred from targets or title words")
        path = Path(os.path.expanduser(os.path.expandvars(str(value))))
        if not path.is_file():
            raise FileNotFoundError(f"BUNT asset missing: {name}: {path}")
        return path

    def _setup(self, train, validation, catalog):
        if not train or not catalog:
            raise ValueError("BUNT requires training sessions and catalog")
        self.item_ids = [row["item_id"] for row in catalog]
        if len(self.item_ids) != len(set(self.item_ids)) or any(not isinstance(item, str) or not item for item in self.item_ids):
            raise ValueError("Catalog IDs must be unique nonempty strings")
        self.index = {item: i for i, item in enumerate(self.item_ids)}
        self.seed = int(self.config.get("seed", 42))
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device(self.config.get("device", "cpu"))
        if self.config.get("cpu_threads"):
            torch.set_num_threads(int(self.config["cpu_threads"]))
        self.max_rounds, self.slot_count = int(self.config.get("max_rounds", 20)), int(self.config.get("slots_per_round", 2))
        if not 1 <= self.max_rounds <= 20 or not 1 <= self.slot_count <= len(catalog):
            raise ValueError("BUNT needs 1–20 rounds and a valid slots_per_round")
        metadata_path, history_path = self._path("catalog_metadata_path"), self._path("history_bundles_path")
        self.metadata = json.loads(metadata_path.read_text())
        self.history_bundles = json.loads(history_path.read_text())
        if set(self.metadata) != set(self.item_ids):
            raise ValueError("BUNT metadata must cover exactly the supplied catalog")
        for item, values in self.metadata.items():
            if set(values) != {"categories", "attributes"}:
                raise ValueError("BUNT item metadata requires categories and attributes")
            for tags in values.values():
                if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags) or len(tags) != len(set(tags)):
                    raise ValueError("BUNT tags must be lists of distinct nonempty strings")
        self.tag_names = {kind: sorted({tag for value in self.metadata.values() for tag in value[key]})
                          for kind, key in (("category", "categories"), ("attribute", "attributes"))}
        if any(not names for names in self.tag_names.values()):
            raise ValueError("BUNT needs nonempty real category and attribute vocabularies")
        self.tag_index = {kind: {tag: i for i, tag in enumerate(names)} for kind, names in self.tag_names.items()}
        self.item_tags = {kind: [set(self.tag_index[kind][tag] for tag in self.metadata[item][key]) for item in self.item_ids]
                          for kind, key in (("category", "categories"), ("attribute", "attributes"))}
        ids = [[row["session_id"] for row in rows] for rows in (train, validation)]
        if any(len(part) != len(set(part)) for part in ids) or set(ids[0]) & set(ids[1]):
            raise ValueError("BUNT train/validation session IDs must be unique and disjoint")
        for row in list(train) + list(validation):
            if not set(gold_items(row)) <= self.index.keys():
                raise ValueError("Training target outside the BUNT catalog")
            self._history(_context(row))
        self.model = BUNTNetwork(len(catalog), self.tag_names, self.config).to(self.device)
        self.tag_weights = {}
        for kind, names in self.tag_names.items():
            counts = torch.tensor([sum(tag in tags for tags in self.item_tags[kind]) for tag in range(len(names))],
                                  dtype=torch.float32, device=self.device)
            self.tag_weights[kind] = len(catalog) / (len(names) * counts.clamp_min(1))
        self.fit_identity = digest({"version": 2, "train": train, "validation": validation, "catalog": catalog,
                                    "config": {key: value for key, value in self.config.items() if key != "work_dir"},
                                    "metadata_sha256": file_hash(metadata_path), "history_sha256": file_hash(history_path),
                                    "code": {name: file_hash(Path(__file__).with_name(name)) for name in ("bunt.py", "bunt_feedback.py")}})

    def _history(self, context):
        if any(key in context for key in ("target", "target_items", "gold_items")):
            raise ValueError("BUNT prediction context cannot include target labels")
        sid = str(context["session_id"])
        if sid not in self.history_bundles:
            raise ValueError("Missing BUNT historical bundle membership for session " + sid)
        history = self.history_bundles[sid]
        if not isinstance(history, list):
            raise ValueError("Historical bundles must be a list")
        bundles = []
        for event in history:
            if set(event) != {"timestamp", "item_ids"} or type(event["timestamp"]) is not int:
                raise ValueError("Historical bundles require timestamp and item_ids")
            if context.get("timestamp") is None or event["timestamp"] >= context["timestamp"]:
                raise ValueError("BUNT history must precede the current session")
            items = event["item_ids"]
            if not isinstance(items, list) or not items or len(items) != len(set(items)) or not set(items) <= self.index.keys():
                raise ValueError("Invalid historical bundle item IDs")
            bundles.append([self.index[item] for item in items])
        return bundles

    def _offline_loss(self, row, rng):
        gold = [self.index[item] for item in gold_items(row)]
        selected = rng.sample(gold, rng.randint(1, len(gold)))
        positions = rng.sample(range(len(selected)), rng.randint(1, min(len(selected), self.slot_count)))
        slots, tags = list(selected), []
        for item in selected:
            tags.append({kind: {tag for tag in self.item_tags[kind][item] if rng.random() >= self.config.get("tag_mask_probability", .5)}
                         for kind in self.tag_names})
        for position in positions:
            slots[position] = self.model.mask
        state = self.model.state(self._history(_context(row)), slots, tags)[positions]
        truth = torch.tensor([selected[position] for position in positions], device=self.device)
        item_logits = self.model.policy("item", state)
        loss = F.cross_entropy(item_logits, truth)
        tag_loss, tag_hit = state.sum() * 0, torch.zeros(len(positions), dtype=torch.bool, device=self.device)
        for kind, names in self.tag_names.items():
            logits = self.model.policy(kind, state)
            indicators = torch.zeros_like(logits)
            for index, item in enumerate(truth.tolist()):
                indicators[index, sorted(self.item_tags[kind][item])] = 1
            tag_loss = tag_loss - (indicators * self.tag_weights[kind] * logits.log_softmax(-1)).sum(-1).mean()
            tag_hit |= indicators.gather(1, logits.detach().argmax(-1, keepdim=True)).squeeze(-1).bool()
        item_hit = item_logits.detach().argmax(-1) == truth
        valid = item_hit | tag_hit
        if valid.any():
            tag_loss += F.cross_entropy(self.model.manager_slot(state[valid]), item_hit[valid].long())
        return loss + float(self.config.get("ask_loss_weight", .1)) * tag_loss

    def _action(self, name, feature, allowed, rng, stochastic):
        logits = self.model.policy(name, feature)
        allowed = sorted(allowed)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[allowed] = False
        log_probabilities = logits.masked_fill(mask, -torch.inf).log_softmax(-1)
        action = rng.choices(allowed, weights=log_probabilities[allowed].exp().cpu().tolist())[0] if stochastic else int(log_probabilities.argmax())
        return action, {"name": name, "feature": feature.detach(), "allowed": allowed, "action": action,
                        "old_logp": float(log_probabilities[action]), "old_value": float(self.model.values[name](feature).squeeze()),
                        "reward": 0.0}

    @torch.no_grad()
    def _episode(self, context, provider, rng, stochastic=False):
        bundles = self._history(context)
        slots, tags, active = [], [], []
        asked, blocked_tags = {}, {kind: set() for kind in self.tag_names}
        blocked_items, accepted, proposed, outcomes, transcript, records = set(), [], [], [], [], []
        def add_slot():
            position = len(slots)
            slots.append(self.model.mask)
            tags.append({kind: set() for kind in self.tag_names})
            active.append(position)
            asked[position] = {kind: set() for kind in self.tag_names}
        for _ in range(self.slot_count):
            add_slot()
        termination = "maximum_rounds"
        for turn in range(self.max_rounds):
            item_pool = set(range(len(self.item_ids))) - blocked_items
            if not item_pool:
                termination = "catalog_exhausted"
                break
            state = self.model.state(bundles, slots, tags)
            manager_feature = self.model.manager_feature(state[active], outcomes)
            can_ask = any(set(range(len(names))) - asked[position][kind] - blocked_tags[kind]
                          for position in active for kind, names in self.tag_names.items())
            action, manager = self._action("manager", manager_feature, [0, 1] if can_ask else [1], rng, stochastic)
            manager["turn"] = turn
            manager["history_codes"] = list(outcomes or [0])
            turn_records, proposals, queries = [], [], []
            if action == 1:
                temporary = list(slots)
                for position in active:
                    available = item_pool - {proposal["index"] for proposal in proposals}
                    if not available:
                        break
                    state = self.model.state(bundles, temporary, tags)
                    item, record = self._action("item", state[position], available, rng, stochastic)
                    temporary[position] = item
                    proposals.append({"slot": position, "item_id": self.item_ids[item], "index": item})
                    record.update(slot=position, turn=turn)
                    turn_records.append(record)
                request = {"turn": turn + 1, "action": "recommend",
                           "proposals": [{key: value for key, value in proposal.items() if key != "index"} for proposal in proposals]}
            else:
                for position in active:
                    for kind, names in self.tag_names.items():
                        available = set(range(len(names))) - asked[position][kind] - blocked_tags[kind]
                        if not available:
                            continue
                        tag, record = self._action(kind, state[position], available, rng, stochastic)
                        queries.append({"slot": position, "kind": kind, "tag": names[tag]})
                        record.update(slot=position, turn=turn)
                        turn_records.append(record)
                request = {"turn": turn + 1, "action": "ask", "queries": queries}
            response = provider(copy.deepcopy(request))
            if response is None:
                termination = "respondent_stopped"
                break
            if not isinstance(response, dict) or set(response) - {"feedback", "done"} or type(response.get("done", False)) is not bool:
                raise ValueError("Feedback must contain only feedback and optional boolean done")
            answers = response.get("feedback")
            expected = request.get("proposals", request.get("queries"))
            if not isinstance(answers, list) or len(answers) != len(expected):
                raise ValueError("Feedback must answer exactly the submitted proposals or queries")
            for answer, query in zip(answers, expected):
                if not isinstance(answer, dict) or set(answer) != set(query) | {"decision"} or any(answer[key] != value for key, value in query.items()):
                    raise ValueError("Feedback may refer only to the submitted item or tag")
                if answer["decision"] not in {"accept", "reject", "ignore"}:
                    raise ValueError("Feedback decision must be accept, reject, or ignore")
            successes, newly_accepted = 0, []
            for answer, record in zip(answers, turn_records):
                position, decision = answer["slot"], answer["decision"]
                record["reward"] = float(decision == "accept")
                successes += decision == "accept"
                if action == 1:
                    item = self.index[answer["item_id"]]
                    blocked_items.add(item)
                    proposed.append(item)
                    if decision == "accept":
                        slots[position] = item
                        accepted.append(item)
                        newly_accepted.append(position)
                else:
                    kind, tag = answer["kind"], self.tag_index[answer["kind"]][answer["tag"]]
                    asked[position][kind].add(tag)
                    if decision == "accept":
                        tags[position][kind].add(tag)
                    elif decision == "reject":
                        blocked_tags[kind].add(tag)
                        blocked_items.update(item for item, item_tags in enumerate(self.item_tags[kind]) if tag in item_tags)
            outcomes.append((3 if successes else 4) if action else (1 if successes else 2))
            transcript.append({"request": request, "response": response})
            records.extend(turn_records + [manager])
            if response.get("done", False):
                outcomes[-1] = 5
                termination = "respondent_complete"
                break
            for position in newly_accepted:
                active.remove(position)
                add_slot()
        return {"proposed": proposed, "accepted": accepted, "transcript": transcript,
                "termination": termination, "slot_count": len(slots), "records": records}

    def _returns(self, records, reward):
        managers = [record for record in records if record["name"] == "manager"]
        if managers:
            managers[-1]["reward"] = reward
        gamma, gae = float(self.config.get("gamma", .99)), float(self.config.get("gae_lambda", .95))
        for name in ("manager", "item", "category", "attribute"):
            stream = [record for record in records if record["name"] == name]
            advantage, next_value = 0.0, 0.0
            for record in reversed(stream):
                delta = record["reward"] + gamma * next_value - record["old_value"]
                advantage = delta + gamma * gae * advantage
                record["advantage"], record["return"] = advantage, advantage + record["old_value"]
                next_value = record["old_value"]

    def _update(self, records, optimizer):
        losses = []
        size, clip = int(self.config.get("rl_batch_size", 128)), float(self.config.get("ppo_clip", .2))
        for _ in range(int(self.config.get("ppo_updates", 10))):
            self.rng.shuffle(records)
            for start in range(0, len(records), size):
                objectives = []
                for record in records[start:start + size]:
                    feature, name = record["feature"], record["name"]
                    if name == "manager":
                        codes = torch.tensor(record["history_codes"], device=self.device)
                        feature = torch.cat((feature[:self.model.item.embedding_dim], self.model.results(codes).mean(0)), -1)
                    scores = self.model.policy(name, feature)
                    mask = torch.ones_like(scores, dtype=torch.bool)
                    mask[record["allowed"]] = False
                    logp = scores.masked_fill(mask, -torch.inf).log_softmax(-1)[record["action"]]
                    ratio, advantage = (logp - record["old_logp"]).exp(), record["advantage"]
                    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                    value = self.model.values[name](feature).squeeze()
                    objectives.append(policy + float(self.config.get("value_coefficient", .2)) * (value - record["return"]).square())
                loss = torch.stack(objectives).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite BUNT PPO objective")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([parameter for parameter in self.model.parameters() if parameter.requires_grad],
                                        float(self.config.get("grad_clip", .5)))
                optimizer.step()
                losses.append(float(loss.detach()))
        return sum(losses) / len(losses) if losses else 0.0

    def fit(self, train, validation, catalog):
        self._setup(train, validation, catalog)
        directory = Path(self.config["work_dir"]) if self.config.get("work_dir") else None
        if directory and (directory / "bunt-fit.json").exists():
            receipt = json.loads((directory / "bunt-fit.json").read_text())
            weights = directory / "bunt-model.pt"
            if receipt["identity_sha256"] != self.fit_identity or file_hash(weights) != receipt["weights_sha256"]:
                raise ValueError("BUNT fitted model identity or weights changed")
            self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
            self.fit_metadata, self.fitted = receipt, True
            self.model.eval()
            return self
        epochs, batch_size = int(self.config.get("epochs", 100)), int(self.config.get("batch_size", 32))
        if epochs < 1 or batch_size < 1:
            raise ValueError("BUNT epochs and batch_size must be positive")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get("learning_rate", .001)))
        best, stale, selected = math.inf, 0, None
        for epoch in range(epochs):
            self.model.train()
            order = list(train)
            self.rng.shuffle(order)
            losses = []
            for start in range(0, len(order), batch_size):
                loss = torch.stack([self._offline_loss(row, self.rng) for row in order[start:start + batch_size]]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite BUNT cloze objective")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip", .5)))
                optimizer.step()
                losses.append(float(loss.detach()))
            self.model.eval()
            with torch.no_grad():
                checks = [float(self._offline_loss(row, random.Random(self.seed + index))) for index, row in enumerate(validation)]
            score = sum(checks or losses) / len(checks or losses)
            self.training_history.append({"stage": "offline_cloze", "epoch": epoch + 1, "training_loss": sum(losses) / len(losses),
                                          "validation_loss": score if checks else None})
            if score < best:
                best, stale, selected = score, 0, copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if validation and stale >= int(self.config.get("patience", 10)):
                    break
        self.model.load_state_dict(selected)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (self.model.actors, self.model.manager_slot, self.model.manager_history, self.model.gate, self.model.values, self.model.results):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        optimizer = torch.optim.Adam([parameter for parameter in self.model.parameters() if parameter.requires_grad],
                                     lr=float(self.config.get("rl_learning_rate", .001)))
        episodes, rollout = int(self.config.get("online_episodes", 50000)), int(self.config.get("rollout_episodes", 16))
        if episodes < 1 or rollout < 1:
            raise ValueError("Full BUNT requires positive online_episodes and rollout_episodes")
        best, selected = -math.inf, None
        for start in range(0, episodes, rollout):
            self.model.eval()
            records, rewards = [], []
            for _ in range(min(rollout, episodes - start)):
                row = self.rng.choice(train)
                result = self._episode(_context(row), TargetFeedback(gold_items(row), self.metadata, protocol="training_simulation"), self.rng, True)
                reward = _f1(result["proposed"], result["accepted"], gold_items(row))
                self._returns(result["records"], reward)
                records.extend(result["records"])
                rewards.append(reward)
            loss = self._update(records, optimizer)
            checks = []
            for index, row in enumerate(validation):
                result = self._episode(_context(row), TargetFeedback(gold_items(row), self.metadata, protocol="validation_simulation"), random.Random(self.seed + index))
                checks.append(_f1(result["proposed"], result["accepted"], gold_items(row)))
            score = sum(checks or rewards) / len(checks or rewards)
            self.training_history.append({"stage": "online_four_policy_ppo", "episodes": min(episodes, start + rollout),
                                          "training_reward": sum(rewards) / len(rewards), "ppo_loss": loss,
                                          "validation_reward": score if checks else None})
            if score > best:
                best, selected = score, copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(selected)
        self.model.eval()
        self.fitted = True
        self.fit_metadata = {"identity_sha256": self.fit_identity, "method": "bunt", "implementation": "bundle_mcr_paper_reimplementation",
                             "training_history": self.training_history, "counts": {"train": len(train), "validation": len(validation), "items": len(catalog)},
                             "history_profile": "timestamp_checked_historical_bundles", "tag_profile": "supplied_native_category_attribute_metadata",
                             "max_rounds": self.max_rounds, "slots_per_round": self.slot_count,
                             "prediction_output": "all_proposed_items", "online_reward": "f1_over_all_proposed_items"}
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "bunt-model.pt.tmp"
            torch.save(self.model.state_dict(), temporary)
            temporary.replace(directory / "bunt-model.pt")
            self.fit_metadata["weights_sha256"] = file_hash(directory / "bunt-model.pt")
            atomic_json(directory / "bunt-fit.json", self.fit_metadata)
        return self

    def predict(self, context, k, seed):
        if not self.fitted:
            raise RuntimeError("BUNT must be fitted before prediction")
        self._history(context)
        provider = context.get("feedback_provider")
        if not callable(provider):
            raise ValueError("BUNT is interactive and requires feedback_provider; one-shot fallback is not permitted")
        self.model.eval()
        result = self._episode(context, provider, random.Random(seed))
        return Prediction([self.item_ids[item] for item in result["proposed"]],
                          {"method": "bunt", "implementation": "bundle_mcr_paper_reimplementation", "llm_calls": 0,
                           "feedback_protocol": getattr(provider, "protocol", "external_respondent"),
                           "conversation_turns": len(result["transcript"]), "termination": result["termination"],
                           "slots_created": result["slot_count"], "slots_per_round": self.slot_count,
                           "accepted_items": [self.item_ids[item] for item in result["accepted"]],
                           "output_semantics": "all_proposed_items", "transcript": result["transcript"],
                           "fixed_k_argument_unused_for_interactive_bundle": k})
