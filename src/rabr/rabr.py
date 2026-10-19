from __future__ import annotations

from dataclasses import asdict
import math
import json

from .baselines.llm_common import CatalogIndex, context_text, parse_items
from .calibration import CalibrationExample, CalibrationGrid, calibrate, ground_round, prefix_scores, select_bundle
from .io import digest, stable_seed
from .types import gold_items, input_only


class RABR:
    def __init__(self, config, backend, catalog, ledger, seed):
        self.config, self.backend, self.ledger, self.seed = dict(config), backend, ledger, seed
        self.index = CatalogIndex(catalog, backend)
        self.grid = CalibrationGrid(tuple(config.get("budgets", [1, 2, 4, 8, 10])),
                                    tuple(config.get("thresholds", [i / 20 for i in range(21)])))
        self.gamma, self.beta = float(config.get("gamma", .5)), float(config.get("beta", 1))
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        self.mode = config.get("score_mode", "equation8")
        self.ablation = config.get("ablation", "none")
        if self.mode != "equation8":
            raise ValueError("RABR uses the Equation 8 score")
        if self.ablation not in {"none", "without_grounding", "without_relevance", "multi_sample_fixed_k", "fixed_k"}:
            raise ValueError("Unknown RABR ablation")
        self.effective_gamma = {"without_grounding": 0.0, "without_relevance": 1.0,
                                "multi_sample_fixed_k": 1.0}.get(self.ablation, self.gamma)
        self.empty_history_policy = config.get("empty_history_policy", "require")
        if self.empty_history_policy not in {"require", "neutral"}:
            raise ValueError("empty_history_policy must be require or neutral")

    def relevance(self, context):
        history = context["input"]["history"]
        profile = "Prior interactions: " + json.dumps(
            [{"title": self.index.catalog[self.index.lookup[item["item_id"]]]["title"],
              "category": self.index.catalog[self.index.lookup[item["item_id"]]].get("category")} for item in history],
            ensure_ascii=False)
        if not history:
            if self.effective_gamma < 1 and self.empty_history_policy == "require":
                raise ValueError("Equation 8 requires interaction history; supply genuine prior interactions "
                                 "or explicitly select empty_history_policy=neutral for a cold-start adaptation")
            return {item: .5 for item in self.index.ids}, digest({"history": [], "policy": self.empty_history_policy})
        cosines = (self.index.encode([profile])[0] @ self.index.vectors.T).clip(-1, 1)
        return {item: (1 + float(score)) / 2 for item, score in zip(self.index.ids, cosines)}, digest(profile)

    def messages(self, context, candidates, v_max):
        content = context_text(context)
        if self.ablation != "multi_sample_fixed_k":
            titles = [self.index.catalog[self.index.lookup[item]]["title"] for item in sorted(candidates)]
            content += "\nCandidate items collected so far: " + json.dumps(titles, ensure_ascii=False)
            content += ("\nUsing the provided interaction history, suggest additional relevant items for the user's bundle. "
                        "Do not repeat any of the listed items.")
        content += f"\nReturn at most {v_max} descriptions."
        return [{"role": "system", "content": "Recommend complementary products for the supplied intent and prior interactions. Return only JSON with an items list of product-description strings."},
                {"role": "user", "content": content}]

    def scores(self, context, budget, split):
        context = input_only(context)
        v_max = int(self.config.get("v_max", 8))
        relevance, profile_hash = self.relevance(context)
        rounds, traces, candidates = [], [], set()
        for index in range(budget):
            messages = self.messages(context, candidates, v_max)
            expected = {"context_sha256": digest(context), "prompt_sha256": digest(messages),
                        "profile_sha256": profile_hash, "candidate_ids": sorted(candidates)}
            key = f"round/{split}/{context['session_id']}/{index}"
            value = self.ledger.get(key)
            if value is None:
                seed = stable_seed(self.seed, split, context["session_id"], index)
                raw = self.backend.generate(messages, seed)
                descriptions, error = parse_items(raw)
                cosines = (self.index.encode(descriptions[:v_max]) @ self.index.vectors.T).clip(-1, 1) if descriptions else []
                scores = ground_round(self.index.ids, cosines, tau=float(self.config.get("tau", 0.0)), v_max=v_max)
                value = {"scores": scores, "raw_output": raw, "parse_error": error, "seed": seed,
                         "cost": dict(self.backend.last_generation), **expected}
                self.ledger.put(key, value)
            if any(value.get(key) != item for key, item in expected.items()):
                raise ValueError("Input, profile or candidate-aware prompt changed since generation checkpoint")
            rounds.append(value["scores"])
            candidates.update(value["scores"])
            traces.append(value)
        return prefix_scores(rounds, relevance, gamma=self.effective_gamma), traces

    def select(self, calibration_rows, k):
        fraction = float(self.config.get("calibration_fraction", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError("calibration_fraction must be in (0,1]")
        rows = sorted(calibration_rows, key=lambda row: digest([self.seed, row["session_id"]]))
        rows = rows[:max(1, math.floor(len(rows) * fraction))]
        budget = max(self.grid.budgets)
        examples = []
        for row in rows:
            scores, _ = self.scores(row, budget, "calibration")
            examples.append(CalibrationExample({s: scores[s - 1] for s in self.grid.budgets}, frozenset(gold_items(row))))
        if self.ablation in {"fixed_k", "multi_sample_fixed_k"}:
            selection = {"status": "uncertified_ablation", "budget": budget,
                         "threshold": float(self.config.get("fixed_threshold", .5)), "fixed_k": k}
            values = [self.bundle(example.scores_by_budget[budget], selection) for example in examples]
        else:
            result = calibrate(examples, self.grid, alpha=float(self.config.get("alpha", .1)),
                               delta=float(self.config.get("delta", .1)), beta=self.beta)
            selection = asdict(result)
            selection["grid"] = {"budgets": list(self.grid.budgets), "thresholds": list(self.grid.thresholds)}
            if result.selected is None:
                selection.update(calibration_mean_bundle_size=None, calibration_session_ids=[row["session_id"] for row in rows])
                return selection
            selection.update(budget=result.selected.budget, threshold=result.selected.threshold)
            values = [self.bundle(example.scores_by_budget[result.selected.budget], selection) for example in examples]
        selection.update(calibration_mean_bundle_size=sum(map(len, values)) / len(values),
                         calibration_session_ids=[row["session_id"] for row in rows],
                         score_mode=self.mode, ablation=self.ablation, effective_gamma=self.effective_gamma,
                         empty_history_policy=self.empty_history_policy,
                         guarantee="none_uncertified_ablation" if selection["status"] == "uncertified_ablation" else
                                   "conditional_on_bounded_loss_sampling_and_prefixed_grid_assumptions")
        return selection

    def bundle(self, scores, selection):
        if self.ablation in {"fixed_k", "multi_sample_fixed_k"}:
            return sorted(scores, key=lambda item: (-scores[item], item))[:selection["fixed_k"]]
        return sorted(select_bundle(scores, selection["threshold"]))

    def predict(self, context, selection):
        prefixes, traces = self.scores(context, selection["budget"], "test")
        return {"session_id": context["session_id"], "predicted_items": self.bundle(prefixes[-1], selection),
                "metadata": {"budget": selection["budget"], "threshold": selection.get("threshold"),
                             "parse_failures": sum(bool(x["parse_error"]) for x in traces)}}
