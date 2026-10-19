from __future__ import annotations

from dataclasses import asdict
import math

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
        if self.mode not in {"equation8", "frequency_similarity"}:
            raise ValueError("Unknown score mode")
        if self.ablation not in {"none", "without_frequency", "without_similarity", "fixed_threshold", "multi_sample_fixed_k", "fixed_k"}:
            raise ValueError("Unknown RABR ablation")
        if self.ablation in {"without_frequency", "without_similarity"} and self.mode != "frequency_similarity":
            raise ValueError("Frequency ablations require the explicitly defined frequency_similarity score")

    def scores(self, context, budget, split):
        context = input_only(context)
        v_max = int(self.config.get("v_max", 8))
        messages = [{"role": "system", "content": "Recommend complementary products for the supplied intent and prior interactions. Return only JSON with an items list of product-description strings."},
                    {"role": "user", "content": context_text(context) + f"\nReturn at most {v_max} descriptions."}]
        rounds = []
        traces = []
        for index in range(budget):
            key = f"round/{split}/{context['session_id']}/{index}"
            value = self.ledger.get(key)
            if value is None:
                seed = stable_seed(self.seed, split, context["session_id"], index)
                raw = self.backend.generate(messages, seed)
                descriptions, error = parse_items(raw)
                cosines = (self.index.encode(descriptions[:v_max]) @ self.index.vectors.T).clip(-1, 1) if descriptions else []
                scores = ground_round(self.index.ids, cosines, tau=float(self.config.get("tau", 0.0)), v_max=v_max)
                value = {"scores": scores, "raw_output": raw, "parse_error": error, "seed": seed,
                         "cost": dict(self.backend.last_generation), "context_sha256": digest(context)}
                self.ledger.put(key, value)
            if value["context_sha256"] != digest(context):
                raise ValueError("Input changed since generation checkpoint")
            rounds.append(value["scores"])
            traces.append(value)
        cosines = (self.index.encode([context_text(context)])[0] @ self.index.vectors.T).clip(-1, 1)
        relevance = {item: (1 + float(score)) / 2 for item, score in zip(self.index.ids, cosines)}
        prefixes = prefix_scores(rounds, relevance, gamma=self.gamma)
        if self.mode == "frequency_similarity" or self.ablation == "multi_sample_fixed_k":
            counts, similarities, alternate = {}, {}, []
            for number, scores in enumerate(rounds, 1):
                for item, value in scores.items():
                    counts[item] = counts.get(item, 0) + 1
                    similarities.setdefault(item, []).append(value)
                current = {}
                for item, count in counts.items():
                    frequency = count / number
                    similarity = (math.fsum(similarities[item]) / count + relevance[item]) / 2
                    if self.ablation in {"without_similarity", "multi_sample_fixed_k"}:
                        current[item] = frequency
                    elif self.ablation == "without_frequency":
                        current[item] = similarity
                    else:
                        current[item] = self.gamma * frequency + (1 - self.gamma) * similarity
                alternate.append(current)
            prefixes = tuple(alternate)
        return prefixes, traces

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
        if self.ablation in {"fixed_threshold", "fixed_k", "multi_sample_fixed_k"}:
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
                         score_mode=self.mode, ablation=self.ablation,
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
