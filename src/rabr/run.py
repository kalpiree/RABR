from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time

from .backend import create_backend
from .data import load_dataset
from .io import Ledger, atomic_json, canonical, digest, file_hash, stable_seed
from .metrics import evaluate_records
from .rabr import RABR
from .types import gold_items, input_only


def code_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def verify_result(path, identity=None):
    path = Path(path)
    result = json.loads(path.read_text())
    if result["status"] not in {"complete", "no_certified_pair"}:
        raise ValueError("Run is not terminal")
    if identity and any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("Completed run identity differs")
    required = {"predictions.jsonl", "metrics.json", "per-session.jsonl"} if result["status"] == "complete" else {"selection.json"}
    if not required <= result["artifacts"].keys():
        raise ValueError("Missing required completion artifacts")
    for name, expected in result["artifacts"].items():
        artifact = (path.parent / name).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or file_hash(artifact) != expected:
            raise ValueError("Completion artifact changed: " + name)
    return result


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _environment():
    versions = {}
    for name in ("numpy", "torch", "transformers", "faiss-cpu", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def _model_inventory(config):
    inventory = {}
    for key in ("model_path", "encoder_path"):
        value = config.get(key)
        if not value:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if path.is_dir():
            files = {str(p.relative_to(path)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                     for p in sorted(path.rglob("*")) if p.is_file()}
            inventory[key] = {"path": str(path.resolve()), "inventory_sha256": digest(files), "files": files,
                              "identity_type": "file_names_sizes_and_modification_times"}
        else:
            revision = config.get(key.replace("path", "revision"))
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"{key} must be an existing local snapshot, or a model ID with a 40-character commit revision")
            inventory[key] = {"model_id": value, "revision": revision}
    return inventory


def execute(config_path, data_root, domain, output, seed):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this run") from error
        return _execute(Path(config_path), data_root, domain, output, seed)


def _execute(config_path, data_root, domain, output, seed):
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or type(seed) is not int or seed < 0:
        raise ValueError("Unsupported configuration or seed")
    dataset = load_dataset(data_root, domain)
    algorithm = {**config.get("algorithm", {}), **config.get("research", {}).get("parameters", {})}
    for budget_key in ("budget", "sampling_budget"):
        if budget_key in algorithm:
            algorithm["budgets"] = [algorithm[budget_key]]
    algorithm.update(seed=seed, work_dir=str(output / "model"))
    backend_config = dict(config["backend"])
    if config["method"] in {"pog", "bunt", "bgn", "byob"}:
        inventory = {}
    else:
        if config["method"] == "ddbc":
            backend_config.pop("model_path", None)
        inventory = _model_inventory(backend_config)
    if config["method"] == "kd4bg" and algorithm.get("mode") == "teacher_student":
        inventory["student_model"] = _model_inventory({"model_path": algorithm.get("student_model", ""),
                                                       "model_revision": algorithm.get("student_revision")})
        if not inventory["student_model"]:
            raise ValueError("teacher_student requires a student_model snapshot")
    identity = {"config_sha256": file_hash(config_path), "data_manifest_sha256": dataset.manifest_sha256,
                "code_sha256": code_hash(), "seed": seed, "method": config["method"], "domain": domain,
                "model_inventory_sha256": digest(inventory)}
    result_path = output / "result.json"
    if result_path.exists():
        return verify_result(result_path, identity)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run inputs changed; use a new output directory")
    atomic_json(identity_path, identity)
    ledger = Ledger(output / "checkpoint.jsonl", digest(identity))
    backend = create_backend(backend_config, config["profile"], dataset.catalog)
    policy = algorithm.get("output_size_policy", "training_median")
    k = int(algorithm.get("fixed_k", dataset.fixed_k))
    if policy == "paper_rabr_mean" and config["method"] != "rabr":
        reference = verify_result(output / algorithm["rabr_reference"])
        if (reference["status"] != "complete" or reference["method"] != "rabr" or reference["domain"] != domain
                or reference["seed"] != seed or reference["data_manifest_sha256"] != dataset.manifest_sha256
                or reference["profile"] != config["profile"] or reference["code_sha256"] != identity["code_sha256"]):
            raise ValueError("Invalid RABR size reference")
        reference_selection = json.loads((output / algorithm["rabr_reference"]).with_name("selection.json").read_text())
        if reference_selection["status"] != "certified":
            raise ValueError("The size reference must be a certified RABR run")
        k = max(1, math.floor(reference["calibration_mean_bundle_size"] + .5))
    elif policy != "training_median" and policy != "paper_rabr_mean":
        raise ValueError("Unknown output_size_policy")
    if k < 1:
        raise ValueError("Output size must be positive")
    status, selection, fit_metadata = "complete", None, {}
    records = []
    if config["method"] == "rabr":
        engine = RABR(algorithm, backend, dataset.catalog, ledger, seed)
        selection = engine.select(dataset.calibration, k)
        atomic_json(output / "selection.json", selection)
        if selection["status"] == "no_certified_pair":
            status = "no_certified_pair"
        else:
            for row in dataset.test:
                context = input_only(row)
                key = "prediction/" + context["session_id"]
                prediction = ledger.get(key)
                if prediction is None:
                    prediction = engine.predict(context, selection)
                    ledger.put(key, prediction)
                records.append(prediction)
    else:
        if config["method"] in {"pog", "bunt", "bgn", "byob"}:
            from .baselines.traditional import create_traditional
            predictor = create_traditional(config["method"], algorithm)
        else:
            from .baselines.generative import create_generative
            predictor = create_generative(config["method"], algorithm, backend)
        contexts = [input_only(row) for row in dataset.test]
        if any(ledger.get("prediction/" + row["session_id"]) is None for row in contexts):
            predictor.fit(dataset.train, dataset.validation, dataset.catalog)
            fit_metadata = getattr(predictor, "fit_metadata", {})
            atomic_json(output / "fit.json", fit_metadata)
        elif (output / "fit.json").exists():
            fit_metadata = json.loads((output / "fit.json").read_text())
        for context in contexts:
            key = "prediction/" + context["session_id"]
            prediction = ledger.get(key)
            if prediction is None:
                previous_stats = dict(backend.stats)
                prediction_start = time.monotonic()
                value = predictor.predict(context, k, stable_seed(seed, "test", context["session_id"]))
                if not isinstance(value.items, list) or not set(value.items) <= {x["item_id"] for x in dataset.catalog}:
                    raise ValueError("Predictor returned invalid catalog IDs")
                prediction = {"session_id": context["session_id"], "predicted_items": list(dict.fromkeys(value.items)),
                              "metadata": value.metadata, "wall_seconds": time.monotonic() - prediction_start,
                              "cost": {key: backend.stats[key] - previous_stats[key] for key in previous_stats}}
                ledger.put(key, prediction)
            records.append(prediction)
    summary = None
    if status == "complete":
        _write_jsonl(output / "predictions.jsonl", records)
        truths = {row["session_id"]: gold_items(row) for row in dataset.test}
        summary, per_session = evaluate_records([{**row, "gold_items": truths[row["session_id"]]} for row in records], beta=algorithm.get("beta", 1))
        atomic_json(output / "metrics.json", summary)
        _write_jsonl(output / "per-session.jsonl", per_session)
    cost = {"attempt_wall_seconds": time.monotonic() - started, "attempt_backend": backend.stats,
                                      "durable_prediction_cost": [row.get("cost", {}) for row in records],
                                      "round_costs": [v["cost"] for key, v in ledger.rows.items() if key.startswith("round/")]}
    atomic_json(output / "cost.json", cost)
    atomic_json(output / "environment.json", _environment())
    atomic_json(output / "models.json", inventory)
    artifacts = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name not in {"result.json", ".run.lock", "failure.json"} and p.suffix not in {".tmp", ".log"}}
    result = {"schema_version": 1, "status": status, **identity, "profile": config["profile"],
              "backbone": config.get("backbone", "none"), "research": config.get("research", {}),
              "output_size_policy": policy, "fixed_k": k, "fit": fit_metadata,
              "calibration_mean_bundle_size": selection.get("calibration_mean_bundle_size") if selection else None,
              "artifacts": artifacts, "protocol": dataset.manifest.get("protocol", {}),
              "metrics": summary, "cost": cost,
              "metric_status": "smoke_only" if config["profile"] == "smoke" else "fresh_protocol"}
    atomic_json(result_path, result)
    return verify_result(result_path, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train, calibrate and evaluate one immutable configuration")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = execute(args.config, args.data, args.domain, args.output, args.seed)
    except Exception as error:
        retryable = isinstance(error, (MemoryError, OSError)) and not isinstance(error, (FileNotFoundError, PermissionError))
        retryable = retryable or type(error).__name__ == "OutOfMemoryError"
        atomic_json(args.output / "failure.json", {"error_type": type(error).__name__, "message": str(error), "retryable": retryable})
        raise
    (args.output / "failure.json").unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
