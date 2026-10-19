from __future__ import annotations

import copy
import itertools
import json
import os
import re
from pathlib import Path

from .data import atomic_json, digest, file_hash, load_dataset

METHODS = ("pog", "bunt", "bgn", "byob", "zero_shot", "llm4b", "text2bundle", "aicl", "kd4bg", "ddbc", "rabr")
NON_LLM = {"pog", "bunt", "bgn", "byob", "ddbc"}


def source_hash():
    root = Path(__file__).parent
    return digest({str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))})


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expanded_experiments(spec):
    for row in spec["experiments"]:
        grid = row.get("grid", {})
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("Parameter grids must contain nonempty finite lists")
        names = sorted(grid)
        for values in itertools.product(*(grid[key] for key in names)):
            parameters = {**row.get("parameters", {}), **dict(zip(names, values))}
            yield row, parameters


def build_suite(suite_paths, data_root, output, *, backbone_configs=None, method_directory=None):
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    specs = [(Path(path), json.loads(Path(path).read_text())) for path in suite_paths]
    if not specs:
        raise ValueError("No suites selected")
    profiles = {spec["profile"] for _, spec in specs}
    if len(profiles) != 1 or not profiles <= {"smoke", "experiment"}:
        raise ValueError("Smoke and experiment suites cannot be combined")
    profile = next(iter(profiles))
    backbones = backbone_configs or {}
    method_directory = Path(method_directory) if method_directory else Path(__file__).resolve().parents[2] / "configs" / "methods"
    datasets, configurations, jobs = {}, {}, []
    for suite_path, spec in specs:
        if spec.get("schema_version") != 1 or not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("Invalid suite version or seeds")
        for domain in spec["domains"]:
            datasets.setdefault(domain, load_dataset(data_root, domain))
            if profile == "experiment" and datasets[domain].manifest.get("protocol", {}).get("dataset_role") == "synthetic_smoke_only":
                raise ValueError("Synthetic fixtures cannot be used for experiment profiles")
        for recipe, parameters in expanded_experiments(spec):
            method = recipe["method"]
            if method not in METHODS:
                raise ValueError("Unknown method " + method)
            method_path = method_directory / (method + ".json")
            method_config = json.loads(method_path.read_text()) if method_path.exists() else {}
            defaults = merge(method_config, spec["defaults"])
            selected_backbones = recipe.get("backbones", ["none"] if method in NON_LLM else spec["backbones"])
            for domain, backbone, seed in itertools.product(spec["domains"], selected_backbones, spec["seeds"]):
                if type(seed) is not int or seed < 0 or not re.fullmatch(r"[a-z0-9_-]+", backbone):
                    raise ValueError("Invalid seed/backbone")
                config = merge(defaults, recipe.get("config", {}))
                backend = merge(config["backend"], backbones.get(backbone, backbones.get("default", {})))
                if (backend["type"] == "fixture") != (profile == "smoke"):
                    raise ValueError("Fixture backend is restricted to smoke suites")
                config.update(schema_version=1, profile=profile, method=method, backbone=backbone, backend=backend,
                              research={"rq": spec["rq"], "variant": recipe["variant"], "sweep": recipe.get("grid", {}),
                                        "parameters": parameters, "protocol_notes": spec.get("protocol_notes", [])})
                suffix = digest({"config": config, "domain": domain, "seed": seed})[:12]
                job_id = f"{spec['rq']}-{method}-{backbone}-{domain}-s{seed}-{suffix}"
                if job_id in configurations:
                    raise ValueError("Duplicate experiment configuration")
                configurations[job_id] = config
                jobs.append({"id": job_id, "config": "configs/" + job_id + ".json", "domain": domain,
                             "method": method, "backbone": backbone, "seed": seed, "research": config["research"],
                             "data_manifest_sha256": datasets[domain].manifest_sha256,
                             "output": "runs/" + job_id, "requires": [],
                             "resource": "cpu" if profile == "smoke" else "gpu",
                             "max_attempts": 3, "timeout_seconds": spec.get("timeout_seconds", 21600)})
    for job in jobs:
        config = configurations[job["id"]]
        effective_algorithm = {**config["algorithm"], **config["research"]["parameters"]}
        if effective_algorithm.get("output_size_policy") == "paper_rabr_mean" and job["method"] != "rabr":
            references = [other for other in jobs if other["method"] == "rabr" and other["research"]["rq"] == "rq1"
                          and other["domain"] == job["domain"] and other["seed"] == job["seed"]
                          and other["backbone"] == (job["backbone"] if job["backbone"] != "none" else specs[0][1]["backbones"][0])]
            if len(references) != 1:
                raise ValueError("paper_rabr_mean requires exactly one matching RQ1 RABR job")
            job["requires"] = [references[0]["id"]]
            config["algorithm"]["rabr_reference"] = "../" + references[0]["id"] + "/result.json"
    manifest = {"schema_version": 1, "profile": profile, "code_sha256": source_hash(), "data": os.path.relpath(data_root, output),
                "suite_sha256": {path.name: file_hash(path) for path, _ in specs}, "jobs": jobs}
    if len(manifest["suite_sha256"]) != len(specs):
        raise ValueError("Suite filenames must be distinct")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "queue.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for job in manifest["jobs"]:
            job["config_sha256"] = hashlib_config(configurations[job["id"]])
        if old != manifest:
            raise ValueError("Existing queue differs; choose a new output directory")
        for job in jobs:
            if file_hash(output / job["config"]) != job["config_sha256"]:
                raise ValueError("Existing job config changed")
        return manifest
    for job in jobs:
        path = output / job["config"]
        atomic_json(path, configurations[job["id"]])
        job["config_sha256"] = file_hash(path)
    atomic_json(manifest_path, manifest)
    return manifest


def hashlib_config(config):
    import hashlib
    from .data import canonical
    return hashlib.sha256(canonical(config) + b"\n").hexdigest()
