from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "validation", "calibration", "test")
DOMAINS = ("electronic", "clothing", "food")
RAW_FILES = ("item_titles.csv", "item_categories.csv", "user_session.csv", "session_bundle.csv",
             "bundle_intent.csv", "bundle_item.csv", "user_item.csv")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def write_jsonl(path, rows):
    with Path(path).open("wb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def id_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def session_key(row):
    return row["timestamp"], id_key(row.get("original_session_id", row["session_id"].split(":", 1)[-1]))


def boundary(rows, fraction, *, reserve_boundaries=0):
    possible = [i for i in range(1, len(rows)) if rows[i-1]["timestamp"] < rows[i]["timestamp"]]
    if reserve_boundaries:
        possible = possible[:-reserve_boundaries]
    if not possible:
        raise ValueError("Insufficient distinct timestamps for nonempty chronological partitions")
    return min(possible, key=lambda i: (abs(i - fraction * len(rows)), i))


def split_sessions(rows, validation_fraction=.1):
    ordered = sorted(rows, key=session_key)
    first = boundary(ordered, .6, reserve_boundaries=1)
    remaining = ordered[first:]
    candidates = [i for i in range(1, len(remaining)) if remaining[i-1]["timestamp"] < remaining[i]["timestamp"]]
    second = min(candidates, key=lambda i: (abs(first + i - .9 * len(ordered)), i))
    train, validation = split_validation(ordered[:first], validation_fraction)
    return {"train": train, "validation": validation, "calibration": remaining[:second], "test": remaining[second:]}


def split_validation(rows, fraction):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie strictly between zero and one")
    ordered = sorted(rows, key=session_key)
    cut = boundary(ordered, 1-fraction)
    return ordered[:cut], ordered[cut:]


def input_only(row):
    value = row["input"]
    result = {"session_id": row["session_id"], "input": {
        "intent": value["intent"], "has_intent": bool(value["intent"].strip()),
        "history": [{"item_id": h["item_id"], "title": h["title"], "category": h.get("category"),
                     "timestamp": h["timestamp"]} for h in value["history"]]}}
    if "user_id" in row:
        result["user_id"] = str(row["user_id"])
    return result


@dataclass
class Dataset:
    catalog: list
    splits: dict
    patterns: list
    manifest: dict
    manifest_sha256: str
    root: Path

    @property
    def train(self): return self.splits["train"]
    @property
    def validation(self): return self.splits["validation"]
    @property
    def calibration(self): return self.splits["calibration"]
    @property
    def test(self): return self.splits["test"]
    @property
    def fixed_k(self): return self.manifest["fixed_k"]

    def inputs(self, split):
        return [input_only(row) for row in self.splits[split]]


def validate_dataset(catalog, splits, domain):
    known = {row["item_id"] for row in catalog}
    if not catalog or len(known) != len(catalog) or any(not isinstance(x, str) or not x for x in known):
        raise ValueError("Catalog IDs must be unique nonempty strings")
    if any(not isinstance(row.get("title"), str) or not row["title"] for row in catalog):
        raise ValueError("Missing catalog title")
    if set(splits) != set(SPLITS):
        raise ValueError("Expected train, validation, calibration and test")
    seen = set()
    for split in SPLITS:
        if not splits[split]:
            raise ValueError("Empty " + split)
        for row in splits[split]:
            sid, when = row["session_id"], row["timestamp"]
            if not isinstance(sid, str) or not sid.startswith(domain + ":") or sid in seen:
                raise ValueError("Duplicate or invalid session identity")
            seen.add(sid)
            if row["domain"] != domain or type(when) not in (int, float) or not math.isfinite(when):
                raise ValueError("Invalid domain or timestamp")
            intent, history = row["input"]["intent"], row["input"]["history"]
            if not isinstance(intent, str) or not isinstance(history, list):
                raise ValueError("Invalid input")
            if row["input"].get("has_intent", bool(intent.strip())) != bool(intent.strip()):
                raise ValueError("Inconsistent intent flag")
            for event in history:
                if event["item_id"] not in known or not isinstance(event["title"], str):
                    raise ValueError("Invalid history item")
                if type(event["timestamp"]) not in (int, float) or not math.isfinite(event["timestamp"]) or event["timestamp"] >= when:
                    raise ValueError("History must be strictly before the session")
            gold = row["target"]["gold_items"]
            if not isinstance(gold, list) or not gold or len(gold) != len(set(gold)) or not set(gold) <= known:
                raise ValueError("Gold items must be distinct catalog IDs")
    for left, right in zip(SPLITS, SPLITS[1:]):
        if max(r["timestamp"] for r in splits[left]) >= min(r["timestamp"] for r in splits[right]):
            raise ValueError("A timestamp crosses partition boundaries")


def mine_patterns(train, catalog, minimum_support=2):
    categories = {row["item_id"]: row.get("category") for row in catalog}
    supporters = defaultdict(set)
    for row in train:
        values = sorted(categories[item] for item in row["target"]["gold_items"] if categories[item] is not None)
        for length in range(2, min(4, len(values))+1):
            for pattern in set(itertools.combinations(values, length)):
                supporters[pattern].add(row["session_id"])
    return [{"categories": list(pattern), "support": len(ids), "supporting_session_ids": sorted(ids)}
            for pattern, ids in sorted(supporters.items(), key=lambda pair: (-len(pair[1]), pair[0]))
            if len(ids) >= minimum_support]


def load_dataset(root, domain):
    root = Path(root) / domain
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {"catalog.jsonl", "patterns.jsonl"} | {f"{split}.{kind}.jsonl" for split in SPLITS for kind in ("inputs", "targets")}
    if manifest["domain"] != domain or set(manifest["files"]) != expected:
        raise ValueError("Invalid dataset manifest")
    for name, expected_hash in manifest["files"].items():
        if file_hash(root / name) != expected_hash:
            raise ValueError("Data hash mismatch: " + name)
    splits = {}
    for split in SPLITS:
        inputs, targets = read_jsonl(root / f"{split}.inputs.jsonl"), read_jsonl(root / f"{split}.targets.jsonl")
        if [row["session_id"] for row in inputs] != [row["session_id"] for row in targets]:
            raise ValueError("Input/target session IDs or order differ")
        if any("target" in row for row in inputs):
            raise ValueError("Input file contains target labels")
        splits[split] = [{**row, "target": target["target"]} for row, target in zip(inputs, targets)]
    catalog = read_jsonl(root / "catalog.jsonl")
    validate_dataset(catalog, splits, domain)
    patterns = read_jsonl(root / "patterns.jsonl")
    if patterns != mine_patterns(splits["train"], catalog):
        raise ValueError("Patterns differ from training-only mining")
    expected_k = max(2, min(8, math.floor(statistics.median(len(r["target"]["gold_items"]) for r in splits["train"]) + .5)))
    if manifest["fixed_k"] != expected_k:
        raise ValueError("Fixed K is not training-derived")
    return Dataset(catalog, splits, patterns, manifest, file_hash(manifest_path), root)


def save_dataset(root, domain, catalog, splits, provenance, protocol):
    validate_dataset(catalog, splits, domain)
    directory = Path(root) / domain
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output domain already exists; choose a new dataset directory")
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "catalog.jsonl", catalog)
    write_jsonl(directory / "patterns.jsonl", mine_patterns(splits["train"], catalog))
    for split, rows in splits.items():
        write_jsonl(directory / f"{split}.inputs.jsonl", [{k: v for k, v in row.items() if k != "target"} for row in rows])
        write_jsonl(directory / f"{split}.targets.jsonl", [{"session_id": row["session_id"], "target": row["target"]} for row in rows])
    manifest = {"schema_version": 1, "domain": domain, "provenance": provenance, "protocol": protocol,
        "fixed_k": max(2, min(8, math.floor(statistics.median(len(r["target"]["gold_items"]) for r in splits["train"]) + .5))),
        "files": {p.name: file_hash(p) for p in sorted(directory.glob("*.jsonl"))},
        "splits": {name: {"sessions": len(rows), "first_timestamp": min(r["timestamp"] for r in rows),
                          "last_timestamp": max(r["timestamp"] for r in rows),
                          "nonempty_history": sum(bool(r["input"]["history"]) for r in rows)} for name, rows in splits.items()}}
    atomic_json(directory / "manifest.json", manifest)
    return load_dataset(root, domain)


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique_map(rows, key, value):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError("Duplicate source ID: " + row[key])
        result[row[key]] = row[value]
    return result


def category(raw):
    raw = raw.strip()
    if raw in ("[]", "[[]]"):
        return None
    if not raw.startswith("[[") or not raw.endswith("]]"):
        raise ValueError("Invalid category encoding")
    paths = [[part.strip() for part in re.split(r",(?!\s)", body)] for body in re.split(r"\]\s*,\s*\[", raw[2:-2])]
    if any(not part or "[" in part or "]" in part for path in paths for part in path):
        raise ValueError("Ambiguous category path")
    return min(paths, key=lambda path: (-len(path), path))[-1]


def raw_domain(source, domain, seed, validation_fraction, history_limit):
    directory = Path(source) / domain
    source_hashes = {name: file_hash(directory / name) for name in RAW_FILES}
    titles = unique_map(csv_rows(directory / "item_titles.csv"), "item ID", "titles")
    categories = unique_map(csv_rows(directory / "item_categories.csv"), "item ID", "categories")
    if set(titles) != set(categories):
        raise ValueError("Catalog title/category coverage differs")
    catalog = [{"item_id": item, "title": titles[item].strip(), "category": category(categories[item])} for item in sorted(titles, key=id_key)]
    by_item = {row["item_id"]: row for row in catalog}
    sessions = {}
    for row in csv_rows(directory / "user_session.csv"):
        sid = row["session ID"]
        if sid in sessions:
            raise ValueError("Duplicate session metadata")
        sessions[sid] = {"user_id": row["user ID"], "timestamp": int(row["timestamp"])}
    bundles, items, events = defaultdict(set), defaultdict(set), defaultdict(list)
    for row in csv_rows(directory / "session_bundle.csv"):
        bundles[row["session ID"]].add(row["bundle ID"])
    for row in csv_rows(directory / "bundle_item.csv"):
        items[row["bundle ID"]].add(row["item ID"])
    for row in csv_rows(directory / "user_item.csv"):
        events[row["user ID"]].append((int(row["timestamp"]), row["item ID"]))
    intents = unique_map(csv_rows(directory / "bundle_intent.csv"), "bundle ID", "intent")
    rows = []
    for sid in sorted(bundles, key=id_key):
        choices = sorted(bundles[sid], key=id_key)
        index = int.from_bytes(hashlib.sha256(f"{seed}\0{domain}\0{sid}".encode()).digest()[:8], "big") % len(choices)
        chosen, info = choices[index], sessions[sid]
        if chosen not in intents or not items[chosen] or not items[chosen] <= by_item.keys():
            raise ValueError("Incomplete source bundle")
        latest = {}
        for when, item in events[info["user_id"]]:
            if when < info["timestamp"] and item in by_item:
                latest[item] = max(when, latest.get(item, when))
        history = [{**by_item[item], "timestamp": latest[item]} for item in sorted(latest, key=lambda x: (-latest[x], id_key(x)))[:history_limit]]
        intent = " ".join(intents[chosen].split())
        rows.append({"session_id": domain + ":" + sid, "original_session_id": sid, "domain": domain, **info,
                     "input": {"intent": intent, "has_intent": bool(intent), "history": history},
                     "target": {"bundle_id": chosen, "gold_items": sorted(items[chosen], key=id_key)}})
    return catalog, split_sessions(rows, validation_fraction), {"mode": "raw_bundlerec", "files": source_hashes, "target_seed": seed}


def import_domain(source, domain, validation_fraction, history_limit=20):
    directory = Path(source) / domain
    paths = [directory / (name + ".jsonl") for name in ("catalog", "train", "calibration", "test")]
    catalog = read_jsonl(paths[0])
    splits = {name: read_jsonl(directory / (name + ".jsonl")) for name in ("train", "calibration", "test")}
    splits["train"], splits["validation"] = split_validation(splits["train"], validation_fraction)
    for rows in splits.values():
        for row in rows:
            row["input"] = input_only(row)["input"]
            row["input"]["history"] = sorted(row["input"]["history"], key=lambda event: (-event["timestamp"], id_key(event["item_id"])))[:history_limit]
    return catalog, splits, {"mode": "prepared_import", "files": {p.name: file_hash(p) for p in paths}}


def prepare(source, output, domains=DOMAINS, mode="raw", seed=42, validation_fraction=.1, history_limit=20):
    if mode not in ("raw", "import"):
        raise ValueError("Expected raw or import mode")
    if type(seed) is not int or seed < 0 or type(history_limit) is not int or history_limit < 0:
        raise ValueError("Seed and history limit must be nonnegative integers")
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("Domains must be nonempty and unique")
    protocol = {"schema_version": 1, "unit": "one_selected_bundle_per_annotated_session",
        "outer_split": "chronological_nearest_60_30_10_whole_timestamp_groups",
        "validation": {"fraction_of_original_train": validation_fraction, "policy": "latest_whole_timestamp_groups"},
        "history": "strictly_prior_only", "maximum_history_items": history_limit,
        "intent": "posthoc_annotations_as_assumed_queries", "catalog": "static_transductive_metadata",
        "patterns": "selected_targets_from_final_train_only", "calibration_labels": "RABR_calibration_only",
        "fixed_k": "final_train_median_round_half_up_clamp_2_8", "source_target_selection_seed": seed}
    staged = {}
    for domain in domains:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", domain):
            raise ValueError("Unsafe domain name")
        staged[domain] = raw_domain(source, domain, seed, validation_fraction, history_limit) if mode == "raw" else import_domain(source, domain, validation_fraction, history_limit)
        validate_dataset(staged[domain][0], staged[domain][1], domain)
    return {domain: save_dataset(output, domain, *value, protocol).manifest for domain, value in staged.items()}
