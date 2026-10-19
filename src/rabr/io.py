import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{index}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected JSONL objects")
            rows.append(row)
    return rows


class Ledger:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.rows = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.splitlines(keepends=True)
            valid_bytes = 0
            for index, line in enumerate(lines):
                if not line.endswith(b"\n") and index == len(lines) - 1:
                    with self.path.open("r+b") as stream:
                        stream.truncate(valid_bytes)
                    break
                row = json.loads(line)
                if row["identity"] != identity or row["key"] in self.rows:
                    raise ValueError("Checkpoint identity or key mismatch")
                if row["sha256"] != digest(row["value"]):
                    raise ValueError("Checkpoint payload hash mismatch")
                self.rows[row["key"]] = row["value"]
                valid_bytes += len(line)

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, value):
        if key in self.rows:
            if self.rows[key] != value:
                raise ValueError("Conflicting checkpoint key")
            return
        row = {"identity": self.identity, "key": key, "value": value, "sha256": digest(value)}
        with self.path.open("ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[key] = value


def stable_seed(seed, *parts):
    return int(digest([int(seed), *parts])[:15], 16) % (2**31 - 1)
