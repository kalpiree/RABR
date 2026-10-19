"""Load audited BYOB model definitions from an immutable external checkout."""
from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .traditional_assets import MissingMethodAsset

BYOB_COMMIT = "538fb8de57fbc4f92dfb456b261a5150aab27121"


def load_byob(config):
    value = os.path.expandvars(os.path.expanduser(str(config.get("byob_source_path", ""))))
    if not value or "$" in value:
        raise MissingMethodAsset("Set byob_source_path to https://github.com/fuxiAIlab/BYOB at " + BYOB_COMMIT)
    root = Path(value).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != BYOB_COMMIT:
        raise ValueError("BYOB source checkout does not match the audited commit")
    entries = {
        "byob/models/policy_net.py": {"ItemPointer"},
        "byob/models/baseline_ncf.py": {"mlp", "ItemNCFModel"},
        "byob/models/item2vec.py": {"SkipGramModel"},
    }
    definitions, hashes = {}, {"commit": actual}
    for relative, names in entries.items():
        path = root / relative
        source = path.read_bytes()
        expected = subprocess.check_output(["git", "-C", str(root), "show", f"{BYOB_COMMIT}:{relative}"])
        if source != expected:
            raise ValueError("Modified upstream BYOB source: " + relative)
        tree = ast.parse(source, filename=str(path))
        body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in body} != names:
            raise ValueError("Missing audited upstream model definition")
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        namespace = {"__name__": "rabr_external_byob", "np": np, "torch": torch, "nn": nn, "F": F, "math": math,
                     "TransformerEncoder": nn.TransformerEncoder, "TransformerEncoderLayer": nn.TransformerEncoderLayer}
        exec(compile(module, str(path), "exec"), namespace)
        definitions.update({name: namespace[name] for name in names})
        hashes[relative] = hashlib.sha256(source).hexdigest()
    return definitions, hashes
