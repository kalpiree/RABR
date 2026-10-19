"""Factories for trainable traditional bundle-generation methods."""
from __future__ import annotations


def create_traditional(name: str, config: dict):
    name = name.lower()
    if name not in {"pog", "bgn", "bunt", "byob"}:
        raise ValueError(f"Unknown traditional method: {name}")
    try:
        from .traditional_models import POG, BGN
        from .traditional_policies import BUNT, BYOB
    except ImportError as error:
        if error.name == "torch":
            raise ImportError("Traditional methods require the 'traditional' optional dependencies (PyTorch).") from error
        raise
    return {"pog": POG, "bgn": BGN, "bunt": BUNT, "byob": BYOB}[name](dict(config))
