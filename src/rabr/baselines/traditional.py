from __future__ import annotations


def create_traditional(name: str, config: dict):
    name = name.lower()
    if name not in {"pog", "bgn", "bunt", "byob"}:
        raise ValueError(f"Unknown traditional method: {name}")
    try:
        if name == "bunt":
            from .bunt import BUNT
            return BUNT(dict(config))
        from .pog import POG
        from .bgn import BGN
        from .byob import BYOB
    except ImportError as error:
        if error.name == "torch":
            raise ImportError("Traditional methods require the 'models' optional dependencies (PyTorch).") from error
        raise
    return {"pog": POG, "bgn": BGN, "byob": BYOB}[name](dict(config))
