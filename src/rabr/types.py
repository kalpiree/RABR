from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class Prediction:
    items: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class Predictor(Protocol):
    def fit(self, train: Sequence[dict], validation: Sequence[dict], catalog: Sequence[dict]) -> None: ...
    def predict(self, context: dict, k: int, seed: int) -> Prediction: ...


def input_only(row: dict) -> dict:
    source = row["input"]
    history = [{key: item[key] for key in ("item_id", "title", "category", "timestamp") if key in item}
               for item in source.get("history", [])]
    result = {"session_id": str(row["session_id"]), "timestamp": row.get("timestamp"),
              "input": {"intent": str(source.get("intent", "")), "history": history}}
    if "user_id" in row:
        result["user_id"] = str(row["user_id"])
    return result


def gold_items(row: dict) -> list[str]:
    values = row.get("target", {}).get("gold_items", row.get("target_items"))
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("Expected nonempty string target item IDs")
    return list(dict.fromkeys(values))
