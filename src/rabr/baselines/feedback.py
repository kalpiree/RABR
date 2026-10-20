from __future__ import annotations


class TargetFeedback:
    def __init__(self, target_items, metadata, *, protocol):
        if protocol not in {"training_simulation", "validation_simulation", "heldout_evaluation"}:
            raise ValueError("A feedback simulator requires an explicit protocol")
        self._targets = frozenset(target_items)
        if not self._targets or not self._targets <= metadata.keys():
            raise ValueError("Feedback targets must belong to the metadata catalog")
        self._metadata, self._accepted, self._slots = metadata, set(), {}
        self.protocol = protocol

    def __call__(self, request):
        feedback = []
        remaining = set(self._targets) - self._accepted
        if request["action"] == "recommend":
            for proposal in request["proposals"]:
                item = proposal["item_id"]
                accepted = item in remaining
                if accepted:
                    self._accepted.add(item)
                    remaining.discard(item)
                feedback.append({**proposal, "decision": "accept" if accepted else "reject"})
        elif request["action"] == "ask":
            for query in request["queries"]:
                slot, kind, tag = query["slot"], query["kind"], query["tag"]
                key = "categories" if kind == "category" else "attributes"
                possible = self._slots.get(slot, set(remaining)) & remaining
                matches = {item for item in possible if tag in self._metadata[item][key]}
                if matches:
                    self._slots[slot] = matches
                    decision = "accept"
                elif all(tag not in self._metadata[item][key] for item in remaining):
                    decision = "reject"
                else:
                    decision = "ignore"
                feedback.append({**query, "decision": decision})
        else:
            raise ValueError("Unknown feedback request")
        return {"feedback": feedback, "done": self._accepted == self._targets}
