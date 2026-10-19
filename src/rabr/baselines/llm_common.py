"""Shared input, parsing, retrieval and grounding operations."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from rabr.types import Prediction


def input_context(context):
    source = context.get('input', context)
    history = [{'item_id': str(item['item_id']), 'title': item.get('title', ''),
                'category': item.get('category')} for item in source.get('history', [])]
    return {'intent': str(source.get('intent', '')), 'history': history}


def context_text(context):
    data = input_context(context)
    return 'Intent: ' + data['intent'] + '\nPrior interactions: ' + json.dumps(data['history'], ensure_ascii=False)


def stable_seed(seed, *parts):
    value = json.dumps([seed, *parts], sort_keys=True, ensure_ascii=False).encode()
    return int(hashlib.sha256(value).hexdigest()[:8], 16) % (2**31)


def parse_object(raw):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value, None
    except (ValueError, TypeError) as error:
        return {}, str(error)


def parse_items(raw):
    value, error = parse_object(raw)
    items = value.get('items')
    if error or not isinstance(items, list) or any(not isinstance(x, str) or not x.strip() for x in items):
        return [], error or 'items must be nonempty strings'
    return [x.strip() for x in items], None


class CatalogIndex:
    def __init__(self, catalog, backend, tau=0.0):
        self.catalog = list(catalog)
        self.backend, self.tau = backend, float(tau)
        self.ids = [str(row['item_id']) for row in catalog]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('catalog IDs must be nonempty and unique')
        self.lookup = {item: i for i, item in enumerate(self.ids)}
        self.vectors = self.encode([row['title'] for row in catalog])

    def encode(self, texts):
        values = np.asarray(self.backend.encode(texts), dtype=np.float32)
        if values.ndim != 2 or len(values) != len(texts) or not np.isfinite(values).all():
            raise ValueError('encoder must return finite text vectors')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError('encoder returned a zero vector')
        return values / norms

    def retrieve(self, query, count):
        scores = self.encode([query])[0] @ self.vectors.T
        return np.argsort(-scores, kind='stable')[:min(count, len(self.ids))].tolist()

    def ground(self, descriptions, k):
        if not descriptions:
            return []
        scores = self.encode(descriptions) @ self.vectors.T
        means = {}
        for row in scores:
            best = int(np.argmax(row))
            if row[best] >= self.tau:
                means.setdefault(best, []).append(float(row[best]))
        ranked = sorted(means, key=lambda i: (-float(np.mean(means[i])), i))
        return [self.ids[i] for i in ranked[:k]]


class LLMBase:
    name = 'zero_shot'
    variant = 'direct_generation'

    def __init__(self, config, backend):
        self.config, self.backend = dict(config), backend
        self.fit_metadata = {}

    def fit(self, train, validation, catalog):
        train, validation = list(train), list(validation)
        if {r['session_id'] for r in train} & {r['session_id'] for r in validation}:
            raise ValueError('training and validation sessions overlap')
        self.train, self.validation = copy.deepcopy(train), copy.deepcopy(validation)
        self.index = CatalogIndex(catalog, self.backend, self.config.get('grounding_tau', 0.0))
        self.catalog = list(catalog)
        return self

    def ask(self, messages, seed, trace, stage):
        raw = self.backend.generate(messages, seed=seed)
        if not isinstance(raw, str):
            raise TypeError('backend.generate must return text')
        trace.append({'stage': stage, 'seed': seed, 'raw_output': raw})
        return raw

    def prompt(self, context, k, guidance=''):
        return [{'role': 'system', 'content': 'Recommend complementary catalog-style product descriptions. Return only JSON with an items list of product-description strings.'},
                {'role': 'user', 'content': context_text(context) + f'\nRecommend up to {k} products.\n' + guidance}]

    def finish(self, raw, k, trace, extra=None, allowed_ids=None):
        descriptions, error = parse_items(raw)
        if allowed_ids is None:
            items = self.index.ground(descriptions, k)
        else:
            allowed = set(allowed_ids)
            items = list(dict.fromkeys(x for x in descriptions if x in allowed))[:k]
            if any(x not in allowed for x in descriptions):
                error = error or 'output included an item outside the candidate set'
        return Prediction(items=items, metadata={'method': self.name, 'implementation_variant': self.variant,
                          'llm_calls': len(trace), 'parse_error': error, 'trace': trace,
                          'fit': self.fit_metadata, **(extra or {})})

    def predict(self, context, k, seed):
        if k < 1:
            raise ValueError('k must be positive')
        trace = []
        raw = self.ask(self.prompt(context, k), seed, trace, 'generate')
        return self.finish(raw, k, trace)
