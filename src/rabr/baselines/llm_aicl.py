"""Neighbor demonstrations with mutual correction and supervised feedback."""
import json
import numpy as np
from .llm_common import LLMBase, context_text, input_context, parse_items, parse_object, stable_seed


class AICL(LLMBase):
    name = 'aicl'
    variant = 'train_neighbor_mutual_correction_auto_feedback'

    def fit(self, train, validation, catalog):
        super().fit(train, validation, catalog)
        if not self.train:
            raise ValueError('AICL requires training demonstrations')
        self.retrieval_vectors = self.index.encode([context_text(r) for r in self.train])
        self.demonstrations = {}
        return self

    def _intent(self, items, context, seed, trace):
        titles = [self.catalog[self.index.lookup[i]]['title'] for i in items]
        prompt = [{'role': 'user', 'content': 'Describe the purpose of these products in 3 to 5 words. Return JSON with an intent string.\nProducts: ' + json.dumps(titles) + '\nContext: ' + context_text(context)}]
        raw = self.ask(prompt, seed, trace, 'infer_intent')
        value, _ = parse_object(raw)
        return str(value.get('intent', ''))

    def _demonstrate(self, row, seed):
        key = (row['session_id'], seed)
        if key in self.demonstrations:
            return self.demonstrations[key], []
        trace, messages = [], self.prompt(row, len(row['target']['gold_items']))
        raw = self.ask(messages, stable_seed(seed, 'initial'), trace, 'neighbor_generation')
        descriptions, _ = parse_items(raw)
        items = self.index.ground(descriptions, 8)
        intent = self._intent(items, row, stable_seed(seed, 'intent'), trace)
        for step in range(int(self.config.get('self_correction_rounds', 1))):
            previous = items
            messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': 'Refine the bundle using this inferred intent: ' + intent + '. Return JSON with items.'}]
            raw = self.ask(messages, stable_seed(seed, 'self', step), trace, 'mutual_correction')
            descriptions, _ = parse_items(raw)
            items = self.index.ground(descriptions, 8)
            if items == previous:
                break
            intent = self._intent(items, row, stable_seed(seed, 'reintent', step), trace)
        gold = set(row['target']['gold_items'])
        for step in range(int(self.config.get('bundle_feedback_rounds', 3))):
            predicted = set(items)
            if predicted == gold:
                break
            titles=lambda values:[self.catalog[self.index.lookup[i]]['title'] for i in sorted(values)]
            feedback = {'keep': titles(predicted & gold), 'remove_unrelated': titles(predicted - gold), 'missing_count': len(gold - predicted)}
            messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': 'Training feedback: ' + json.dumps(feedback) + '. Correct the bundle. Return JSON with items.'}]
            raw = self.ask(messages, stable_seed(seed, 'feedback', step), trace, 'bundle_auto_feedback')
            descriptions, _ = parse_items(raw)
            items = self.index.ground(descriptions, 8)
        intent = self._intent(items, row, stable_seed(seed, 'final_intent'), trace)
        for step in range(int(self.config.get('intent_feedback_rounds', 1))):
            scores = []
            for rater in range(int(self.config.get('intent_raters', 2))):
                rater_scores=[]
                for repeat in range(int(self.config.get('rater_repeats', 3))):
                    prompt = [{'role': 'user', 'content': 'Rate the generated and reference intents on naturalness, product coverage, and motivation from 1 to 5. Return JSON with generated and reference arrays of three numbers.\n' + json.dumps({'generated': intent, 'reference': input_context(row)['intent'], 'products': [self.catalog[self.index.lookup[i]]['title'] for i in items]})}]
                    value, _ = parse_object(self.ask(prompt, stable_seed(seed, 'rater', step, rater, repeat), trace, 'intent_rating'))
                    a, b = value.get('generated'), value.get('reference')
                    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 3 and all(type(x) in (int, float) and 1 <= x <= 5 for x in a + b):
                        rater_scores.append(np.asarray(b) - np.asarray(a))
                if rater_scores:scores.append(np.mean(rater_scores,axis=0))
            if not scores or not (np.max(scores, axis=0) > 0).any():
                break
            aspects = [name for name, difference in zip(('naturalness', 'coverage', 'motivation'), np.max(scores, axis=0)) if difference > 0]
            prompt = [{'role': 'user', 'content': 'Revise this training intent to improve ' + ', '.join(aspects) + '. Return JSON with intent.\n' + json.dumps({'intent': intent, 'products': items})}]
            value, _ = parse_object(self.ask(prompt, stable_seed(seed, 'intent_feedback', step), trace, 'intent_auto_feedback'))
            intent = str(value.get('intent', intent))
        prompt = [{'role': 'user', 'content': 'Summarize reusable product-bundling rules from this training example and its correction history. Do not mention session IDs.\n' + json.dumps({'context': input_context(row), 'gold_titles': [self.catalog[self.index.lookup[i]]['title'] for i in row['target']['gold_items']], 'trace': trace}, ensure_ascii=False)}]
        rules = self.ask(prompt, stable_seed(seed, 'summary'), trace, 'rule_summary')
        demo = {'context': input_context(row), 'bundle': [self.catalog[self.index.lookup[i]]['title'] for i in row['target']['gold_items']], 'intent': intent, 'rules': rules}
        self.demonstrations[key] = demo
        return demo, trace

    def predict(self, context, k, seed):
        if k < 1:
            raise ValueError('k must be positive')
        scores = self.index.encode([context_text(context)])[0] @ self.retrieval_vectors.T
        neighbors = np.argsort(-scores, kind='stable')[:int(self.config.get('neighbors', 1))]
        demonstrations, trace = [], []
        for index in neighbors:
            row = self.train[int(index)]
            demo, calls = self._demonstrate(row, stable_seed(int(self.config.get('demonstration_seed', 42)), row['session_id']))
            demonstrations.append(demo)
            trace += calls
        raw = self.ask(self.prompt(context, k, 'Neighbor demonstrations and rules:\n' + json.dumps(demonstrations, ensure_ascii=False)), seed, trace, 'demonstration_guided_inference')
        return self.finish(raw, k, trace, {'neighbor_session_ids': [self.train[int(i)]['session_id'] for i in neighbors], 'intent_rater_variant': 'same_backbone_independent_seeds'})
