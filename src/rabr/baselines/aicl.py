import json
import os
from pathlib import Path
import numpy as np
from .common import LLMBase, context_text, input_context, parse_items, parse_object, stable_seed


class AICL(LLMBase):
    name = 'aicl'
    variant = 'train_neighbor_mutual_correction_auto_feedback'

    def fit(self, train, validation, catalog):
        super().fit(train, validation, catalog)
        if not self.train:
            raise ValueError('AICL requires training demonstrations')
        from rabr.backend import create_backend
        raters = self.config.get('intent_evaluator_backends', [])
        if len(raters) != 2:
            raise ValueError('AICL requires two distinct intent_evaluator_backends')
        names = [os.path.expanduser(os.path.expandvars(str(r.get('model_path', r.get('model_id', ''))))) for r in raters]
        names = [str(Path(name).resolve()) if Path(name).exists() else name for name in names]
        if not all(names) or len(set(names)) != 2:
            raise ValueError('AICL evaluators must identify two different models, not different seeds')
        self.evaluators = [create_backend(r, self.config.get('evaluation_profile', 'experiment'), catalog) for r in raters]
        self.evaluator_names = names
        self.fit_metadata = {'intent_evaluators': names, 'rating_scale': [3, 3, 2],
                             'task_protocol': 'query_history_adaptation', 'demonstration_labels': 'training_only'}
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
            for rater, evaluator in enumerate(self.evaluators):
                rater_scores=[]
                for repeat in range(int(self.config.get('rater_repeats', 3))):
                    prompt = [{'role': 'user', 'content': 'Compare the generated and reference intents. Score naturalness 1–3 (hard to easy to understand), product coverage 1–3 (few to most products), and motivation 1–2 (absent or present). Return JSON with generated and reference arrays of three numbers.\n' + json.dumps({'generated': intent, 'reference': input_context(row)['intent'], 'products': [self.catalog[self.index.lookup[i]]['title'] for i in items]})}]
                    rating_seed = stable_seed(seed, 'rater', step, rater, repeat)
                    rating_raw = evaluator.generate(prompt, rating_seed)
                    trace.append({'stage': 'intent_rating', 'evaluator': self.evaluator_names[rater],
                                  'seed': rating_seed, 'raw_output': rating_raw, 'cost': dict(evaluator.last_generation)})
                    value, _ = parse_object(rating_raw)
                    a, b = value.get('generated'), value.get('reference')
                    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 3 and all(type(x) in (int, float) and 1 <= x <= limit for values in (a, b) for x, limit in zip(values, (3, 3, 2))):
                        rater_scores.append(np.asarray(b) - np.asarray(a))
                if len(rater_scores) != int(self.config.get('rater_repeats', 3)):
                    raise ValueError('AICL evaluator returned an invalid rating; cannot silently skip feedback')
                scores.append(np.mean(rater_scores,axis=0))
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
        return self.sample_predictions(self.prompt(context, k, 'Neighbor demonstrations and rules:\n' + json.dumps(demonstrations, ensure_ascii=False)), k, seed, trace, 'demonstration_guided_inference', {'neighbor_session_ids': [self.train[int(i)]['session_id'] for i in neighbors], 'intent_evaluators': self.evaluator_names})
