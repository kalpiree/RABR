from collections import Counter
from itertools import combinations
import json
import hashlib
from pathlib import Path
import numpy as np
from .common import LLMBase, context_text, input_context, parse_object, stable_seed


class KD4BG(LLMBase):
    name = 'kd4bg'

    def fit(self, train, validation, catalog):
        super().fit(train, validation, catalog)
        self.mode = self.config.get('mode', 'pattern_icl')
        if self.mode not in {'pattern_icl', 'teacher_student'}:
            raise ValueError('KD4BG mode must be pattern_icl or teacher_student')
        self.variant = 'category_pattern_icl' if self.mode == 'pattern_icl' else 'teacher_rules_lora_student'
        transactions = [frozenset(str(self.catalog[self.index.lookup[i]].get('category'))
                        for i in row['target']['gold_items'] if self.catalog[self.index.lookup[i]].get('category'))
                        for row in self.train]
        support = int(self.config.get('minimum_support', 2))
        candidates = {frozenset([item]) for transaction in transactions for item in transaction}
        patterns = []
        length = 1
        while candidates:
            frequent = {candidate: sum(candidate <= transaction for transaction in transactions)
                        for candidate in candidates}
            frequent = {candidate: count for candidate, count in frequent.items() if count >= support}
            if length >= 2:
                patterns.extend({'categories': sorted(pattern), 'support': count} for pattern, count in frequent.items())
            length += 1
            previous = set(frequent)
            candidates = {left | right for left in previous for right in previous
                          if len(left | right) == length and all(frozenset(subset) in previous
                             for subset in combinations(left | right, length - 1))}
        self.patterns = sorted(patterns, key=lambda value: (-value['support'], value['categories']))
        self.pattern_vectors = self.index.encode(['Categories: '+', '.join(p['categories']) for p in self.patterns]) if self.patterns else None
        self.rules = [];self.rules_vectors = None
        if self.mode == 'teacher_student':
            if not self.train or not self.validation:raise ValueError('distillation requires train and validation rows')
            from rabr.backend import create_backend
            teacher_config = self.config.get('teacher_backend')
            if not teacher_config:
                raise ValueError('KD4BG teacher_student requires an explicitly identified teacher_backend')
            self.teacher = create_backend(teacher_config, self.config.get('evaluation_profile', 'experiment'), catalog)
            trace = []
            directory=Path(self.config.get('work_dir',self.config.get('artifact_dir','runs/kd4bg-student')));directory.mkdir(parents=True,exist_ok=True)
            cache=directory/'teacher-knowledge.json'
            identity=hashlib.sha256(json.dumps({'train':self.train,'catalog':self.catalog,'config':self.config},sort_keys=True).encode()).hexdigest()
            if cache.exists():
                cached=json.loads(cache.read_text())
                if cached['identity']!=identity:raise ValueError('KD4BG teacher cache input/config mismatch')
                self.rules=cached['rules'];trace=cached['trace']
            for row in self.train:
                if any(rule['session_id']==row['session_id'] for rule in self.rules):continue
                gold = [{'title':self.catalog[self.index.lookup[i]]['title'],'category':self.catalog[self.index.lookup[i]].get('category')} for i in row['target']['gold_items']]
                training_seed = stable_seed(int(self.config.get('training_seed', self.config.get('seed', 42))), row['session_id'])
                messages = self.prompt(row, len(gold))
                def teacher_call(stage, prompt):
                    call_seed = stable_seed(training_seed, stage)
                    raw = self.teacher.generate(prompt, call_seed)
                    trace.append({'session_id': row['session_id'], 'stage': stage, 'seed': call_seed,
                                  'raw_output': raw, 'cost': dict(self.teacher.last_generation)})
                    return raw
                initial = teacher_call('initial_prediction', messages)
                messages += [{'role': 'assistant', 'content': initial}, {'role': 'user', 'content':
                    'Compare your proposed bundle against this labeled training bundle. Explain each mismatch using product purposes and relationships. Return JSON with an errors object. Training bundle: '+json.dumps(gold, ensure_ascii=False)}]
                errors = teacher_call('compare_training_labels', messages)
                messages += [{'role': 'assistant', 'content': errors}, {'role': 'user', 'content':
                    'Review the causes of those mistakes: inferred intent, complementary or substitute relationships, and coherent usage. Return JSON with a review object.'}]
                review = teacher_call('self_reflection', messages)
                messages += [{'role': 'assistant', 'content': review}, {'role': 'user', 'content':
                    'Derive reusable criteria for grouping products and avoiding the identified mistakes. Return JSON with a rules list of strings.'}]
                rule_raw = teacher_call('formalized_rules', messages)
                value, error = parse_object(rule_raw)
                if error or not isinstance(value.get('rules'), list) or any(not isinstance(x, str) for x in value['rules']):
                    raise ValueError('Teacher rules must be a JSON string list')
                messages += [{'role': 'assistant', 'content': rule_raw}, {'role': 'user', 'content':
                    'Explain how to apply these rules to this training example, connecting user motivation, product functions and tradeoffs. Return JSON with a thoughts list of strings.'}]
                thought_raw = teacher_call('deep_thoughts', messages)
                thoughts, error = parse_object(thought_raw)
                if error or not isinstance(thoughts.get('thoughts'), list) or any(not isinstance(x, str) for x in thoughts['thoughts']):
                    raise ValueError('Teacher thoughts must be a JSON string list')
                self.rules.append({'session_id': row['session_id'], 'context': context_text(row),
                                   'rules': value['rules'], 'thoughts': thoughts['thoughts']})
                temporary=cache.with_suffix('.tmp');temporary.write_text(json.dumps({'identity':identity,'rules':self.rules,'trace':trace},ensure_ascii=False,indent=2)+'\n');temporary.replace(cache)
            self.rules_vectors = self.index.encode([r['context'] for r in self.rules])
            records = []
            for split in (self.train,self.validation):
                records.append([{'messages':self.prompt(row,len(row['target']['gold_items']),self.guidance(row)),
                                 'response':json.dumps({'items':[self.catalog[self.index.lookup[i]]['title'] for i in row['target']['gold_items']]})} for row in split])
            del self.teacher
            from .finetune import train_student
            self.backend, training = train_student(records[0],records[1],self.config,self.backend)
            self.fit_metadata = {'teacher_calls':len(trace),'teacher_backend': teacher_config, 'teacher_trace': trace, 'knowledge':self.rules,'student_training':training, 'task_protocol': 'query_history_adaptation', 'knowledge_labels': 'training_only'}
        return self

    def guidance(self, context):
        vector = self.index.encode([context_text(context)])[0]
        selected = []
        if self.pattern_vectors is not None:
            scores = vector @ self.pattern_vectors.T
            indices = sorted(range(len(scores)),key=lambda i:(-float(scores[i]),-self.patterns[i]['support'],self.patterns[i]['categories']))[:int(self.config.get('pattern_count',8))]
            selected = [self.patterns[i] for i in indices]
        result = 'Training category patterns: '+json.dumps(selected,ensure_ascii=False)
        if self.rules_vectors is not None:
            indices = np.argsort(-(vector@self.rules_vectors.T),kind='stable')[:int(self.config.get('rule_neighbors',1))]
            result += '\nTeacher-derived training rules: '+json.dumps([{key: self.rules[int(i)][key] for key in ('rules', 'thoughts')} for i in indices],ensure_ascii=False)
        return result

    def predict(self, context, k, seed):
        if k<1:raise ValueError('k must be positive')
        trace=[]
        return self.sample_predictions(self.prompt(context,k,self.guidance(context)),k,seed,trace,'student_generation' if self.mode=='teacher_student' else 'pattern_icl')
