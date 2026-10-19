"""Category-pattern prompting and teacher/student knowledge distillation."""
from collections import Counter
from itertools import combinations
import json
import hashlib
from pathlib import Path
import numpy as np
from .llm_common import LLMBase, context_text, input_context, parse_object, stable_seed


class KD4BG(LLMBase):
    name = 'kd4bg'

    def fit(self, train, validation, catalog):
        super().fit(train, validation, catalog)
        self.mode = self.config.get('mode', 'pattern_icl')
        if self.mode not in {'pattern_icl', 'teacher_student'}:
            raise ValueError('KD4BG mode must be pattern_icl or teacher_student')
        self.variant = 'category_pattern_icl' if self.mode == 'pattern_icl' else 'teacher_rules_lora_student'
        counts = Counter()
        for row in self.train:
            categories = sorted(str(self.catalog[self.index.lookup[i]].get('category') or 'Uncategorized') for i in row['target']['gold_items'])
            patterns = {tuple(c) for length in range(2, min(4,len(categories))+1) for c in combinations(categories,length)}
            counts.update(patterns)
        self.patterns = [{'categories':list(p),'support':count} for p,count in sorted(counts.items()) if count>=int(self.config.get('minimum_support',2))]
        self.pattern_vectors = self.index.encode(['Categories: '+', '.join(p['categories']) for p in self.patterns]) if self.patterns else None
        self.rules = [];self.rules_vectors = None
        if self.mode == 'teacher_student':
            if not self.train or not self.validation:raise ValueError('distillation requires train and validation rows')
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
                prompt = [{'role':'user','content':'Extract reusable category patterns and concise complementary-product rules from this labeled training example. Return JSON with a rules list.\n'+json.dumps({'input':input_context(row),'bundle':gold},ensure_ascii=False)}]
                raw = self.ask(prompt,stable_seed(int(self.config.get('training_seed',self.config.get('seed',42))),row['session_id'],'teacher'),trace,'teacher_knowledge')
                value,error = parse_object(raw)
                if error or not isinstance(value.get('rules'),list) or any(not isinstance(x,str) for x in value['rules']):
                    raise ValueError('teacher output must contain a rules list')
                self.rules.append({'session_id':row['session_id'],'context':context_text(row),'rules':value['rules']})
                temporary=cache.with_suffix('.tmp');temporary.write_text(json.dumps({'identity':identity,'rules':self.rules,'trace':trace},ensure_ascii=False,indent=2)+'\n');temporary.replace(cache)
            self.rules_vectors = self.index.encode([r['context'] for r in self.rules])
            records = []
            for split in (self.train,self.validation):
                records.append([{'messages':self.prompt(row,len(row['target']['gold_items']),self.guidance(row)),
                                 'response':json.dumps({'items':[self.catalog[self.index.lookup[i]]['title'] for i in row['target']['gold_items']]})} for row in split])
            from .llm_distill import train_student
            self.backend, training = train_student(records[0],records[1],self.config,self.backend)
            self.fit_metadata = {'teacher_calls':len(trace),'knowledge':self.rules,'student_training':training}
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
            result += '\nTeacher-derived training rules: '+json.dumps([self.rules[int(i)]['rules'] for i in indices],ensure_ascii=False)
        return result

    def predict(self, context, k, seed):
        if k<1:raise ValueError('k must be positive')
        trace=[]
        raw=self.ask(self.prompt(context,k,self.guidance(context)),seed,trace,'student_generation' if self.mode=='teacher_student' else 'pattern_icl')
        return self.finish(raw,k,trace)
