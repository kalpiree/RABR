import json
from .common import LLMBase, context_text, input_context


class LLM4B(LLMBase):
    name = 'llm4b'
    variant = 'text2bundle_section5_2_top30_retrieve_select'

    def predict(self, context, k, seed):
        if k < 1:
            raise ValueError('k must be positive')
        indices = self.index.retrieve(input_context(context)['intent'], int(self.config.get('candidate_count', 30)))
        candidates = [{'item_id': self.index.ids[i], 'title': self.catalog[i]['title']} for i in indices]
        prompt = [{'role': 'system', 'content': 'Select complementary products from the supplied candidates. Return only JSON with an items list of exact item_id strings.'},
                  {'role': 'user', 'content': context_text(context) + f'\nSelect up to {k} items matching the query and preferences.\nCandidates: ' + json.dumps(candidates, ensure_ascii=False)}]
        trace = []
        return self.sample_predictions(prompt, k, seed, trace, 'retrieve_select', {'candidate_ids': [x['item_id'] for x in candidates]}, [x['item_id'] for x in candidates])
