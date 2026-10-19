"""Generative bundle recommender factory."""
from .llm_common import LLMBase


def create_generative(name, config, backend):
    if name == 'zero_shot':
        return LLMBase(config, backend)
    if name == 'llm4b':
        from .llm_retrieval import LLM4B
        return LLM4B(config, backend)
    if name == 'aicl':
        from .llm_aicl import AICL
        return AICL(config, backend)
    if name == 'kd4bg':
        from .llm_kd4bg import KD4BG
        return KD4BG(config, backend)
    if name == 'text2bundle':
        from .llm_text2bundle import Text2Bundle
        return Text2Bundle(config, backend)
    if name == 'ddbc':
        from .ddbc import DDBC
        return DDBC(config, backend)
    raise ValueError('Unknown generative method: ' + str(name))
