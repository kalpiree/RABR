from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np


class FixtureBackend:
    def __init__(self, config, catalog):
        self.config, self.catalog = dict(config), list(catalog)
        self.stats = {"generation_calls": 0, "input_tokens": 0, "output_tokens": 0, "generation_seconds": 0.0}
        self.last_generation = {}

    def encode(self, texts):
        result = np.zeros((len(texts), 128), dtype=np.float32)
        for index, text in enumerate(texts):
            for word in re.findall(r"[\w]+", str(text).lower()) or ["empty"]:
                value = hashlib.sha256(word.encode()).digest()
                result[index, int.from_bytes(value[:4], "big") % 128] += 1
        return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)

    def generate(self, messages, seed):
        text = "\n".join(str(message["content"]) for message in messages)
        scores = self.encode([text])[0] @ self.encode([x["title"] for x in self.catalog]).T
        rng = np.random.default_rng(seed)
        indices = np.argsort(-(scores + rng.uniform(0, .02, len(scores))), kind="stable")[:3]
        reply = {"items": [self.catalog[i]["title"] for i in indices]}
        if "Candidate products:" in text:
            suffix = text.split("Candidate products:", 1)[1].lstrip()
            try:
                candidates, _ = json.JSONDecoder().raw_decode(suffix)
                reply = {"items": [str(item["item_id"]) for item in candidates[:3]]}
            except (ValueError, KeyError, TypeError):
                pass
        lowered = text.lower()
        if '"interests"' in lowered or "interests list" in lowered:
            reply = {"interests": ["complementary music products"]}
        elif '"rules"' in lowered or "rules list" in lowered:
            reply = {"rules": ["Prefer complementary items that address the supplied intent."]}
        elif '"rating"' in lowered:
            reply = {"rating": 0.5}
        elif '"intent"' in lowered and ("infer" in lowered or "rewrite" in lowered):
            reply = {"intent": "complementary music products"}
        raw = json.dumps(reply)
        self.last_generation = {"input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "synthetic": True}
        self.stats["generation_calls"] += 1
        return raw


class TransformersBackend:
    def __init__(self, config):
        self.config = dict(config)
        self.encoder = self.encoder_tokenizer = self.model = self.tokenizer = None
        self.stats = {"generation_calls": 0, "input_tokens": 0, "output_tokens": 0, "generation_seconds": 0.0}
        self.last_generation = {}

    def _path(self, key):
        value = os.path.expanduser(os.path.expandvars(self.config.get(key, "")))
        if not value or "$" in value:
            raise ValueError(f"Set backend.{key} to a local model path or permitted model ID")
        return value

    def _options(self, kind):
        options = {"local_files_only": not self.config.get("allow_download", False), "trust_remote_code": False}
        revision = self.config.get(kind + "_revision")
        if revision:
            options["revision"] = revision
        return options

    def encode(self, texts):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self.encoder is None:
            path = self._path("encoder_path")
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(path, **self._options("encoder"))
            self.encoder = AutoModel.from_pretrained(path, **self._options("encoder"))
            self.encoder.to(self.config.get("encoder_device", self.config.get("device", "cpu"))).eval()
        values = []
        batch_size = int(self.config.get("encoder_batch_size", 64))
        if batch_size < 1:
            raise ValueError("encoder_batch_size must be positive")
        for start in range(0, len(texts), batch_size):
            tokens = self.encoder_tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True,
                                            max_length=int(self.config.get("encoder_max_length", 256)), return_tensors="pt")
            tokens = {k: v.to(next(self.encoder.parameters()).device) for k, v in tokens.items()}
            with torch.inference_mode():
                output = self.encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
                values.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
        return np.concatenate(values) if values else np.empty((0, self.encoder.config.hidden_size), dtype=np.float32)

    def _load_generator(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is not None:
            return
        path = self._path("model_path")
        dtype_name = self.config.get("dtype", "float32")
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(path, **self._options("model"))
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype_name), **self._options("model"))
        self.model.to(self.config.get("device", "cpu")).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages, seed):
        import torch
        self._load_generator()
        if not self.tokenizer.chat_template:
            raise ValueError("The generation model needs a chat template")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        max_new = int(self.config.get("max_new_tokens", 256))
        limit = int(self.config.get("max_context_tokens", getattr(self.model.config, "max_position_embeddings", 4096)))
        if tokens["input_ids"].shape[1] + max_new > limit:
            raise ValueError("Prompt exceeds the configured context window")
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        temperature = float(self.config.get("temperature", .7))
        options = {"do_sample": temperature > 0, "max_new_tokens": max_new, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            options["temperature"] = temperature
            options["top_p"] = float(self.config.get("top_p", 1.0))
            options["top_k"] = int(self.config.get("top_k", 50))
        started = time.monotonic()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(int(seed))
            output = self.model.generate(**tokens, **options)
        n_input = tokens["input_ids"].shape[1]
        generated = output[0, n_input:]
        self.last_generation = {"input_tokens": n_input, "output_tokens": len(generated), "seconds": time.monotonic() - started}
        self.stats["generation_calls"] += 1
        for name in ("input_tokens", "output_tokens"):
            self.stats[name] += self.last_generation[name]
        self.stats["generation_seconds"] += self.last_generation["seconds"]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def create_backend(config, profile, catalog):
    kind = config.get("type")
    if profile not in {"smoke", "experiment"}:
        raise ValueError("Unknown run profile")
    if kind == "fixture":
        if profile != "smoke":
            raise ValueError("Fixture generation cannot be used for experiments")
        return FixtureBackend(config, catalog)
    if kind == "transformers":
        return TransformersBackend(config)
    raise ValueError("Unknown backend type")
