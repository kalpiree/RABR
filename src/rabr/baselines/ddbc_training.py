"""DDBC optimization, validation and resumable Torch state."""
import collections
import math
import random
import time
import os

def cpu_limit(requested):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(int(requested), available))

def example_batches(rows, batch_size):
    """Deterministic length groups, no padding and no discarded examples."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[len(row["item_indices"])].append(row)
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]

def draw_training_batch(rows, batch_size, rng):
    """A uniform example chooses its length; conditional draws are uniform.

    Thus every example has identical marginal sampling probability even when
    length frequencies are very different. Adjacent swaps preserve membership.
    """
    length = len(rows[rng.randrange(len(rows))]["item_indices"])
    group = [row for row in rows if len(row["item_indices"]) == length]
    result = []
    for _ in range(batch_size):
        indices = list(group[rng.randrange(len(group))]["item_indices"])
        for _ in range(int(0.4 * length)):
            index = rng.randrange(length - 1)
            indices[index], indices[index + 1] = indices[index + 1], indices[index]
        result.append(indices)
    return result

def clean_sequences(item_rows, catalog_codes, tokens):
    import numpy as np
    sequences, masks = [], []
    for items in item_rows:
        sequence, mask = [tokens["bos_id"]], [False]
        for index in items:
            sequence += [tokens["boi_id"]] + [int(x) for x in catalog_codes[index]]
            mask += [False] + [True] * 4
        sequences.append(sequence + [tokens["eos_id"]])
        masks.append(mask + [False])
    if len({len(x) for x in sequences}) != 1:
        raise ValueError("Training batch must have a single bundle length")
    return np.asarray(sequences, dtype=np.int64), np.asarray(masks, dtype=bool)

class TorchEngine:
    def __init__(self, args, prepared, codes, weights):
        import numpy as np
        import torch
        from . import ddbc_model
        self.np, self.torch, self.module = np, torch, ddbc_model
        self.args, self.tokens, self.codes = args, prepared["tokens"], codes
        torch.set_num_threads(cpu_limit(args.cpu_threads))
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Declared CUDA training device is unavailable")
        self.device = torch.device(args.device)
        self.model = ddbc_model.DDBCModel(vocab_size=self.tokens["vocab_size"],
            codebook_weights=torch.as_tensor(weights, dtype=torch.float32), hidden_size=64,
            n_blocks=6, n_heads=8, cond_dim=128, dropout=0.1, time_conditioning=False).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.ema = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed + 1009)
        self.batch_rng = random.Random(args.seed + 2003)
        self.cost = {"train_steps": 0, "train_examples": 0, "train_code_tokens": 0,
                     "train_forward_calls": 0, "train_backward_calls": 0, "train_seconds": 0.0,
                     "validation_passes": 0, "validation_examples": 0, "validation_forward_calls": 0,
                     "validation_seconds": 0.0}

    def _tensors(self, rows):
        ids, mask = clean_sequences(rows, self.codes, self.tokens)
        return self.torch.as_tensor(ids, device=self.device), self.torch.as_tensor(mask, device=self.device)

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def train_step(self, rows, step):
        batch = draw_training_batch(rows, self.args.batch_size, self.batch_rng)
        ids, mask = self._tensors(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate = self.args.learning_rate * min(1.0, step / max(1, self.args.warmup_steps))
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self._sync()
        started = time.monotonic()
        with self.torch.autocast(device_type=self.device.type, dtype=self.torch.bfloat16,
                                 enabled=self.device.type == "cuda"):
            loss = self.module.masked_diffusion_loss(self.model, ids, mask, self.tokens["mask_id"],
                                                    generator=self.generator)
        if not bool(self.torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with self.torch.no_grad():
            for key, value in self.model.state_dict().items():
                self.ema[key].mul_(self.args.ema_decay).add_(value.detach(), alpha=1-self.args.ema_decay)
        self._sync()
        self.cost["train_steps"] += 1
        self.cost["train_examples"] += len(batch)
        self.cost["train_code_tokens"] += int(mask.sum())
        self.cost["train_forward_calls"] += 1
        self.cost["train_backward_calls"] += 1
        self.cost["train_seconds"] += time.monotonic() - started
        return float(loss.detach().cpu())

    def validate(self, rows):
        # Separate deterministic stream and eval mode: identical masks/times at
        # every checkpoint; training/dropout RNG is untouched.
        generator = self.torch.Generator(device=self.device).manual_seed(self.args.seed + 3011)
        original = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema)
        self.model.eval()
        weighted, count, calls = 0.0, 0, 0
        self._sync()
        started = time.monotonic()
        try:
            with self.torch.inference_mode():
                for batch in example_batches(rows, self.args.batch_size):
                    ids, mask = self._tensors([row["item_indices"] for row in batch])
                    with self.torch.autocast(device_type=self.device.type, dtype=self.torch.bfloat16,
                                             enabled=self.device.type == "cuda"):
                        loss = self.module.masked_diffusion_loss(self.model, ids, mask, self.tokens["mask_id"],
                                                                generator=generator)
                    weighted += float(loss.detach().cpu()) * len(batch)
                    count += len(batch)
                    calls += 1
        finally:
            self.model.load_state_dict(original)
        self._sync()
        self.cost["validation_passes"] += 1
        self.cost["validation_examples"] += count
        self.cost["validation_forward_calls"] += calls
        self.cost["validation_seconds"] += time.monotonic() - started
        if not count or not math.isfinite(weighted):
            raise ValueError("Invalid calibration diffusion loss")
        return weighted / count

    def selected_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.ema.items()}

    def use_selected(self, state):
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, seed_index, sample_seed):
        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(sample_seed)
        self._sync()
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                                   enabled=self.device.type == "cuda"):
            result = self.module.sample_bundles(self.model,
                torch.as_tensor(self.codes, device=self.device),
                torch.tensor([seed_index], device=self.device),
                **{key: self.tokens[key] for key in ("bos_id", "boi_id", "eos_id", "mask_id")},
                total_items=3, steps=self.args.sampling_steps, generator=generator,
                return_diagnostics=True)
        self._sync()
        predictions, diagnostics = result
        diagnostics = {key: value.tolist() if hasattr(value, "tolist") else value
                       for key, value in diagnostics.items() if key != "trace"}
        return predictions[0].detach().cpu().tolist(), {
            "generation_seconds": time.monotonic() - started, "diffusion_samples": 1,
            "reverse_steps": self.args.sampling_steps, "denoiser_forward_calls": self.args.sampling_steps,
            "sampler_diagnostics": diagnostics}

    def snapshot(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "ema": self.ema,
                "torch_rng": self.torch.get_rng_state(),
                "cuda_rng": self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else [],
                "generator_rng": self.generator.get_state(), "batch_rng": self.batch_rng.getstate(), "cost": self.cost}

    def restore(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema = {key: value.to(self.device) for key, value in state["ema"].items()}
        self.torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        self.generator.set_state(state["generator_rng"].cpu())
        self.batch_rng.setstate(state["batch_rng"])
        self.cost = state["cost"]

    def save(self, path, state):
        self.torch.save(state, path)

    def load(self, path):
        return self.torch.load(path, map_location=self.device, weights_only=False)
