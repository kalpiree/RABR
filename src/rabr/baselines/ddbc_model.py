"""Discrete masked diffusion over residual-quantized item codes.

Modified from the DDBC/MDLM architecture; see THIRD_PARTY_NOTICES.md.
MDLM: Copyright 2024 Cornell University, Apache License 2.0.
Changes include SDPA attention, code-token loss normalization, and
seed-conditioned catalog-constrained sampling.
"""

from __future__ import annotations

import math
from itertools import islice
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "e1f4afc59121662e626ffefde79316a5e0dde044"
MODEL_VERSION = "ddbc-minilm-seed-complete-v1"


class LayerNorm(nn.Module):
    """Upstream float32 normalization with trainable scale, no learned bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x.float(), (self.dim,)) * self.weight


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.register_buffer("inv_freq", 1.0 / base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, length: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(length, device=self.inv_freq.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq)
        return frequencies.cos().to(dtype)[None, None], frequencies.sin().to(dtype)[None, None]


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """FlashAttention's default noninterleaved half-rotation; values unchanged."""
    def rotate(x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return rotate(q), rotate(k)


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    """q/k/v are [batch, heads, length, head_dim]; attention dropout is zero."""
    allowed = None if attention_mask is None else attention_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * frequencies[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim))
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: Tensor, rotary: tuple[Tensor, Tensor], c: Tensor,
                attention_mask: Tensor | None = None) -> Tensor:
        batch, length, dim = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c)[:, None].chunk(6, dim=-1)
        normalized = self.norm1(x) * (1 + scale_a) + shift_a
        qkv = self.attn_qkv(normalized).reshape(batch, length, 3, self.n_heads, dim // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        cos, sin = rotary
        q, k = apply_rotary(q, k, cos.to(q.dtype), sin.to(q.dtype))
        attended = sdpa_attention(q, k, v, attention_mask).transpose(1, 2).reshape(batch, length, dim)
        x = x + gate_a * F.dropout(self.attn_out(attended), self.dropout, self.training)
        mlp = self.mlp(self.norm2(x) * (1 + scale_m) + shift_m)
        return x + gate_m * F.dropout(mlp, self.dropout, self.training)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int, codebook_weights: Tensor | None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        if codebook_weights is not None:
            weights = torch.as_tensor(codebook_weights, dtype=torch.float32)
            if weights.ndim != 2 or weights.shape[1] != dim or weights.shape[0] + 1 >= vocab_size:
                raise ValueError("codebook_weights must be [semantic_code_count, hidden_size]")
            if not torch.isfinite(weights).all():
                raise ValueError("nonfinite codebook weights")
            with torch.no_grad():
                self.embedding[1:len(weights) + 1].copy_(weights)

    def forward(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.embedding)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        for layer in (self.linear, self.adaLN_modulation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class DDBCModel(nn.Module):
    """Audited upstream DDiT backbone, returning raw vocabulary logits."""

    def __init__(self, vocab_size: int, codebook_weights: Tensor | None = None,
                 hidden_size: int = 64, n_blocks: int = 6, n_heads: int = 8,
                 cond_dim: int = 128, dropout: float = 0.1, time_conditioning: bool = False):
        super().__init__()
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("hidden_size must give even-sized attention heads")
        if vocab_size < 4 or n_blocks < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid model configuration")
        self.vocab_size = vocab_size
        self.time_conditioning = time_conditioning
        self.config = dict(vocab_size=vocab_size, hidden_size=hidden_size, n_blocks=n_blocks,
                           n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
                           time_conditioning=time_conditioning)
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size, codebook_weights)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList([DDiTBlock(hidden_size, n_heads, cond_dim, dropout) for _ in range(n_blocks)])
        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)

    def forward(self, token_ids: Tensor, sigma: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2 or sigma.numel() != token_ids.shape[0]:
            raise ValueError("expected token_ids [B,L] and sigma [B] or [B,1]")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        sigma = sigma.reshape(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        x = self.vocab_embed(token_ids)
        c = F.silu(self.sigma_map(sigma))
        rotary = self.rotary_emb(token_ids.shape[1], x.dtype)
        for block in self.blocks:
            x = block(x, rotary, c, attention_mask)
        return self.output_layer(x, c)


def loglinear_noise(t: Tensor, eps: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Upstream LogLinearNoise: survival 1-(1-eps)*t."""
    if not 0 < eps < 1:
        raise ValueError("noise epsilon must be between zero and one")
    return -torch.log1p(-(1 - eps) * t), (1 - eps) / (1 - (1 - eps) * t)


def subs_log_probs(logits: Tensor, corrupted_ids: Tensor, mask_token_id: int) -> Tensor:
    """SUBS: exclude MASK from clean predictions; clamp every observed token."""
    logits = logits.float().clone()
    logits[..., mask_token_id] = -torch.inf
    predicted = logits.log_softmax(dim=-1)
    clamped = torch.full_like(predicted, -torch.inf)
    clamped.scatter_(-1, corrupted_ids[..., None], 0.0)
    return torch.where((corrupted_ids == mask_token_id)[..., None], predicted, clamped)


def masked_diffusion_loss(model: nn.Module, clean_ids: Tensor, code_mask: Tensor, mask_token_id: int,
                          *, t: Tensor | None = None, generator: torch.Generator | None = None,
                          noise_eps: float = 1e-3, sampling_eps: float = 1e-3,
                          antithetic: bool = True, reduction: str = "mean",
                          return_details: bool = False,
                          attention_mask: Tensor | None = None) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Released weighted absorbing-mask CE, normalized per code token/example.

    The generator must match clean_ids.device. Fixed t plus a reset generator
    makes validation corruption reproducible. This function does not switch
    model.train()/eval(), and does not force a mask when none were sampled.
    """
    if clean_ids.ndim != 2 or code_mask.shape != clean_ids.shape or code_mask.dtype != torch.bool:
        raise ValueError("clean_ids/code_mask must be matching [B,L], code_mask bool")
    if not 0 < sampling_eps < 1:
        raise ValueError("sampling_eps must be between zero and one")
    counts = code_mask.sum(dim=1)
    if (counts == 0).any() or (clean_ids == mask_token_id).any():
        raise ValueError("clean samples need code tokens and cannot contain MASK")
    batch = clean_ids.shape[0]
    if t is None:
        uniform = torch.rand(batch, device=clean_ids.device, generator=generator)
        if antithetic:
            uniform = (uniform + torch.arange(batch, device=clean_ids.device)) / batch
        t = sampling_eps + (1 - sampling_eps) * uniform
    else:
        t = torch.as_tensor(t, dtype=torch.float32, device=clean_ids.device).reshape(-1)
        if t.numel() != batch or not torch.isfinite(t).all() or (t <= 0).any() or (t > 1).any():
            raise ValueError("t must have one finite value in (0,1] per example")
    sigma, dsigma = loglinear_noise(t, noise_eps)
    chance = -torch.expm1(-sigma)
    masked = (torch.rand(clean_ids.shape, device=clean_ids.device, generator=generator) < chance[:, None]) & code_mask
    corrupted = clean_ids.masked_fill(masked, mask_token_id)
    if attention_mask is None:
        logits = model(corrupted, sigma)
    else:
        logits = model(corrupted, sigma, attention_mask=attention_mask)
    log_probs = subs_log_probs(logits, corrupted, mask_token_id)
    token_nll = -log_probs.gather(-1, clean_ids[..., None]).squeeze(-1)
    # Unmasked ground-truth tokens have exactly log probability zero under SUBS.
    weighted = torch.where(masked, token_nll, 0.0) * (dsigma / torch.expm1(sigma))[:, None]
    per_example = weighted.sum(dim=1) / counts
    if reduction == "mean":
        loss = per_example.mean()
    elif reduction == "none":
        loss = per_example
    elif reduction == "sum":
        loss = per_example.sum()
    else:
        raise ValueError("reduction must be mean, sum, or none")
    if return_details:
        return loss, {"t": t.detach(), "masked_count": masked.sum(dim=1).detach(),
                      "per_example_loss": per_example.detach(), "corrupted_ids": corrupted.detach()}
    return loss


def serialize_item_codes(item_codes: Tensor, bos_id: int, boi_id: int, eos_id: int) -> tuple[Tensor, Tensor]:
    """[B,K,L] -> BOS (BOI code[0:L])*K EOS, with code-position mask."""
    if item_codes.ndim != 3:
        raise ValueError("item_codes must be [B,K,L]")
    batch, count, depth = item_codes.shape
    sequence = torch.full((batch, 2 + count * (depth + 1)), boi_id,
                          dtype=torch.long, device=item_codes.device)
    sequence[:, 0], sequence[:, -1] = bos_id, eos_id
    code_mask = torch.zeros_like(sequence, dtype=torch.bool)
    for slot in range(count):
        start = 2 + slot * (depth + 1)
        sequence[:, start:start + depth] = item_codes[:, slot]
        code_mask[:, start:start + depth] = True
    return sequence, code_mask


def _can_assign_distinct(candidate_sets: Sequence[set[int]]) -> bool:
    """Small exact bipartite matching; prevents duplicate-only dead ends."""
    if any(not candidates for candidates in candidate_sets):
        return False
    assignment: dict[int, int] = {}
    def augment(slot: int, seen: set[int]) -> bool:
        # With m slots, retaining any m alternatives from a larger set cannot
        # destroy a matching: at most m-1 alternatives can be occupied elsewhere.
        # Avoid sorting thousands of catalog IDs for every candidate token.
        for item in islice(candidate_sets[slot], len(candidate_sets)):
            if item in seen:
                continue
            seen.add(item)
            if item not in assignment or augment(assignment[item], seen):
                assignment[item] = slot
                return True
        return False
    for slot in sorted(range(len(candidate_sets)), key=lambda s: len(candidate_sets[s])):
        if not augment(slot, set()):
            return False
    return True


@torch.no_grad()
def sample_bundles(model: nn.Module, catalog_codes: Tensor, seed_item_indices: Tensor,
                   bos_id: int, boi_id: int, eos_id: int, mask_id: int,
                   *, total_items: int = 3, steps: int = 25, temperature: float = 1.0,
                   generator: torch.Generator | None = None,
                   return_diagnostics: bool = False) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """One observed seed -> full-catalog, distinct-item bundle via DDBC denoising.

    Catalog codes are [N,L] globally offset integer token IDs (including the
    collision digit); all tuples must be unique. Seed indices are [B]. No gold
    labels are accepted. Structural tokens, seed tokens, and already revealed
    codes remain clamped. Reveal probability is (t-s)/t = 1/remaining_steps.
    Full-catalog feasibility constraints never consult item titles or targets.
    temperature=0 is greedy code selection; reveal order remains stochastic.
    """
    if catalog_codes.ndim != 2 or catalog_codes.dtype != torch.long:
        raise ValueError("catalog_codes must be an int64 [N,L] tensor")
    if seed_item_indices.ndim != 1 or seed_item_indices.dtype != torch.long:
        raise ValueError("seed_item_indices must be an int64 [B] tensor")
    count, depth = catalog_codes.shape
    if total_items < 1 or total_items > count or steps < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("invalid sample size, step count or temperature")
    if len({bos_id, boi_id, eos_id, mask_id}) != 4:
        raise ValueError("structural and mask tokens must be distinct")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = catalog_codes.device
    codes = catalog_codes.to(device)
    seeds = seed_item_indices.to(device)
    if seeds.numel() == 0 or (seeds < 0).any() or (seeds >= count).any():
        raise ValueError("seed indices must reference the catalog")
    code_rows = [tuple(row) for row in codes.cpu().tolist()]
    tuple_to_item = {row: index for index, row in enumerate(code_rows)}
    if len(tuple_to_item) != count or depth < 1:
        raise ValueError("catalog code tuples must be nonempty and unique")
    code_values = set(value for row in code_rows for value in row)
    if code_values & {bos_id, boi_id, eos_id, mask_id} or min(code_values) < 0:
        raise ValueError("catalog codes overlap structural tokens or contain negatives")
    domains = [set(row[level] for row in code_rows) for level in range(depth)]
    if any(domains[left] & domains[right] for left in range(depth) for right in range(left)):
        raise ValueError("code levels must occupy disjoint token domains")
    item_slots = torch.full((len(seeds), total_items, depth), mask_id, device=device, dtype=torch.long)
    item_slots[:, 0] = codes[seeds]
    tokens, code_mask = serialize_item_codes(item_slots, bos_id, boi_id, eos_id)
    original = tokens.clone()
    unknown = code_mask & (tokens == mask_id)
    candidates = [[set(range(count)) - {seed} for _ in range(total_items - 1)]
                  for seed in seeds.cpu().tolist()]
    was_training = model.training
    model.eval()
    reveal_counts: list[int] = []
    trace: list[Tensor] = [tokens.cpu().clone()] if return_diagnostics else []
    try:
        for iteration in range(steps):
            remaining = steps - iteration
            time = torch.full((len(seeds),), remaining / steps, device=device)
            sigma, _ = loglinear_noise(time)
            logits = model(tokens, sigma).float()
            if logits.shape[:2] != tokens.shape or max(code_values | {bos_id, boi_id, eos_id, mask_id}) >= logits.shape[-1]:
                raise ValueError("model vocabulary/sequence shape does not match catalog")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite DDBC generation logits")
            still_masked = unknown & (tokens == mask_id)
            reveal = still_masked & (torch.rand(tokens.shape, device=device, generator=generator) < 1.0 / remaining)
            positions = reveal.nonzero(as_tuple=False)
            # Random token-level reveal order avoids a fixed item completion order.
            order = torch.randperm(len(positions), device=device, generator=generator)
            for row, position in positions[order].cpu().tolist():
                slot, level = divmod(position - 2, depth + 1)
                if slot == 0 or level >= depth:
                    raise AssertionError("attempted to reveal a clamped/non-code token")
                slot_index = slot - 1
                groups: dict[int, set[int]] = {}
                for item in candidates[row][slot_index]:
                    groups.setdefault(code_rows[item][level], set()).add(item)
                valid_tokens: list[int] = []
                for token in sorted(groups):
                    trial = candidates[row].copy()
                    trial[slot_index] = groups[token]
                    if _can_assign_distinct(trial):
                        valid_tokens.append(token)
                if not valid_tokens:
                    raise RuntimeError("no valid distinct catalog completion; constraint invariant failed")
                valid = torch.tensor(valid_tokens, device=device, dtype=torch.long)
                scores = logits[row, position, valid]
                if temperature == 0:
                    chosen = int(scores.argmax().item())
                else:
                    probability = (scores / temperature).softmax(dim=-1)
                    chosen = int(torch.multinomial(probability, 1, generator=generator).item())
                token = valid_tokens[chosen]
                tokens[row, position] = token
                candidates[row][slot_index] = groups[token]
            reveal_counts.append(len(positions))
            if return_diagnostics:
                trace.append(tokens.cpu().clone())
        if (tokens == mask_id).any() or not torch.equal(tokens[~unknown], original[~unknown]):
            raise AssertionError("incomplete decoding or modified observed tokens")
        result = seeds.new_empty((len(seeds), total_items))
        for row in range(len(seeds)):
            for slot in range(total_items):
                start = 2 + slot * (depth + 1)
                value = tuple(tokens[row, start:start + depth].cpu().tolist())
                result[row, slot] = tuple_to_item[value]
            if len(set(result[row].cpu().tolist())) != total_items:
                raise AssertionError("duplicate item in generated bundle")
        if not torch.equal(result[:, 0], seeds):
            raise AssertionError("seed was not preserved")
        if return_diagnostics:
            return result, {"reveal_counts": reveal_counts, "token_ids": tokens.cpu(), "trace": trace,
                            "steps": steps, "temperature": temperature,
                            "constraint_policy": "existing_catalog_tuple_and_distinct_item_feasibility"}
        return result
    finally:
        model.train(was_training)
