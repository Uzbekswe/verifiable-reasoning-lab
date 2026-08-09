"""A small, readable pure-PyTorch Qwen3 implementation.

The attention, RoPE, tokenizer, and cache shapes follow the Apache-2.0 licensed
reference implementation in Sebastian Raschka's official repository. The
experiment-facing interface lives in ``backend.py``; this module only owns the
model implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151_936
    context_length: int = 40_960
    emb_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    hidden_dim: int = 3072
    head_dim: int = 128
    qk_norm: bool = True
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16


QWEN3_06B_CONFIG = Qwen3Config()


class KVCache:
    """Per-layer key/value storage for incremental decoding."""

    def __init__(self, n_layers: int):
        self._values: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def get(self, layer_idx: int):
        return self._values[layer_idx]

    def update(self, layer_idx: int, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._values[layer_idx] = value

    def reset(self) -> None:
        self._values = [None] * len(self._values)


def _rope_tables(head_dim: int, theta: float, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length).float()
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int
) -> torch.Tensor:
    _, _, seq_len, head_dim = x.shape
    cos = cos[offset : offset + seq_len][None, None]
    sin = sin[offset : offset + seq_len][None, None]
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return (normalized * self.scale).to(dtype=input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        if cfg.n_heads % cfg.n_kv_groups:
            raise ValueError("n_heads must be divisible by n_kv_groups")
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        self.group_size = cfg.n_heads // cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        self.d_out = cfg.n_heads * cfg.head_dim
        dtype = cfg.dtype
        self.W_query = nn.Linear(cfg.emb_dim, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, cfg.emb_dim, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape
        queries = self.W_query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = self.W_key(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = self.W_value(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            queries, keys_new = self.q_norm(queries), self.k_norm(keys_new)
        queries = _apply_rope(queries, cos, sin, start_pos)
        keys_new = _apply_rope(keys_new, cos, sin, start_pos)
        if cache is None:
            keys, values = keys_new, values_new
        else:
            keys = torch.cat((cache[0], keys_new), dim=2)
            values = torch.cat((cache[1], values_new), dim=2)
        next_cache = (keys, values)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        scores = (queries @ keys.transpose(2, 3)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ values).transpose(1, 2).reshape(batch, seq_len, self.d_out)
        return self.out_proj(context), next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        dtype = cfg.dtype
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.nn.functional.silu(self.fc1(x)) * self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.norm2 = RMSNorm(cfg.emb_dim)

    def forward(self, x, mask, cos, sin, start_pos, cache):
        residual = x
        attended, next_cache = self.att(self.norm1(x), mask, cos, sin, start_pos, cache)
        x = residual + attended
        return x + self.ff(self.norm2(x)), next_cache


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config = QWEN3_06B_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        cos, sin = _rope_tables(cfg.head_dim, cfg.rope_base, cfg.context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0

    def reset_kv_cache(self) -> None:
        self.current_pos = 0

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        x = self.tok_emb(input_ids)
        seq_len = x.shape[1]
        start_pos = self.current_pos if cache is not None else 0
        total_len = start_pos + seq_len
        if total_len > self.cfg.context_length:
            raise ValueError("sequence exceeds the configured context length")
        if cache is not None:
            self.current_pos = total_len
        mask = torch.triu(torch.ones(total_len, total_len, device=x.device, dtype=torch.bool), diagonal=1)
        mask = mask[start_pos:total_len][None, None]
        for layer_idx, block in enumerate(self.trf_blocks):
            layer_cache = cache.get(layer_idx) if cache is not None else None
            x, next_cache = block(x, mask, self.cos, self.sin, start_pos, layer_cache)
            if cache is not None:
                cache.update(layer_idx, next_cache)
        return self.out_head(self.final_norm(x).to(self.cfg.dtype))


class Qwen3Tokenizer:
    """Minimal tokenizer wrapper for the official Qwen3 tokenizer JSON."""

    _SPECIALS = (
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
        "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
        "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    )
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file: str | Path):
        from tokenizers import Tokenizer

        path = Path(tokenizer_file)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file does not exist: {path}")
        self._tokenizer = Tokenizer.from_file(str(path))
        self._special_to_id = {token: self._tokenizer.token_to_id(token) for token in self._SPECIALS}
        self.eos_token = "<|endoftext|>" if "base" in path.name.lower() else "<|im_end|>"
        self.eos_token_id = self._special_to_id[self.eos_token]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id and self._special_to_id[part] is not None:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tokenizer.encode(part).ids)
        return ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)
