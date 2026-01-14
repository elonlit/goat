#!/usr/bin/env python3
"""Train 125M GPT-style causal LMs on C4 comparing RoPE, ALiBi, and GOAT."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import inspect
import io
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.checkpoint import checkpoint

from goat import GoatAttention

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_rank0() -> bool:
    return get_rank() == 0

def barrier() -> None:
    if is_distributed():
        dist.barrier()

def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def setup_distributed(backend: str = "auto") -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        elif backend == "nccl" and (not torch.cuda.is_available()):
            backend = "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        dist.barrier()
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()



def unwrap_model(m: nn.Module) -> nn.Module:
    if isinstance(m, DDP):
        m = m.module
    return m



class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: Optional[io.TextIOWrapper] = None

    def __enter__(self):
        if is_rank0():
            self._fp = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None

    def log(self, obj: Dict[str, Any]) -> None:
        if not is_rank0():
            return
        assert self._fp is not None
        self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fp.flush()

def save_json(path: Path, obj: Any) -> None:
    if not is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def atomic_torch_save(obj: Any, path: Path) -> None:
    if not is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class C4TokenBlockDataset(IterableDataset):
    def __init__(
        self,
        *,
        split: Literal["train", "validation"],
        tokenizer_name: str,
        seq_len: int,
        seed: int,
        shuffle: bool,
        shuffle_buffer: int,
        hf_c4_config: str = "en",
        hf_cache_dir: Optional[str] = None,
        tokenize_batch_size: int = 32,
        rank: int = 0,
        world_size: int = 1,
        max_examples: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = int(shuffle_buffer)
        self.hf_c4_config = hf_c4_config
        self.hf_cache_dir = hf_cache_dir
        self.tokenize_batch_size = int(tokenize_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_examples = max_examples
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except Exception as e:
                raise RuntimeError("transformers is required: pip install transformers") from e

            tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
            try:
                tok.model_max_length = int(10**9)
            except Exception:
                pass
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            if tok.bos_token_id is None:
                tok.bos_token = tok.eos_token
            self._tokenizer = tok
        return self._tokenizer

    def __iter__(self) -> Iterator[torch.Tensor]:
        try:
            import datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("datasets is required: pip install datasets") from e

        tok = self._get_tokenizer()
        bos_id = int(tok.bos_token_id)
        eos_id = int(tok.eos_token_id)

        ds = datasets.load_dataset(
            "allenai/c4",
            self.hf_c4_config,
            split=self.split,
            streaming=True,
            cache_dir=self.hf_cache_dir,
        )

        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(wi.id)
            num_workers = int(wi.num_workers)

        global_shards = self.world_size * num_workers
        global_index = self.rank * num_workers + worker_id
        if global_shards > 1:
            ds = ds.shard(num_shards=global_shards, index=global_index)

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed + global_index, buffer_size=self.shuffle_buffer)

        buffer: List[int] = []
        texts: List[str] = []
        yielded = 0

        def flush_texts():
            nonlocal buffer, texts, yielded
            if self.max_examples is not None and yielded >= self.max_examples:
                texts.clear()
                return
            if not texts:
                return
            enc = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            input_ids_batch: List[List[int]] = enc["input_ids"]
            for ids in input_ids_batch:
                if not ids:
                    continue
                buffer.extend(ids)
                buffer.append(eos_id)

                while len(buffer) >= self.seq_len:
                    if self.max_examples is not None and yielded >= self.max_examples:
                        texts.clear()
                        return
                    payload = buffer[: self.seq_len]
                    buffer = buffer[self.seq_len :]
                    block = [bos_id] + payload  # (seq_len+1,)
                    yield torch.tensor(block, dtype=torch.long)
                    yielded += 1
                    if self.max_examples is not None and yielded >= self.max_examples:
                        texts.clear()
                        return
            texts.clear()

        for ex in ds:
            if self.max_examples is not None and yielded >= self.max_examples:
                break
            txt = ex.get("text", "")
            if not isinstance(txt, str) or len(txt) == 0:
                continue
            texts.append(txt)
            if len(texts) >= self.tokenize_batch_size:
                for item in flush_texts() or []:
                    yield item
                    if self.max_examples is not None and yielded >= self.max_examples:
                        break

        for item in flush_texts() or []:
            if self.max_examples is not None and yielded >= self.max_examples:
                break
            yield item


def make_dataloader(
    *,
    split: Literal["train", "validation"],
    tokenizer_name: str,
    seq_len: int,
    micro_batch_size: int,
    seed: int,
    shuffle: bool,
    shuffle_buffer: int,
    hf_cache_dir: Optional[str],
    tokenize_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    rank: int,
    world_size: int,
    max_examples: Optional[int] = None,
) -> DataLoader:
    ds = C4TokenBlockDataset(
        split=split,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        seed=seed,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        hf_cache_dir=hf_cache_dir,
        tokenize_batch_size=tokenize_batch_size,
        rank=rank,
        world_size=world_size,
        max_examples=max_examples,
    )
    loader = DataLoader(
        ds,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader



def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even.")
        self.dim = int(dim)
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cache_len: int = 0
        self._cache_device: Optional[torch.device] = None
        self._cache_dtype: Optional[torch.dtype] = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self._cos_cached = cos[None, None, :, :]
        self._sin_cached = sin[None, None, :, :]
        self._cache_len = int(seq_len)
        self._cache_device = device
        self._cache_dtype = dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor, *, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        need = offset + T
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cache_len < need
            or self._cache_device != q.device
            or self._cache_dtype != q.dtype
        ):
            self._build_cache(need, device=q.device, dtype=q.dtype)

        cos = self._cos_cached[:, :, offset : offset + T, :]
        sin = self._sin_cached[:, :, offset : offset + T, :]
        return _apply_rope(q, sin=sin, cos=cos), _apply_rope(k, sin=sin, cos=cos)



_ALIBI_BIAS_CACHE: "OrderedDict[Tuple[int, int, str, torch.dtype, int], torch.Tensor]" = OrderedDict()
_ALIBI_BIAS_CACHE_MAX_ITEMS = 8

def clear_alibi_bias_cache() -> None:
    _ALIBI_BIAS_CACHE.clear()

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)

def alibi_slopes(n_heads: int) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if _is_power_of_two(n_heads):
        slopes = slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** int(math.floor(math.log2(n_heads)))
        slopes = slopes_power_of_2(closest_pow2)
        slopes_extra = alibi_slopes(2 * closest_pow2).tolist()
        slopes += slopes_extra[0::2][: (n_heads - closest_pow2)]
    return torch.tensor(slopes, dtype=torch.float32)

class AlibiBias(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=True)

    @torch.no_grad()
    def full_bias(self, T: int, device: torch.device, dtype: torch.dtype, *, offset: int = 0) -> torch.Tensor:
        key = (self.slopes.numel(), T, str(device), dtype)
        cache_ok = T <= 2048
        if cache_ok and key in _ALIBI_BIAS_CACHE:
            bias = _ALIBI_BIAS_CACHE.pop(key)
            _ALIBI_BIAS_CACHE[key] = bias  # LRU refresh
            return bias

        pos = torch.arange(T, device=device, dtype=torch.long)
        i = pos.view(T, 1)
        j = pos.view(1, T)
        dist = i - j
        slopes = self.slopes.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        bias = -slopes * dist.to(dtype).view(1, 1, T, T)
        bias = bias.masked_fill(dist.view(1, 1, T, T) < 0, float("-inf"))
        if cache_ok:
            _ALIBI_BIAS_CACHE[key] = bias
            while len(_ALIBI_BIAS_CACHE) > _ALIBI_BIAS_CACHE_MAX_ITEMS:
                _ALIBI_BIAS_CACHE.popitem(last=False)
        return bias

@torch.no_grad()
def alibi_attention_blockwise_eval(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slopes: torch.Tensor,
    *,
    position_offset: int = 0,
    block_q: int = 128,
    block_k: int = 1024,
) -> torch.Tensor:
    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype

    slopes_f = slopes.to(device=device, dtype=torch.float32).view(1, H, 1, 1)
    inv_sqrt_d = 1.0 / math.sqrt(D)

    pos = (position_offset + torch.arange(T, device=device, dtype=torch.long))

    out = torch.empty((B, H, T, D), device=device, dtype=torch.float32)

    for i0 in range(0, T, block_q):
        i1 = min(T, i0 + block_q)
        Br = i1 - i0

        qb = q[:, :, i0:i1, :].to(torch.float32)
        qpos = pos[i0:i1]
        m = torch.full((B, H, Br, 1), float("-inf"), device=device, dtype=torch.float32)
        l = torch.zeros((B, H, Br, 1), device=device, dtype=torch.float32)
        acc = torch.zeros((B, H, Br, D), device=device, dtype=torch.float32)
        max_key = i1

        for j0 in range(0, max_key, block_k):
            j1 = min(max_key, j0 + block_k)
            Bc = j1 - j0

            kb = k[:, :, j0:j1, :].to(torch.float32)
            vb = v[:, :, j0:j1, :].to(torch.float32)
            kpos = pos[j0:j1]

            scores = torch.matmul(qb, kb.transpose(-1, -2)) * inv_sqrt_d
            dist = (qpos[:, None] - kpos[None, :]).to(torch.float32)
            bias = -slopes_f * dist.view(1, 1, Br, Bc)
            scores = scores + bias

            allowed = (kpos[None, :] <= qpos[:, None])
            scores = scores.masked_fill(~allowed.view(1, 1, Br, Bc), float("-inf"))

            block_max = scores.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m, block_max)

            exp_m = torch.exp(m - m_new)
            exp_scores = torch.exp(scores - m_new)

            l = l * exp_m + exp_scores.sum(dim=-1, keepdim=True)
            acc = acc * exp_m + torch.matmul(exp_scores, vb)

            m = m_new

        out[:, :, i0:i1, :] = acc / l

    return out.to(dtype=dtype)



@dataclass
class AttnStats:
    omega_mean: torch.Tensor
    p_token0_mean: torch.Tensor
    p_max_mean: torch.Tensor
    p_recent_window_mean: torch.Tensor
    recent_window: int

def _causal_upper_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)



class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, rope_base: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(dim=self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn_stats: bool = False,
        recent_window: int = 128,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttnStats]]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, offset=position_offset)

        stats: Optional[AttnStats] = None

        if return_attn_stats:
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask_future = _causal_upper_mask(T, device=x.device)
            logits_masked = logits.masked_fill(mask_future, float("-inf"))

            weights = F.softmax(logits_masked, dim=-1)
            if self.training and self.attn_dropout.p > 0:
                weights = self.attn_dropout(weights)
            y = torch.matmul(weights, v)

            logits_for_max = logits.masked_fill(mask_future, float("-inf"))
            logits_for_min = logits.masked_fill(mask_future, float("inf"))
            omega = (logits_for_max.amax(dim=-1) - logits_for_min.amin(dim=-1)).mean(dim=1)

            p0 = weights[..., 0].mean(dim=1)
            pmax = weights.amax(dim=-1).mean(dim=1)

            w = int(recent_window)
            if w <= 0:
                precent = torch.zeros((B, T), device=x.device, dtype=weights.dtype)
            else:
                idx = torch.arange(T, device=x.device)
                jdx = torch.arange(T, device=x.device)
                win_mask = (jdx[None, :] <= idx[:, None]) & (jdx[None, :] >= (idx[:, None] - (w - 1)))
                precent = (weights * win_mask.view(1, 1, T, T)).sum(dim=-1).mean(dim=1)

            stats = AttnStats(
                omega_mean=omega.detach(),
                p_token0_mean=p0.detach(),
                p_max_mean=pmax.detach(),
                p_recent_window_mean=precent.detach(),
                recent_window=w,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, stats


class CausalSelfAttentionALiBi(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, *, long_eval_threshold: int = 4096):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.alibi = AlibiBias(n_heads=n_head)
        self.long_eval_threshold = int(long_eval_threshold)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn_stats: bool = False,
        recent_window: int = 128,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttnStats]]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        stats: Optional[AttnStats] = None

        if return_attn_stats:
            content_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            bias = self.alibi.full_bias(T, device=x.device, dtype=q.dtype, offset=position_offset)  # (1,H,T,T)
            logits = content_logits + bias

            weights = F.softmax(logits, dim=-1)
            if self.training and self.attn_dropout.p > 0:
                weights = self.attn_dropout(weights)
            y = torch.matmul(weights, v)

            mask_future = _causal_upper_mask(T, device=x.device)
            cl_for_max = content_logits.masked_fill(mask_future, float("-inf"))
            cl_for_min = content_logits.masked_fill(mask_future, float("inf"))
            omega = (cl_for_max.amax(dim=-1) - cl_for_min.amin(dim=-1)).mean(dim=1)

            p0 = weights[..., 0].mean(dim=1)
            pmax = weights.amax(dim=-1).mean(dim=1)

            w = int(recent_window)
            if w <= 0:
                precent = torch.zeros((B, T), device=x.device, dtype=weights.dtype)
            else:
                idx = torch.arange(T, device=x.device)
                jdx = torch.arange(T, device=x.device)
                win_mask = (jdx[None, :] <= idx[:, None]) & (jdx[None, :] >= (idx[:, None] - (w - 1)))
                precent = (weights * win_mask.view(1, 1, T, T)).sum(dim=-1).mean(dim=1)

            stats = AttnStats(
                omega_mean=omega.detach(),
                p_token0_mean=p0.detach(),
                p_max_mean=pmax.detach(),
                p_recent_window_mean=precent.detach(),
                recent_window=w,
            )
        else:
            if (not self.training) and T >= self.long_eval_threshold:
                y = alibi_attention_blockwise_eval(
                    q, k, v,
                    self.alibi.slopes,  # (H,) fp32
                    position_offset=position_offset,
                    block_q=128,
                    block_k=1024,
                )
            else:
                bias = self.alibi.full_bias(T, device=x.device, dtype=q.dtype, offset=position_offset)  # (1,H,T,T)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=bias,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, stats


class CausalSelfAttentionGOAT(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        goat_pos_rank: int,
        goat_abs_rank: int,
        goat_pos_base: float,
        goat_abs_base: float,
        goat_enable_key_bias: bool,
        goat_training_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.mha = GoatAttention.for_gpt(
            embed_dim=n_embd,
            num_heads=n_head,
            kv_num_heads=None,
            dropout=dropout,
            pos_rank=goat_pos_rank,
            abs_rank=goat_abs_rank,
            pos_base=goat_pos_base,
            abs_base=goat_abs_base,
            enable_key_bias=goat_enable_key_bias,
            training_seq_len=goat_training_seq_len,
        )
        self.resid_dropout = nn.Dropout(dropout)
        self.long_eval_threshold = 8192
        self.long_eval_block_q = 512

    @torch.no_grad()
    def _content_logits(self, x: torch.Tensor, *, position_offset: int) -> torch.Tensor:
        B, T, E = x.shape
        attn = self.mha
        device = x.device
        dtype = x.dtype

        xt = x.transpose(0, 1)  # (T,B,E)
        if attn.in_proj is None:
            raise RuntimeError("Expected fuse_qkv=True for GOAT (in_proj must exist).")

        qkv = attn.in_proj(xt)  # (T,B,E+2E)= (T,B,3E)
        q_lin, k_lin, _v_lin = torch.split(qkv, [E, E, E], dim=-1)

        D = attn.head_dim
        H = attn.num_heads
        q = q_lin.view(T, B, H, D).permute(1, 2, 0, 3).contiguous()
        k = k_lin.view(T, B, H, D).permute(1, 2, 0, 3).contiguous()

        Dc = attn.content_dim
        if Dc <= 0:
            return torch.zeros((B, H, T, T), device=device, dtype=dtype)
        qc = q[..., :Dc] / math.sqrt(Dc)
        kc = k[..., :Dc]
        logits = torch.matmul(qc, kc.transpose(-1, -2)).to(dtype=dtype)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn_stats: bool = False,
        recent_window: int = 128,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttnStats]]:
        stats: Optional[AttnStats] = None
        if return_attn_stats:
            out, weights = self.mha(
                x, x, x,
                is_causal=True,
                need_weights=True,
                average_attn_weights=False,
                position_offset_q=position_offset,
                position_offset_k=position_offset,
            )
            assert weights is not None  # (B,H,T,T)

            content_logits = self._content_logits(x, position_offset=position_offset)  # (B,H,T,T)
            B, H, T, _ = weights.shape
            mask_future = _causal_upper_mask(T, device=x.device)

            cl_for_max = content_logits.masked_fill(mask_future, float("-inf"))
            cl_for_min = content_logits.masked_fill(mask_future, float("inf"))
            omega = (cl_for_max.amax(dim=-1) - cl_for_min.amin(dim=-1)).mean(dim=1)

            p0 = weights[..., 0].mean(dim=1)
            pmax = weights.amax(dim=-1).mean(dim=1)

            w = int(recent_window)
            if w <= 0:
                precent = torch.zeros((B, T), device=x.device, dtype=weights.dtype)
            else:
                idx = torch.arange(T, device=x.device)
                jdx = torch.arange(T, device=x.device)
                win_mask = (jdx[None, :] <= idx[:, None]) & (jdx[None, :] >= (idx[:, None] - (w - 1)))
                precent = (weights * win_mask.view(1, 1, T, T)).sum(dim=-1).mean(dim=1)

            stats = AttnStats(
                omega_mean=omega.detach(),
                p_token0_mean=p0.detach(),
                p_max_mean=pmax.detach(),
                p_recent_window_mean=precent.detach(),
                recent_window=w,
            )
        else:
            B, T, _E = x.shape

            if (not self.training) and T >= self.long_eval_threshold:
                base = position_offset
                outs: List[torch.Tensor] = []
                past_kv = None

                for i0 in range(0, T, self.long_eval_block_q):
                    i1 = min(T, i0 + self.long_eval_block_q)
                    chunk = x[:, i0:i1, :]  # (B, Q, E)

                    block_out, _weights, past_kv = self.mha(
                        chunk, chunk, chunk,
                        is_causal=True,
                        need_weights=False,
                        past_key_value=past_kv,
                        use_cache=True,
                        return_present_kv=True,
                        position_offset_q=base + i0,
                        position_offset_k=base + i0,
                    )
                    outs.append(block_out)

                out = torch.cat(outs, dim=1)
            else:
                out, _ = self.mha(
                    x, x, x,
                    is_causal=True,
                    need_weights=False,
                    position_offset_q=position_offset,
                    position_offset_k=position_offset,
                )

        out = self.resid_dropout(out)
        return out, stats



class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_embd, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        try:
            x = F.gelu(x, approximate="tanh")
        except TypeError:
            x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        attn_variant: Literal["rope", "alibi", "goat"],
        rope_base: float,
        goat_pos_rank: int,
        goat_abs_rank: int,
        goat_pos_base: float,
        goat_abs_base: float,
        goat_enable_key_bias: bool,
        alibi_long_eval_threshold: int,
        goat_training_seq_len: int = 2048,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)

        if attn_variant == "rope":
            self.attn = CausalSelfAttentionRoPE(n_embd, n_head, dropout, rope_base=rope_base)
        elif attn_variant == "alibi":
            self.attn = CausalSelfAttentionALiBi(n_embd, n_head, dropout, long_eval_threshold=alibi_long_eval_threshold)
        elif attn_variant == "goat":
            self.attn = CausalSelfAttentionGOAT(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                goat_pos_rank=goat_pos_rank,
                goat_abs_rank=goat_abs_rank,
                goat_pos_base=goat_pos_base,
                goat_abs_base=goat_abs_base,
                goat_enable_key_bias=goat_enable_key_bias,
                goat_training_seq_len=goat_training_seq_len,
            )
        else:
            raise ValueError(f"Unknown attn_variant: {attn_variant}")

        self.mlp = MLP(n_embd, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn_stats: bool = False,
        recent_window: int = 128,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttnStats]]:
        a, stats = self.attn(
            self.ln1(x),
            return_attn_stats=return_attn_stats,
            recent_window=recent_window,
            position_offset=position_offset,
        )
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, stats


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    rope_base: float = 10000.0

    goat_pos_rank: int = 4
    goat_abs_rank: int = 4
    goat_pos_base: float = 10000.0
    goat_abs_base: float = 10000.0
    goat_enable_key_bias: bool = True
    goat_training_seq_len: int = 2048

    alibi_long_eval_threshold: int = 4096

    attn_variant: Literal["rope", "alibi", "goat"] = "rope"
    gradient_checkpointing: bool = False


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            Block(
                n_embd=cfg.n_embd,
                n_head=cfg.n_head,
                dropout=cfg.dropout,
                attn_variant=cfg.attn_variant,
                rope_base=cfg.rope_base,
                goat_pos_rank=cfg.goat_pos_rank,
                goat_abs_rank=cfg.goat_abs_rank,
                goat_pos_base=cfg.goat_pos_base,
                goat_abs_base=cfg.goat_abs_base,
                goat_enable_key_bias=cfg.goat_enable_key_bias,
                alibi_long_eval_threshold=cfg.alibi_long_eval_threshold,
                goat_training_seq_len=cfg.goat_training_seq_len,
            )
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie

        self.gradient_checkpointing = bool(cfg.gradient_checkpointing)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        return_attn_stats: bool = False,
        attn_stats_layer: int = -1,
        recent_window: int = 128,
        position_offset: int = 0,
    ) -> (
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        | Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[AttnStats]]
    ):
        B, T = idx.shape
        x = self.wte(idx)
        x = self.drop(x)

        stats_out: Optional[AttnStats] = None
        if attn_stats_layer < 0:
            attn_stats_layer = len(self.blocks) - 1

        for li, block in enumerate(self.blocks):
            want_stats = return_attn_stats and (li == attn_stats_layer)

            if self.gradient_checkpointing and self.training and (not want_stats):
                def _ckpt_fn(y):
                    y2, _ = block(y, return_attn_stats=False, recent_window=recent_window, position_offset=position_offset)
                    return y2
                try:
                    x = checkpoint(_ckpt_fn, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(_ckpt_fn, x)
            else:
                x, stats = block(
                    x,
                    return_attn_stats=want_stats,
                    recent_window=recent_window,
                    position_offset=position_offset,
                )
                if want_stats:
                    stats_out = stats

        x = self.ln_f(x)
        LOGIT_CHUNK = 2048

        logits: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

        if targets is not None and T > LOGIT_CHUNK:
            total_loss = torch.zeros((), device=x.device, dtype=torch.float32)
            total_tok = 0
            for t0 in range(0, T, LOGIT_CHUNK):
                t1 = min(T, t0 + LOGIT_CHUNK)
                logits_chunk = self.lm_head(x[:, t0:t1, :])  # (B,chunk,V)
                loss_chunk = F.cross_entropy(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    targets[:, t0:t1].reshape(-1),
                    reduction="sum",
                )
                total_loss = total_loss + loss_chunk.float()
                total_tok += int((t1 - t0) * B)
            loss = total_loss / float(total_tok)
        else:
            logits = self.lm_head(x)
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )

        if return_attn_stats:
            return logits, loss, stats_out
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    m = unwrap_model(model)
    return sum(p.numel() for p in m.parameters())



@dataclass
class TrainConfig:
    tokenizer_name: str = "gpt2"
    hf_cache_dir: Optional[str] = None

    seq_len_train: int = 2048
    total_tokens: int = 4_000_000_000
    micro_batch_size: int = 2
    grad_accum: int = 8
    max_steps: Optional[int] = None

    lr: float = 1.5e-4
    min_lr: float = 3e-5
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    log_interval: int = 10
    eval_interval: int = 2000
    eval_batches: int = 50
    save_interval: int = 10_000
    save_best: bool = True

    extrap_lengths: Tuple[int, ...] = (2048, 4096, 8192, 16384)
    extrap_eval_batches: int = 100
    extrap_batch_size: int = 1

    diag_layer: int = -1
    diag_batches: int = 8
    diag_batch_size: int = 1
    diag_recent_window: int = 128

    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    tf32: bool = True

    shuffle_buffer: int = 100_000
    tokenize_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    resume: bool = True
    seed: int = 1337


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (
            name.endswith(".bias")
            or ("ln" in lname)
            or ("layernorm" in lname)
            or ("norm" in lname)
            or ("lambda_" in lname)
            or ("key_bias" in lname)
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_ok = False
    try:
        sig = inspect.signature(torch.optim.AdamW)
        fused_ok = ("fused" in sig.parameters) and torch.cuda.is_available()
    except Exception:
        fused_ok = False

    opt_kwargs = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    if fused_ok:
        opt_kwargs["fused"] = True

    return torch.optim.AdamW(param_groups, **opt_kwargs)


def lr_schedule(step: int, *, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * float(step) / float(max(1, cfg.warmup_steps))
    progress = float(step - cfg.warmup_steps) / float(max(1, total_steps - cfg.warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.lr - cfg.min_lr)



def _autocast_context(precision: str, *, device: torch.device):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        return contextlib.nullcontext()
    if precision == "bf16":
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        except Exception:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if precision == "fp16":
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            return torch.cuda.amp.autocast(dtype=torch.float16)
    raise ValueError(f"Unknown precision: {precision}")

def _make_grad_scaler(precision: str) -> Optional[torch.cuda.amp.GradScaler]:
    if precision == "fp16" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None



def torch_load_full(path: Path, *, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    train_cfg: TrainConfig,
    model_cfg: GPTConfig,
) -> None:
    if not is_rank0():
        return
    raw = unwrap_model(model)
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_cfg": dataclasses.asdict(train_cfg),
        "model_cfg": dataclasses.asdict(model_cfg),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    atomic_torch_save(state, path)

def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    map_location: str,
) -> Tuple[int, int, float]:
    ckpt = torch_load_full(path, map_location=map_location)
    raw = unwrap_model(model)
    raw.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    rng = ckpt.get("rng_state", None)
    if rng is not None:
        try:
            if get_world_size() == 1:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass

    return int(ckpt.get("step", 0)), int(ckpt.get("tokens_seen", 0)), float(ckpt.get("best_val_loss", float("inf")))



@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    data_iter: Iterator[torch.Tensor],
    *,
    device: torch.device,
    autocast_ctx,
    eval_batches: int,
    distributed_reduce: bool = True,
) -> float:
    was_training = model.training
    model.eval()
    try:
        loss_sum = 0.0
        tok_sum = 0

        for _ in range(eval_batches):
            batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            x = batch[:, :-1]
            y = batch[:, 1:]

            with autocast_ctx:
                _logits, loss = model(x, targets=y, return_attn_stats=False)
            assert loss is not None
            loss_sum += loss.item() * x.numel()
            tok_sum += x.numel()

        loss_sum_t = torch.tensor(loss_sum, device=device, dtype=torch.float64)
        tok_sum_t = torch.tensor(tok_sum, device=device, dtype=torch.float64)
        if distributed_reduce:
            ddp_all_reduce_sum(loss_sum_t)
            ddp_all_reduce_sum(tok_sum_t)
        return float((loss_sum_t / tok_sum_t).item())
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_ppl_by_length(
    model: nn.Module,
    *,
    lengths: List[int],
    train_cfg: TrainConfig,
    model_device: torch.device,
    rank: int,
    world_size: int,
) -> Dict[int, float]:
    results: Dict[int, float] = {}
    for L in lengths:
        loader = make_dataloader(
            split="validation",
            tokenizer_name=train_cfg.tokenizer_name,
            seq_len=L,
            micro_batch_size=train_cfg.extrap_batch_size,
            seed=train_cfg.seed + 123,
            shuffle=False,
            shuffle_buffer=train_cfg.shuffle_buffer,
            hf_cache_dir=train_cfg.hf_cache_dir,
            tokenize_batch_size=train_cfg.tokenize_batch_size,
            # Streaming HF datasets are fragile with >0 workers and/or heavy sharding.
            # Extrapolation is rank0-only (caller enforces) so keep it simple and robust.
            num_workers=0,
            pin_memory=train_cfg.pin_memory,
            rank=rank,
            world_size=world_size,
        )
        it = iter(loader)
        autocast_ctx = _autocast_context(train_cfg.precision, device=model_device)

        with contextlib.nullcontext():
            base_batches = int(train_cfg.extrap_eval_batches)
            ref_len = 8192
            target_tokens = base_batches * ref_len
            eval_batches = base_batches if int(L) <= ref_len else max(1, int(target_tokens // int(L)))
            loss = evaluate_loss(
                model,
                it,
                device=model_device,
                autocast_ctx=autocast_ctx,
                eval_batches=eval_batches,
                distributed_reduce=(world_size > 1),
            )
        ppl = math.exp(loss)
        results[L] = ppl
        if is_rank0():
            print(f"[extrap] L={L} loss={loss:.4f} ppl={ppl:.2f}", flush=True)
    return results



@torch.no_grad()
def collect_signal_vs_sink(
    model: nn.Module,
    *,
    train_cfg: TrainConfig,
    seq_len: int,
    device: torch.device,
    rank: int,
    world_size: int,
    variant: str,
    out_path: Path,
) -> None:
    if not is_rank0():
        return
    model.eval()

    loader = make_dataloader(
        split="validation",
        tokenizer_name=train_cfg.tokenizer_name,
        seq_len=seq_len,
        micro_batch_size=train_cfg.diag_batch_size,
        seed=train_cfg.seed + 999,
        shuffle=False,
        shuffle_buffer=train_cfg.shuffle_buffer,
        hf_cache_dir=train_cfg.hf_cache_dir,
        tokenize_batch_size=train_cfg.tokenize_batch_size,
        num_workers=0,
        pin_memory=train_cfg.pin_memory,
        rank=0,
        world_size=1,
    )
    it = iter(loader)

    raw_model = unwrap_model(model)
    last_layer = len(raw_model.blocks) - 1
    layer_idx = train_cfg.diag_layer if train_cfg.diag_layer >= 0 else last_layer

    omega_all: List[torch.Tensor] = []
    p0_all: List[torch.Tensor] = []
    pmax_all: List[torch.Tensor] = []
    precent_all: List[torch.Tensor] = []

    autocast_ctx = _autocast_context(train_cfg.precision, device=device)
    diag_model = unwrap_model(model)

    for b in range(train_cfg.diag_batches):
        batch = next(it).to(device, non_blocking=True)
        x = batch[:, :-1]

        with autocast_ctx:
            _logits, _loss, stats = diag_model(
                x,
                targets=None,
                return_attn_stats=True,
                attn_stats_layer=layer_idx,
                recent_window=train_cfg.diag_recent_window,
                position_offset=0,
            )
        if stats is None:
            raise RuntimeError("Expected AttnStats but got None. Check diag layer selection and wiring.")

        omega_all.append(stats.omega_mean.detach().float().cpu().reshape(-1))
        p0_all.append(stats.p_token0_mean.detach().float().cpu().reshape(-1))
        pmax_all.append(stats.p_max_mean.detach().float().cpu().reshape(-1))
        precent_all.append(stats.p_recent_window_mean.detach().float().cpu().reshape(-1))

        if is_rank0():
            print(f"[diag] {variant} batch {b+1}/{train_cfg.diag_batches} collected", flush=True)

    payload = {
        "variant": variant,
        "seq_len": seq_len,
        "recent_window": train_cfg.diag_recent_window,
        "omega_mean": torch.cat(omega_all, dim=0),
        "p_token0_mean": torch.cat(p0_all, dim=0),
        "p_max_mean": torch.cat(pmax_all, dim=0),
        "p_recent_window_mean": torch.cat(precent_all, dim=0),
        "n_points": torch.cat(omega_all, dim=0).numel(),
    }

    if is_rank0():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        print(f"[diag] saved: {out_path}", flush=True)



@torch.no_grad()
def extract_goat_bias_curve(
    model: nn.Module,
    *,
    seq_len: int,
    device: torch.device,
    out_path: Path,
) -> None:
    if not is_rank0():
        return

    m = unwrap_model(model)
    per_layer: List[torch.Tensor] = []

    for block in m.blocks:
        attn = getattr(block, "attn", None)
        if not isinstance(attn, CausalSelfAttentionGOAT):
            continue
        eot = attn.mha
        if getattr(eot, "key_bias", None) is None:
            raise RuntimeError("GOAT key_bias is disabled; cannot extract u(j).")

        abs_feats = eot._get_abs_feats(seq_len, dtype=torch.float32, device=device, offset=0)  # type: ignore
        abs_in = eot._augment_abs_features(abs_feats, pos_offset=0)  # (L, 2M+2) fp32
        u_shared = eot.key_bias(abs_in).squeeze(-1)  # (L,) fp32
        u = u_shared.detach().cpu()
        per_layer.append(u)

    if not per_layer:
        raise RuntimeError("No GOAT layers found to extract bias curve.")

    u_per_layer = torch.stack(per_layer, dim=0)
    u_h_per_layer: List[torch.Tensor] = []
    slopes_per_layer: List[torch.Tensor] = []
    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (L,)
    for block in m.blocks:
        attn = getattr(block, "attn", None)
        if not isinstance(attn, CausalSelfAttentionGOAT):
            continue
        eot = attn.mha
        if getattr(eot, "key_bias", None) is None or getattr(eot, "recency_slope_raw", None) is None:
            continue
        abs_feats = eot._get_abs_feats(seq_len, dtype=torch.float32, device=device, offset=0)  # type: ignore
        abs_in = eot._augment_abs_features(abs_feats, pos_offset=0)
        u_shared = eot.key_bias(abs_in).squeeze(-1)  # (L,)
        slopes = F.softplus(eot.recency_slope_raw.to(device=device, dtype=torch.float32))  # (H_kv,)
        u_h = u_shared.view(1, seq_len) + slopes.view(-1, 1) * t.view(1, seq_len)  # (H_kv, L)
        u_h_per_layer.append(u_h.detach().cpu())
        slopes_per_layer.append(slopes.detach().cpu())

    payload = {
        "positions": torch.arange(seq_len, dtype=torch.long),
        "u_per_layer": u_per_layer,
        "u_mean": u_per_layer.mean(dim=0),
        "u_h_per_layer": torch.stack(u_h_per_layer, dim=0) if u_h_per_layer else None,
        "slopes_per_layer": torch.stack(slopes_per_layer, dim=0) if slopes_per_layer else None,
        "seq_len": seq_len,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[goat] saved bias curve: {out_path}", flush=True)


def reinit_goat_priors(model: nn.Module) -> None:
    if GoatAttention is None:
        return
    for mod in model.modules():
        if isinstance(mod, GoatAttention):
            s = getattr(mod, "init_scale_prior", 1e-3)
            kb = getattr(mod, "key_bias", None)
            if kb is not None:
                fc1 = getattr(kb, "fc1", None)
                fc2 = getattr(kb, "fc2", None)
                if isinstance(fc1, nn.Linear):
                    nn.init.normal_(fc1.weight, mean=0.0, std=s)
                    if fc1.bias is not None:
                        nn.init.zeros_(fc1.bias)
                if isinstance(fc2, nn.Linear):
                    nn.init.normal_(fc2.weight, mean=0.0, std=s)
                    if fc2.bias is not None:
                        nn.init.zeros_(fc2.bias)
            if hasattr(mod, "lambda_asym"):
                try:
                    nn.init.zeros_(getattr(mod, "lambda_asym"))
                except Exception:
                    pass



def train_one_variant(
    *,
    variant: Literal["rope", "alibi", "goat"],
    output_dir: Path,
    run_name: str,
    train_cfg: TrainConfig,
    model_cfg: GPTConfig,
    rank: int,
    local_rank: int,
    world_size: int,
    run_posthoc: bool = True,
) -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    clear_alibi_bias_cache()
    seed_off = 0 if variant == "rope" else 17 if variant == "alibi" else 33
    set_seed(train_cfg.seed + seed_off + rank)

    if train_cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_rank0():
        save_json(output_dir / "config.json", {
            "run_name": run_name,
            "variant": variant,
            "train_cfg": dataclasses.asdict(train_cfg),
            "model_cfg": dataclasses.asdict(model_cfg),
            "world_size": world_size,
        })

    train_loader = make_dataloader(
        split="train",
        tokenizer_name=train_cfg.tokenizer_name,
        seq_len=train_cfg.seq_len_train,
        micro_batch_size=train_cfg.micro_batch_size,
        seed=train_cfg.seed,
        shuffle=True,
        shuffle_buffer=train_cfg.shuffle_buffer,
        hf_cache_dir=train_cfg.hf_cache_dir,
        tokenize_batch_size=train_cfg.tokenize_batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        rank=rank,
        world_size=world_size,
    )
    val_loader: Optional[DataLoader] = None
    val_iter: Optional[Iterator[torch.Tensor]] = None
    if is_rank0():
        val_loader = make_dataloader(
            split="validation",
            tokenizer_name=train_cfg.tokenizer_name,
            seq_len=train_cfg.seq_len_train,
            micro_batch_size=train_cfg.micro_batch_size,
            seed=train_cfg.seed + 123,
            shuffle=False,
            shuffle_buffer=train_cfg.shuffle_buffer,
            hf_cache_dir=train_cfg.hf_cache_dir,
            tokenize_batch_size=train_cfg.tokenize_batch_size,
            num_workers=0,
            pin_memory=train_cfg.pin_memory,
            rank=0,
            world_size=1,
        )
        val_iter = iter(val_loader)

    train_iter = iter(train_loader)

    def _eval_on_validation_iterator() -> float:
        nonlocal val_iter
        if not is_rank0():
            barrier()
            return float("nan")

        assert val_loader is not None
        assert val_iter is not None
        while True:
            try:
                val_loss = evaluate_loss(
                    model,
                    val_iter,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    eval_batches=train_cfg.eval_batches,
                    distributed_reduce=False,
                )
                barrier()
                return val_loss
            except StopIteration:
                val_iter = iter(val_loader)

    raw_model = GPT(model_cfg).to(device)
    if variant == "goat":
        reinit_goat_priors(raw_model)

    if is_rank0():
        print(f"[{variant}] model params: {count_parameters(raw_model)/1e6:.2f}M", flush=True)

    optimizer = build_optimizer(raw_model, train_cfg)
    scaler = _make_grad_scaler(train_cfg.precision)
    tokens_per_step = train_cfg.seq_len_train * train_cfg.micro_batch_size * train_cfg.grad_accum * world_size
    total_steps = train_cfg.max_steps if train_cfg.max_steps is not None else int(math.ceil(train_cfg.total_tokens / tokens_per_step))
    ckpt_latest = ckpt_dir / "ckpt_latest.pt"
    ckpt_best = ckpt_dir / "ckpt_best.pt"

    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    if train_cfg.resume and ckpt_latest.exists():
        map_location = "cpu" if device.type == "cpu" else f"cuda:{local_rank}"
        try:
            start_step, tokens_seen, best_val_loss = load_checkpoint(
                ckpt_latest,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                map_location=map_location,
            )
            if is_rank0():
                print(f"[{variant}] resumed from {ckpt_latest} step={start_step} tokens={tokens_seen} best_val={best_val_loss:.4f}", flush=True)
        except Exception as e:
            if is_rank0():
                print(f"[{variant}] resume failed ({e}); starting fresh.", flush=True)

    model: nn.Module = raw_model
    if world_size > 1:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        else:
            model = DDP(model, broadcast_buffers=False, find_unused_parameters=False)

    train_log_path = output_dir / "train_metrics.jsonl"
    eval_log_path = output_dir / "eval_metrics.jsonl"
    autocast_ctx = _autocast_context(train_cfg.precision, device=device)

    model.train()
    t0 = time.time()

    with JsonlLogger(train_log_path) as train_logger, JsonlLogger(eval_log_path) as eval_logger:
        if start_step == 0:
            try:
                val_loss = _eval_on_validation_iterator()
                if is_rank0():
                    eval_logger.log({
                        "step": 0,
                        "tokens_seen": tokens_seen,
                        "val_loss": val_loss,
                        "val_ppl": math.exp(val_loss),
                        "time_sec": time.time() - t0,
                    })
                    print(f"[{variant}] step=0 val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}", flush=True)
                    best_val_loss = min(best_val_loss, val_loss)
            except StopIteration:
                pass

        for step in range(start_step, total_steps):
            lr = lr_schedule(step, cfg=train_cfg, total_steps=total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            micro_losses: List[float] = []

            for micro in range(train_cfg.grad_accum):
                sync_ctx = model.no_sync() if (isinstance(model, DDP) and micro < train_cfg.grad_accum - 1) else contextlib.nullcontext()
                with sync_ctx:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    batch = batch.to(device, non_blocking=True)
                    x = batch[:, :-1]
                    y = batch[:, 1:]

                    with autocast_ctx:
                        _logits, loss = model(x, targets=y, return_attn_stats=False)
                    assert loss is not None
                    loss = loss / float(train_cfg.grad_accum)

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_losses.append(float(loss.item()) * float(train_cfg.grad_accum))

            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip))
                optimizer.step()

            tokens_seen += tokens_per_step

            if (step + 1) % train_cfg.log_interval == 0:
                loss_mean = np.mean(micro_losses) if micro_losses else float("nan")
                if is_rank0():
                    train_logger.log({
                        "step": step + 1,
                        "tokens_seen": tokens_seen,
                        "lr": lr,
                        "train_loss": loss_mean,
                        "grad_norm": grad_norm,
                        "time_sec": time.time() - t0,
                    })
                    print(f"[{variant}] step={step+1}/{total_steps} tokens={tokens_seen} loss={loss_mean:.4f} lr={lr:.2e}", flush=True)

            if (step + 1) % train_cfg.eval_interval == 0 or (step + 1) == total_steps:
                val_loss = _eval_on_validation_iterator()

                if is_rank0():
                    eval_logger.log({
                        "step": step + 1,
                        "tokens_seen": tokens_seen,
                        "val_loss": val_loss,
                        "val_ppl": math.exp(val_loss),
                        "time_sec": time.time() - t0,
                    })
                    print(f"[{variant}] EVAL step={step+1} val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}", flush=True)

                    if train_cfg.save_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            path=ckpt_best,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            step=step + 1,
                            tokens_seen=tokens_seen,
                            best_val_loss=best_val_loss,
                            train_cfg=train_cfg,
                            model_cfg=model_cfg,
                        )
                        print(f"[{variant}] saved BEST ckpt (val_loss={best_val_loss:.4f})", flush=True)

                    save_checkpoint(
                        path=ckpt_latest,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step + 1,
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        train_cfg=train_cfg,
                        model_cfg=model_cfg,
                    )

            if (step + 1) % train_cfg.save_interval == 0 and is_rank0():
                save_checkpoint(
                    path=ckpt_dir / f"ckpt_step_{step+1}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step + 1,
                    tokens_seen=tokens_seen,
                    best_val_loss=best_val_loss,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                )

    barrier()

    if is_rank0():
        print(f"[{variant}] training done. Running extrapolation + analysis...", flush=True)

    if run_posthoc:
        ppl_results: Dict[int, float] = {}
        if is_rank0():
            ppl_results = evaluate_ppl_by_length(
                model,
                lengths=list(train_cfg.extrap_lengths),
                train_cfg=train_cfg,
                model_device=device,
                rank=0,
                world_size=1,
            )
        barrier()
        if is_rank0():
            save_json(output_dir / "eval_extrapolation.json", {
                "variant": variant,
                "lengths": list(train_cfg.extrap_lengths),
                "ppl": {str(k): v for k, v in ppl_results.items()},
            })

        if variant == "goat":
            extract_goat_bias_curve(
                model,
                seq_len=train_cfg.seq_len_train,
                device=device,
                out_path=output_dir / "goat_bias_curve.pt",
            )

        if train_cfg.diag_batches > 0:
            collect_signal_vs_sink(
                model,
                train_cfg=train_cfg,
                seq_len=train_cfg.seq_len_train,
                device=device,
                rank=rank,
                world_size=world_size,
                variant=variant,
                out_path=output_dir / "signal_vs_sink.pt",
            )

    barrier()

    del model, raw_model, optimizer, scaler
    if variant == "alibi":
        clear_alibi_bias_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--run_name", type=str, default="c4_125m")
    p.add_argument("--variants", type=str, nargs="+", default=["rope", "alibi", "goat"], choices=["rope", "alibi", "goat"])

    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--seq_len_train", type=int, default=2048)
    p.add_argument("--total_tokens", type=int, default=4_000_000_000)
    p.add_argument("--micro_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=10_000)

    p.add_argument("--extrap_lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    p.add_argument("--extrap_eval_batches", type=int, default=100)
    p.add_argument("--extrap_batch_size", type=int, default=1)

    p.add_argument("--diag_layer", type=int, default=-1)
    p.add_argument("--diag_batches", type=int, default=8)
    p.add_argument("--diag_batch_size", type=int, default=1)
    p.add_argument("--diag_recent_window", type=int, default=128)

    p.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_tf32", action="store_true")

    p.add_argument("--shuffle_buffer", type=int, default=10_000)
    p.add_argument("--tokenize_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_pin_memory", action="store_true")

    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--rope_base", type=float, default=10000.0)

    p.add_argument("--goat_pos_rank", type=int, default=4)
    p.add_argument("--goat_abs_rank", type=int, default=4)
    p.add_argument("--goat_pos_base", type=float, default=10000.0)
    p.add_argument("--goat_abs_base", type=float, default=10000.0)
    p.add_argument("--goat_disable_key_bias", action="store_true")

    p.add_argument("--alibi_long_eval_threshold", type=int, default=4096)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()

def main() -> None:
    rank, local_rank, world_size = setup_distributed()
    args = parse_args()

    output_root = Path(args.output_root)
    (output_root / args.run_name).mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("transformers is required for --tokenizer_name (pip install transformers)") from e
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    try:
        tok.model_max_length = 10**9
    except Exception:
        pass
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token
    vocab_size = len(tok)

    train_cfg = TrainConfig(
        tokenizer_name=args.tokenizer_name,
        hf_cache_dir=args.hf_cache_dir,
        seq_len_train=args.seq_len_train,
        total_tokens=args.total_tokens,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        extrap_lengths=tuple(args.extrap_lengths),
        extrap_eval_batches=args.extrap_eval_batches,
        extrap_batch_size=args.extrap_batch_size,
        diag_layer=args.diag_layer,
        diag_batches=args.diag_batches,
        diag_batch_size=args.diag_batch_size,
        diag_recent_window=args.diag_recent_window,
        precision=args.precision,
        tf32=(not args.no_tf32),
        shuffle_buffer=args.shuffle_buffer,
        tokenize_batch_size=args.tokenize_batch_size,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        resume=(not args.no_resume),
        seed=args.seed,
    )

    if is_rank0():
        print(f"DDP world_size={world_size} rank={rank} local_rank={local_rank}", flush=True)
        print(f"Running variants: {args.variants}", flush=True)

    for variant in args.variants:
        variant_dir = output_root / args.run_name / variant

        model_cfg = GPTConfig(
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            rope_base=args.rope_base,
            goat_pos_rank=args.goat_pos_rank,
            goat_abs_rank=args.goat_abs_rank,
            goat_pos_base=args.goat_pos_base,
            goat_abs_base=args.goat_abs_base,
            goat_enable_key_bias=(not args.goat_disable_key_bias),
            goat_training_seq_len=args.seq_len_train,
            alibi_long_eval_threshold=args.alibi_long_eval_threshold,
            attn_variant=variant,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        if is_rank0():
            print("\n=============================", flush=True)
            print(f"TRAIN VARIANT: {variant}", flush=True)
            print(f"OUTPUT DIR: {variant_dir}", flush=True)
            print("=============================\n", flush=True)

        train_one_variant(
            variant=variant,
            output_dir=variant_dir,
            run_name=args.run_name,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            run_posthoc=True,
        )
        barrier()

    if is_rank0():
        print("All variants completed.", flush=True)

    cleanup_distributed()

if __name__ == "__main__":
    main()
