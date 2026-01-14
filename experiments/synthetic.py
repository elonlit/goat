#!/usr/bin/env python3
"""Synthetic benchmarks: Passkey Retrieval, NIAH"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, replace
import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm


def sdpa_kernel_cm():
    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    try:
        from torch.nn.attention import sdpa_kernel as _new_cm, SDPBackend
        return _new_cm(SDPBackend.FLASH_ATTENTION,
                       SDPBackend.EFFICIENT_ATTENTION,
                       SDPBackend.MATH)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel as _old_cm
            return _old_cm(enable_flash=True,
                           enable_math=True,
                           enable_mem_efficient=True)
        except Exception:
            return contextlib.nullcontext()

from goat import GoatAttention

def all_cuda_devices() -> list[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model, device_ids=all_cuda_devices())
    return model


def core_module(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m



def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class CharTokenizer:
    def __init__(self):
        base = list("abcdefghijklmnopqrstuvwxyz0123456789 :;,.?+-_=/'\"[](){}<>#@!$%^&*|\\")
        specials = ["<pad>", "<bos>", "<eos>"]
        self.vocab = specials + base
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, s: str, add_special: bool = False) -> List[int]:
        s = s.lower()
        toks = [self.stoi.get(ch, self.stoi[" "]) for ch in s]
        if add_special:
            return [self.bos_id] + toks + [self.eos_id]
        return toks

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)


@dataclass
class PasskeySample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_span: Tuple[int, int]


def make_passkey_sequence(tok: CharTokenizer, L: int, key_len: int = 8, rng: np.random.RandomState | None = None) -> PasskeySample:
    if rng is None:
        rng = np.random.RandomState()

    key = "".join(rng.choice(list("0123456789"), size=key_len))
    preamble = " system log: "
    question = " q: what is the passkey? a: "

    fixed_prefix = preamble + "passkey: " + key + " "
    fixed_suffix = question + key
    fixed_prefix_tokens = len(tok.encode(fixed_prefix, add_special=False))
    fixed_suffix_tokens = len(tok.encode(fixed_suffix, add_special=False))
    reserved_tokens = fixed_prefix_tokens + fixed_suffix_tokens

    filler_chars = list("abcdefghijklmnopqrstuvwxyz ")
    target_filler_tokens = max(0, L - reserved_tokens)
    filler_len = target_filler_tokens
    filler = "".join(rng.choice(filler_chars, size=filler_len)) if filler_len > 0 else ""
    insert_pos = int(rng.uniform(0, 0.8) * len(filler)) if filler_len > 0 else 0
    content = filler[:insert_pos] + "passkey: " + key + " " + filler[insert_pos:]

    seq = preamble + content + question + key
    ids = tok.encode(seq, add_special=False)

    if len(ids) > L:
        ids = ids[-(L):]

    inp = torch.tensor(ids[:-1], dtype=torch.long)
    tgt = torch.tensor(ids[1:], dtype=torch.long)

    start = len(ids) - key_len - 1
    span = (start, start + key_len)
    return PasskeySample(inp, tgt, span)


@dataclass
class NIAHSample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_span: Tuple[int, int]


def make_niah_sequence(
    tok: CharTokenizer,
    L: int,
    needle_len: int = 12,
    depth_frac: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> NIAHSample:
    if rng is None:
        rng = np.random.RandomState()

    alphabet = list("abcd")
    filler_chars = alphabet + [" "]
    needle = "".join(rng.choice(alphabet, size=needle_len))
    pre = " doc: "
    needle_hdr = "needle: "
    sep_after = " "
    post_q = " q: what is the needle? a: "
    pre_T = len(tok.encode(pre, add_special=False))
    hdr_T = len(tok.encode(needle_hdr, add_special=False))
    needle_T = len(tok.encode(needle, add_special=False))
    sep_T = len(tok.encode(sep_after, add_special=False))
    suffix_T = len(tok.encode(post_q + needle, add_special=False))

    max_front = max(0, L - suffix_T)
    target_start = int(depth_frac * max_front)
    target_start = max(pre_T, min(target_start, max_front))
    left_T = max(0, target_start - pre_T)
    core_T = hdr_T + needle_T + sep_T
    right_T = L - (pre_T + left_T + core_T + suffix_T)
    if right_T < 0:
        shift = min(left_T, -right_T)
        left_T -= shift
        right_T += shift
        target_start -= shift
        if right_T < 0:
            right_T = 0
            left_T = max(0, L - (pre_T + core_T + suffix_T))
            target_start = pre_T

    left_fill = "".join(rng.choice(filler_chars, size=left_T)) if left_T > 0 else ""
    right_fill = "".join(rng.choice(filler_chars, size=right_T)) if right_T > 0 else ""

    seq = pre + left_fill + needle_hdr + needle + sep_after + right_fill + post_q + needle
    ids = tok.encode(seq, add_special=False)

    if len(ids) > L:
        ids = ids[-L:]
    elif len(ids) < L:
        pad_tok = tok.stoi.get(" ", tok.pad_id)
        ids = [pad_tok] * (L - len(ids)) + ids

    inp = torch.tensor(ids[:-1], dtype=torch.long)
    tgt = torch.tensor(ids[1:], dtype=torch.long)
    start = len(ids) - needle_len - 1
    span = (start, start + needle_len)
    return NIAHSample(inp, tgt, span)


def build_alibi_slopes(n_heads: int) -> torch.Tensor:
    def _get_slopes_power_of_2(n: int):
        start = 2.0 ** (-2.0 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if (math.log2(n_heads)).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        m = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(m)
        slopes += build_alibi_slopes(2 * m).tolist()[0::2][: n_heads - m]
    return torch.tensor(slopes, dtype=torch.float32)


def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, T, D = q.shape
    half = D // 2
    cos = freqs[..., :half].unsqueeze(0).unsqueeze(0)
    sin = freqs[..., half:].unsqueeze(0).unsqueeze(0)

    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]

    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    q_rot[..., ::2] = q_even * cos - q_odd * sin
    q_rot[..., 1::2] = q_even * sin + q_odd * cos
    k_rot[..., ::2] = k_even * cos - k_odd * sin
    k_rot[..., 1::2] = k_even * sin + k_odd * cos
    return q_rot, k_rot


def build_rope_cache(T: int, dim: int, base: float = 10000.0, device=None, dtype=None, pos_scale: float = 1.0) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    theta = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    t = (torch.arange(T, dtype=torch.float32) * float(pos_scale))
    freqs = torch.einsum("t,d->td", t, theta)  # (T, half)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(device=device, dtype=dtype)


class SDPAWithRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, rope_base: float=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor, is_causal=True):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        freqs = build_rope_cache(T, D, base=self.rope_base, device=x.device, dtype=x.dtype)
        q, k = apply_rope(q, k, freqs)

        q = q.reshape(B*H, T, D) * math.sqrt(D)  # cancel internal scaling
        k = k.reshape(B*H, T, D)
        v = v.reshape(B*H, T, D)

        cm = sdpa_kernel_cm()
        with cm:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)  # (B*H,T,D)
        y = y.view(B, H, T, D).transpose(1, 2).reshape(B, T, E)
        return self.out_proj(y)


class SDPAWithRoPEPI(SDPAWithRoPE):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, rope_base: float = 10000.0, train_max_len: int = 4096):
        super().__init__(embed_dim, num_heads, dropout=dropout, rope_base=rope_base)
        self.train_max_len = train_max_len

    def forward(self, x: torch.Tensor, is_causal=True):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        if T > self.train_max_len:
            pos_scale = (self.train_max_len - 1) / max(1, (T - 1))
        else:
            pos_scale = 1.0

        freqs = build_rope_cache(T, D, base=self.rope_base, device=x.device, dtype=x.dtype, pos_scale=pos_scale)
        q, k = apply_rope(q, k, freqs)

        q = q.reshape(B * H, T, D) * math.sqrt(D)
        k = k.reshape(B * H, T, D)
        v = v.reshape(B * H, T, D)

        cm = sdpa_kernel_cm()
        with cm:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
            )
        y = y.view(B, H, T, D).transpose(1, 2).reshape(B, T, E)
        return self.out_proj(y)


class SDPAWithALiBi(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.register_buffer("slopes", build_alibi_slopes(num_heads), persistent=False)

    def forward(self, x: torch.Tensor, is_causal=True):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        i32 = torch.arange(T, device=x.device, dtype=torch.float32)
        j32 = torch.arange(T, device=x.device, dtype=torch.float32)
        slopes32 = self.slopes.to(x.device, torch.float32).view(1, H, 1)

        q_lane0 = -(slopes32 * i32.view(1, 1, T))
        q_lane1 = torch.ones((1, H, T), dtype=torch.float32, device=x.device)
        k_lane0 = torch.ones((1, H, T), dtype=torch.float32, device=x.device)
        k_lane1 = slopes32 * j32.view(1, 1, T)

        q_extra = torch.stack([q_lane0, q_lane1], dim=-1).to(x.dtype).expand(B, -1, -1, -1)
        k_extra = torch.stack([k_lane0, k_lane1], dim=-1).to(x.dtype).expand(B, -1, -1, -1)
        v_extra = torch.zeros((B, H, T, 2), dtype=x.dtype, device=x.device)

        q_total = torch.cat([q, q_extra], dim=-1)  # (B,H,T,D+2)
        k_total = torch.cat([k, k_extra], dim=-1)
        v_total = torch.cat([v, v_extra], dim=-1)
        D_total = D + 2

        q_bh = (q_total * math.sqrt(D_total)).reshape(B * H, T, D_total)
        k_bh = k_total.reshape(B * H, T, D_total)
        v_bh = v_total.reshape(B * H, T, D_total)

        cm = sdpa_kernel_cm()
        with cm:
            y = F.scaled_dot_product_attention(
                q_bh, k_bh, v_bh,
                dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        y = y.view(B, H, T, D_total)[..., :D]
        y = y.transpose(1, 2).reshape(B, T, E)
        return self.out_proj(y)


class SDPAPlain(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, is_causal=True):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = q.reshape(B*H, T, D) * math.sqrt(D)
        k = k.reshape(B*H, T, D)
        v = v.reshape(B*H, T, D)
        cm = sdpa_kernel_cm()
        with cm:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        y = y.view(B, H, T, D).transpose(1, 2).reshape(B, T, E)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, attn_type: str, max_len: int, dropout: float=0.0, checkpoint: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_type = attn_type
        self.checkpoint = checkpoint

        if attn_type == "rope":
            self.attn = SDPAWithRoPE(d_model, n_heads, dropout=dropout)
            self.pos_emb = None
        elif attn_type == "rope_pi":
            self.attn = SDPAWithRoPEPI(d_model, n_heads, dropout=dropout, rope_base=10000.0, train_max_len=max_len)
            self.pos_emb = None
        elif attn_type == "alibi":
            self.attn = SDPAWithALiBi(d_model, n_heads, dropout=dropout)
            self.pos_emb = None
        elif attn_type == "sinus":
            self.attn = SDPAPlain(d_model, n_heads, dropout=dropout)
            self.pos_emb = None
        elif attn_type == "goat":
            self.attn = GoatAttention.for_gpt(
                embed_dim=d_model,
                num_heads=n_heads,
                kv_num_heads=None,
                dropout=dropout,
                pos_rank=8,
                abs_rank=8,
                enable_key_bias=True,
                init_scale_prior=3e-3,
            )
            self.pos_emb = None
        else:
            raise ValueError(f"Unknown attn_type={attn_type}")

        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _attn_forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.attn_type == "goat":
            out = self.attn(h, h, h, is_causal=True, need_weights=False)
            return out[0] if isinstance(out, tuple) else out
        else:
            return self.attn(h, is_causal=True)

    def forward(self, x: torch.Tensor, pos_ids: Optional[torch.Tensor] = None):
        h = self.ln1(x)
        if self.checkpoint and self.training:
            y = torch.utils.checkpoint.checkpoint(self._attn_forward, h, use_reentrant=False)
        else:
            y = self._attn_forward(h)
        x = x + self.drop(y)
        h2 = self.ln2(x)
        if self.checkpoint and self.training:
            h2 = torch.utils.checkpoint.checkpoint(lambda _h2: self.mlp(_h2), h2, use_reentrant=False)
        else:
            h2 = self.mlp(h2)
        x = x + self.drop(h2)
        return x


class CausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, mlp_ratio: float, attn_type: str, max_len: int, dropout: float=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.attn_type = attn_type
        self.max_len = max_len
        self.pos_emb = None
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, attn_type, max_len, dropout=dropout, checkpoint=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.attn_type == "sinus":
            _, _, E = x.shape
            pos = build_rope_cache(T, E, base=10000.0, device=x.device, dtype=x.dtype)
            x = x + pos.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


@dataclass
class TrainConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 400
    max_steps: int = 2000
    grad_clip: float = 1.0
    train_context_len: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    checkpoint_dir: Optional[str] = "checkpoints"
    load_checkpoints: bool = True
    save_checkpoints: bool = True
    span_loss_only: bool = False
    use_amp: Optional[bool] = None
    use_data_parallel: Optional[bool] = None


def make_optimizer(model: nn.Module, cfg: TrainConfig):
    fused_ok = torch.cuda.is_available()
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=fused_ok,   # silently ignored if unsupported
    )


def _config_signature(attn_type: str, tok: CharTokenizer, cfg: TrainConfig, task: str = "passkey") -> Dict[str, object]:
    return {
        "task": task,
        "attn_type": attn_type,
        "vocab_size": len(tok.vocab),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "mlp_ratio": cfg.mlp_ratio,
        "dropout": cfg.dropout,
        "train_context_len": cfg.train_context_len,
        "span_loss_only": cfg.span_loss_only,
        "seed": cfg.seed,
    }


def _checkpoint_path(cfg: TrainConfig, sig: Dict[str, object]) -> Optional[str]:
    if cfg.checkpoint_dir is None:
        return None
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    fname = (
        f"{sig.get('task', 'passkey')}_{sig['attn_type']}"
        f"_d{sig['d_model']}_L{sig['n_layers']}_H{sig['n_heads']}"
        f"_ctx{sig['train_context_len']}_spanonly{int(bool(sig['span_loss_only']))}"
        f"_seed{sig['seed']}.pt"
    )
    return os.path.join(cfg.checkpoint_dir, fname)


def cosine_schedule(step: int, max_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step+1) / warmup
    t = (step - warmup) / max(1, (max_steps - warmup))
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def passkey_batch(tok: CharTokenizer, B: int, L: int, key_len: int, rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]]]:
    xs, ys, spans = [], [], []
    for _ in range(B):
        s = make_passkey_sequence(tok, L=L, key_len=key_len, rng=rng)
        xs.append(s.input_ids)
        ys.append(s.target_ids)
        spans.append(s.answer_span)
    x = torch.stack(xs, dim=0)  # (B, L-1)
    y = torch.stack(ys, dim=0)  # (B, L-1)
    return x, y, spans


def niah_batch(tok: CharTokenizer, B: int, L: int, needle_len: int, depth_frac: float, rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]]]:
    xs, ys, spans = [], [], []
    for _ in range(B):
        s = make_niah_sequence(tok, L=L, needle_len=needle_len, depth_frac=depth_frac, rng=rng)
        xs.append(s.input_ids)
        ys.append(s.target_ids)
        spans.append(s.answer_span)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y, spans


@torch.no_grad()
def retrieval_accuracy(model: nn.Module, tok: CharTokenizer, x: torch.Tensor, y: torch.Tensor, spans: List[Tuple[int,int]]) -> float:
    model.eval()
    logits = model(x)
    pred = logits.argmax(dim=-1)  # (B, T)
    B = x.size(0)
    ok = 0
    for b in range(B):
        s, e = spans[b]
        if s < pred.size(1) and e <= pred.size(1):
            if torch.equal(pred[b, s:e], y[b, s:e]):
                ok += 1
    return ok / B


@torch.no_grad()
def retrieval_scores(model: nn.Module, x: torch.Tensor, y: torch.Tensor, spans: List[Tuple[int,int]]) -> Dict[str, float]:
    model.eval()
    logits = model(x)                # (B,T,V)
    logp = logits.log_softmax(-1)
    pred = logits.argmax(-1)
    B = x.size(0)
    strict, tok_hits, tok_total, avg_lp = 0, 0, 0, 0.0
    for b in range(B):
        s, e = spans[b]
        if s < pred.size(1) and e <= pred.size(1):
            strict += int(torch.equal(pred[b, s:e], y[b, s:e]))
            tok_hits += (pred[b, s:e] == y[b, s:e]).sum().item()
            tok_total += (e - s)
            avg_lp += logp[b, torch.arange(s, e, device=x.device), y[b, s:e]].mean().item()
    return {
        "strict_acc": strict / B if B > 0 else 0.0,
        "span_tok_acc": (tok_hits / tok_total) if tok_total else 0.0,
        "span_avg_logp": (avg_lp / B) if B > 0 else 0.0,
    }


def train_model(attn_type: str, tok: CharTokenizer, cfg: TrainConfig, deterministic: bool = False) -> nn.Module:
    set_seed(cfg.seed, deterministic=deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else cfg.device)
    model = CausalLM(
        vocab_size=len(tok.vocab),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        attn_type=attn_type,
        max_len=cfg.train_context_len,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.use_data_parallel is False:
        model = model.to(device)
    elif cfg.use_data_parallel is True or (cfg.use_data_parallel is None and torch.cuda.device_count() > 1):
        model = maybe_data_parallel(model)
    else:
        model = model.to(device)

    opt = make_optimizer(model, cfg)
    sig = _config_signature(attn_type, tok, cfg, task="passkey")
    ckpt_path = _checkpoint_path(cfg, sig)
    loaded_from_ckpt = False
    if cfg.load_checkpoints and ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and ckpt.get("signature") == sig and "model" in ckpt:
                base = model.module if isinstance(model, nn.DataParallel) else model
                base.load_state_dict(ckpt["model"])
                loaded_from_ckpt = True
                print(f"[checkpoint] Loaded checkpoint from {ckpt_path}")
            else:
                print(f"[checkpoint] Signature mismatch at {ckpt_path}")
        except Exception as e:
            print(f"[checkpoint] Load failed: {e}")

    if loaded_from_ckpt:
        model.eval()
        return model

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_amp = cfg.use_amp if cfg.use_amp is not None else torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    rng = np.random.RandomState(cfg.seed)

    key_len = 8
    pbar = tqdm(range(cfg.max_steps), desc=f"train[{attn_type}]")
    target_bsz = cfg.batch_size
    cur_bsz = target_bsz
    model.train()

    for step in pbar:
        lr = cosine_schedule(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        attempt_done = False
        while not attempt_done:
            try:
                x, y, spans = passkey_batch(tok, B=cur_bsz, L=cfg.train_context_len + 1, key_len=key_len, rng=rng)
                x, y = x.to(device), y.to(device)

                opt.zero_grad(set_to_none=True)
                autocast_ctx = autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()

                with autocast_ctx:
                    logits = model(x)  # (B,T,V)
                    per_tok = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
                    mask = torch.zeros_like(per_tok)
                    for b, (s, e) in enumerate(spans):
                        if s < per_tok.size(1) and e <= per_tok.size(1):
                            mask[b, s:e] = 1.0

                    global_loss = per_tok.mean()
                    span_loss = (per_tok * mask).sum() / (mask.sum() + 1e-8)

                    if cfg.span_loss_only:
                        loss = span_loss
                    else:
                        alpha = 10.0
                        loss = (global_loss + alpha * span_loss) / (1.0 + alpha)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()

                if (step + 1) % 10 == 0:
                    pbar.set_postfix(loss=float(loss.detach().cpu()), bsz=cur_bsz, lr=lr)
                if (step + 1) % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_B = min(max(4, cur_bsz), 32)
                        eval_x, eval_y, eval_spans = passkey_batch(
                            tok,
                            B=eval_B,
                            L=cfg.train_context_len + 1,
                            key_len=key_len,
                            rng=rng,
                        )
                        eval_x, eval_y = eval_x.to(device), eval_y.to(device)
                        scores = retrieval_scores(model, eval_x, eval_y, eval_spans)
                        pbar.write(f"Step {step+1}: strict={scores['strict_acc']:.3f}, tok_acc={scores['span_tok_acc']:.3f}, logp={scores['span_avg_logp']:.2f}")
                    model.train()
                attempt_done = True

            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if cur_bsz > 1:
                    cur_bsz = max(1, cur_bsz // 2)
                    print(f"[OOM] Reducing train batch size to {cur_bsz} and retrying this step.")
                    continue
                else:
                    raise

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if cur_bsz > 1:
                        cur_bsz = max(1, cur_bsz // 2)
                        print(f"[OOM] Reducing train batch size to {cur_bsz} and retrying this step.")
                        continue
                raise

    if cfg.save_checkpoints and ckpt_path is not None:
        try:
            base = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"signature": sig, "model": base.state_dict()}, ckpt_path)
            print(f"[checkpoint] Saved checkpoint to {ckpt_path}")
        except Exception as e:
            print(f"[checkpoint] Failed to save checkpoint to {ckpt_path}: {e}")

    model.eval()
    return model


def train_model_niah(attn_type: str, tok: CharTokenizer, cfg: TrainConfig, deterministic: bool = False) -> nn.Module:
    set_seed(cfg.seed, deterministic=deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else cfg.device)
    model = CausalLM(
        vocab_size=len(tok.vocab),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        attn_type=attn_type,
        max_len=cfg.train_context_len,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.use_data_parallel is False:
        model = model.to(device)
    elif cfg.use_data_parallel is True or (cfg.use_data_parallel is None and torch.cuda.device_count() > 1):
        model = maybe_data_parallel(model)
    else:
        model = model.to(device)

    opt = make_optimizer(model, cfg)
    sig = _config_signature(attn_type, tok, cfg, task="niah")
    ckpt_path = _checkpoint_path(cfg, sig)
    loaded_from_ckpt = False
    if cfg.load_checkpoints and ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and ckpt.get("signature") == sig and "model" in ckpt:
                base = model.module if isinstance(model, nn.DataParallel) else model
                base.load_state_dict(ckpt["model"])
                loaded_from_ckpt = True
                print(f"[niah checkpoint] Loaded checkpoint from {ckpt_path}")
            else:
                print(f"[niah checkpoint] Ignoring checkpoint at {ckpt_path} (signature mismatch).")
        except Exception as e:
            print(f"[niah checkpoint] Failed to load checkpoint at {ckpt_path}: {e}")

    if loaded_from_ckpt:
        model.eval()
        return model

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_amp = cfg.use_amp if cfg.use_amp is not None else torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    rng = np.random.RandomState(cfg.seed + 10)
    needle_len = 12
    pbar = tqdm(range(cfg.max_steps), desc=f"train_niah[{attn_type}]")
    target_bsz = cfg.batch_size
    cur_bsz = target_bsz
    model.train()

    for step in pbar:
        lr = cosine_schedule(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        attempt_done = False
        while not attempt_done:
            try:
                depth = float(rng.uniform(0.1, 0.9))
                x, y, spans = niah_batch(
                    tok,
                    B=cur_bsz,
                    L=cfg.train_context_len + 1,
                    needle_len=needle_len,
                    depth_frac=depth,
                    rng=rng,
                )
                x, y = x.to(device), y.to(device)

                opt.zero_grad(set_to_none=True)
                autocast_ctx = autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()

                with autocast_ctx:
                    logits = model(x)  # (B,T,V)
                    per_tok = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
                    mask = torch.zeros_like(per_tok)
                    for b, (s, e) in enumerate(spans):
                        if s < per_tok.size(1) and e <= per_tok.size(1):
                            mask[b, s:e] = 1.0

                    global_loss = per_tok.mean()
                    span_loss = (per_tok * mask).sum() / (mask.sum() + 1e-8)

                    if cfg.span_loss_only:
                        loss = span_loss
                    else:
                        alpha = 10.0
                        loss = (global_loss + alpha * span_loss) / (1.0 + alpha)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()

                if (step + 1) % 10 == 0:
                    pbar.set_postfix(loss=float(loss.detach().cpu()), bsz=cur_bsz, lr=lr)
                if (step + 1) % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_B = min(max(4, cur_bsz), 32)
                        depth_eval = float(rng.uniform(0.1, 0.9))
                        eval_x, eval_y, eval_spans = niah_batch(
                            tok,
                            B=eval_B,
                            L=cfg.train_context_len + 1,
                            needle_len=needle_len,
                            depth_frac=depth_eval,
                            rng=rng,
                        )
                        eval_x, eval_y = eval_x.to(device), eval_y.to(device)
                        scores = retrieval_scores(model, eval_x, eval_y, eval_spans)
                        pbar.write(
                            f"[NIAH] Step {step+1}: strict={scores['strict_acc']:.3f}, "
                            f"tok_acc={scores['span_tok_acc']:.3f}, logp={scores['span_avg_logp']:.2f}"
                        )
                    model.train()
                attempt_done = True

            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if cur_bsz > 1:
                    cur_bsz = max(1, cur_bsz // 2)
                    print(f"[OOM] Reducing NIAH train batch size to {cur_bsz} and retrying this step.")
                    continue
                else:
                    raise

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if cur_bsz > 1:
                        cur_bsz = max(1, cur_bsz // 2)
                        print(f"[OOM] Reducing NIAH train batch size to {cur_bsz} and retrying this step.")
                        continue
                raise

    model.eval()
    if cfg.save_checkpoints and ckpt_path is not None:
        try:
            base = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({"signature": sig, "model": base.state_dict()}, ckpt_path)
            print(f"[niah checkpoint] Saved checkpoint to {ckpt_path}")
        except Exception as e:
            print(f"[niah checkpoint] Failed to save checkpoint to {ckpt_path}: {e}")

    return model


def nanmean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    a = np.asarray(values, dtype=np.float64)
    return float(np.nan) if np.isnan(a).all() else float(np.nanmean(a))


def safe_retrieval_accuracy(model: nn.Module, tok: CharTokenizer, x: torch.Tensor, y: torch.Tensor, spans: List[Tuple[int,int]]) -> float:
    return retrieval_accuracy(model, tok, x, y, spans)


def eval_passkey_across_lengths(
    model_dict: Dict[str, nn.Module],
    tok: CharTokenizer,
    lengths: List[int],
    key_len: int,
    batch_size: int,
    n_batches: int,
    device: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    rng = np.random.RandomState(2025)
    for L in tqdm(lengths, desc="eval_passkey"):
        cpu_batches = [
            passkey_batch(tok, B=batch_size, L=L, key_len=key_len, rng=rng)
            for _ in range(n_batches)
        ]
        for name, m in model_dict.items():
            m.to(device)
            m.eval()
            m_base = m.module if isinstance(m, nn.DataParallel) else m
            max_len = getattr(m_base, "max_len", None)
            attn_type = getattr(m_base, "attn_type", None)
            cur_eval_bs = batch_size
            strict_accs: List[float] = []
            tok_accs: List[float] = []
            logps: List[float] = []
            for xb, yb, spans_b in cpu_batches:
                start = 0
                while start < xb.size(0):
                    this_bs = min(cur_eval_bs, xb.size(0) - start)
                    try:
                        x = xb[start:start+this_bs].to(device, non_blocking=True)
                        y = yb[start:start+this_bs].to(device, non_blocking=True)
                        spans_slice = spans_b[start:start+this_bs]
                        scores = retrieval_scores(m, x, y, spans_slice)
                        strict_accs.append(scores["strict_acc"])
                        tok_accs.append(scores["span_tok_acc"])
                        logps.append(scores["span_avg_logp"])
                        start += this_bs
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        msg = str(e).lower()
                        if "out of memory" in msg and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if this_bs > 1:
                                cur_eval_bs = max(1, this_bs // 2)
                                continue  # retry this slice with smaller bs
                            else:
                                strict_accs.append(float("nan"))
                                tok_accs.append(float("nan"))
                                logps.append(float("nan"))
                                start += 1  # make forward progress
                        else:
                            raise
            rows.append({
                "variant": name,
                "length": L,
                "accuracy": nanmean_or_nan(strict_accs),
                "strict_acc": nanmean_or_nan(strict_accs),
                "span_tok_acc": nanmean_or_nan(tok_accs),
                "span_avg_logp": nanmean_or_nan(logps),
            })
            m.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def eval_niah_grid(model_dict: Dict[str, nn.Module], tok: CharTokenizer,
                   lengths: List[int], depths: List[float], needle_len: int,
                   batch_size: int, n_batches: int, device: str) -> Dict[str, np.ndarray]:
    results = {name: np.zeros((len(depths), len(lengths)), dtype=np.float32) for name in model_dict}
    rng = np.random.RandomState(777)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    for j, L in enumerate(tqdm(lengths, desc="eval_niah_lengths")):
        for i, d in enumerate(depths):
            cpu_batches: list[Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]]]] = []
            for _ in range(n_batches):
                xb, yb, sp = niah_batch(tok, B=batch_size, L=L, needle_len=needle_len, depth_frac=d, rng=rng)
                cpu_batches.append((xb, yb, sp))

            for name, m in model_dict.items():
                m.to(device)
                m.eval()
                cur_eval_bs = batch_size
                tok_accs: list[float] = []
                m_base = m.module if isinstance(m, nn.DataParallel) else m
                max_len = getattr(m_base, "max_len", None)
                attn_type = getattr(m_base, "attn_type", None)
                for xb, yb, sp in cpu_batches:
                    start = 0
                    while start < xb.size(0):
                        this_bs = min(cur_eval_bs, xb.size(0) - start)
                        try:
                            x = xb[start:start+this_bs].to(device, non_blocking=True)
                            y = yb[start:start+this_bs].to(device, non_blocking=True)
                            spans_slice = sp[start:start+this_bs]
                            scores = retrieval_scores(m, x, y, spans_slice)
                            tok_accs.append(scores["span_tok_acc"])
                            start += this_bs
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            msg = str(e).lower()
                            if "out of memory" in msg and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                if this_bs > 1:
                                    cur_eval_bs = max(1, this_bs // 2)
                                    continue  # retry this slice with smaller bs
                                else:
                                    tok_accs.append(float("nan"))
                                    start += 1
                            else:
                                raise

                results[name][i, j] = nanmean_or_nan(tok_accs) if tok_accs else float("nan")
                m.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


def save_passkey_results(df: pd.DataFrame, train_L: int, out_dir: str = "."):
    """Save passkey retrieval results to JSON and CSV."""
    import json

    df_eval = df[df["length"] >= train_L].copy()

    # Save full results as CSV
    csv_path = os.path.join(out_dir, "passkey_results.csv")
    df_eval.to_csv(csv_path, index=False)

    # Save structured JSON
    json_path = os.path.join(out_dir, "passkey_results.json")
    results = {"train_context_len": train_L, "variants": {}}
    for variant in df_eval["variant"].unique():
        sub = df_eval[df_eval["variant"] == variant]
        results["variants"][variant] = {
            "lengths": sub["length"].tolist(),
            "accuracy": sub["accuracy"].tolist(),
            "strict_acc": sub["strict_acc"].tolist(),
            "span_tok_acc": sub["span_tok_acc"].tolist(),
            "span_avg_logp": sub["span_avg_logp"].tolist(),
        }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def save_niah_results(grids: Dict[str, np.ndarray], lengths: List[int], depths: List[float], out_dir: str = "."):
    """Save NIAH evaluation results to NPZ."""
    npz_data = {"lengths": np.array(lengths), "depths": np.array(depths)}
    for key, grid in grids.items():
        if grid is not None and not np.isnan(grid).all():
            npz_data[f"{key}_grid"] = grid

    npz_path = os.path.join(out_dir, "niah_results.npz")
    np.savez(npz_path, **npz_data)


@torch.no_grad()
def save_goat_prior(model: nn.Module, L: int, out_dir: str, device: str):
    """Extract and save GOAT prior data to NPZ."""
    root = core_module(model)
    if not any(isinstance(m, GoatAttention) for m in root.modules()):
        raise RuntimeError("GOAT attention not found in model")

    goat_attn = None
    for b in root.blocks:
        if isinstance(b.attn, GoatAttention):
            goat_attn = b.attn
            break
    if goat_attn is None:
        raise RuntimeError("No GoatAttention block found")

    m = goat_attn
    dtype = torch.float32
    mod_dev = next(m.parameters()).device
    req_dev = torch.device(device)
    m_moved = False
    try:
        if req_dev.type != mod_dev.type or req_dev != mod_dev:
            m.to(req_dev)
            m_moved = True
        dev = req_dev if m_moved else mod_dev

        if hasattr(m, "compute_log_prior"):
            log_prior = m.compute_log_prior(L, dev, dtype)
        elif hasattr(m, "envelope"):
            psi_q = m._get_rel_feats(L, dtype, dev, offset=0)
            psi_k = m._get_rel_feats(L, dtype, dev, offset=0)
            abs_q = m._get_abs_feats(L, dtype, dev, offset=0)
            abs_k = m._get_abs_feats(L, dtype, dev, offset=0)

            if isinstance(m.envelope, nn.ModuleList):
                e_shared = torch.stack(
                    [m.envelope[h](abs_q.to(torch.float32)).to(dtype) for h in range(m.num_heads)],
                    dim=0,
                )
            else:
                e_shared = m.envelope(abs_q.to(torch.float32)).to(dtype)
                e_shared = e_shared.unsqueeze(0).expand(m.num_heads, L, -1)

            phi_nb = e_shared * psi_q.unsqueeze(0)
            psi_nb = psi_k.unsqueeze(0).expand(m.kv_num_heads, L, -1)

            if m.enable_polynomial:
                t = torch.arange(L, device=dev, dtype=dtype)
                phi_poly = torch.stack((-t, torch.ones_like(t)), dim=-1)
                psi_poly = torch.stack((torch.ones_like(t), t), dim=-1)
                phi_poly = phi_poly.unsqueeze(0).expand(m.num_heads, L, 2)
                psi_poly = psi_poly.unsqueeze(0).expand(m.kv_num_heads, L, 2)
                phi_nb = torch.cat([phi_nb, phi_poly], dim=-1)
                psi_nb = torch.cat([psi_nb, psi_poly], dim=-1)

            Dp = phi_nb.size(-1)
            sqrt_tau = math.sqrt(m.prior_tau)
            gamma = torch.sigmoid(m.raw_gamma.to(dev)).to(dtype).view(m.num_heads, 1, 1)

            if m.mixer_type == "full":
                A = m.A[:, :Dp, :Dp].to(device=dev, dtype=dtype)
                B = m.B[:, :Dp, :Dp].to(device=dev, dtype=dtype)
                phi_nb = torch.einsum("hld,hdf->hlf", phi_nb, A)
                psi_nb = torch.einsum("hld,hdf->hlf", psi_nb, B)
            else:
                Ad = m.A_diag[..., :Dp].to(device=dev, dtype=dtype).view(m.num_heads, 1, Dp)
                Bd = m.B_diag[..., :Dp].to(device=dev, dtype=dtype).view(m.kv_num_heads, 1, Dp)
                phi_nb = phi_nb * Ad
                psi_nb = psi_nb * Bd

            phi_nb = phi_nb * (sqrt_tau * gamma)
            psi_nb = psi_nb * sqrt_tau

            if m.enable_key_bias:
                u = m.key_bias(abs_k.to(torch.float32)).to(dtype).squeeze(-1)
            else:
                u = torch.zeros(L, dtype=dtype, device=dev)

            if m.kv_num_heads != m.num_heads:
                group = m.num_heads // m.kv_num_heads
                idx = (torch.arange(m.num_heads, device=dev) // group).to(torch.long)
                psi_nb = psi_nb.index_select(0, idx)

            psi_T = psi_nb.transpose(1, 2).contiguous()
            log_prior_heads = torch.matmul(phi_nb, psi_T) + m.prior_tau * u.view(1, 1, L)
            log_prior = log_prior_heads.mean(dim=0)
        else:
            raise RuntimeError("No compatible prior interface found")

        log_prior_np = log_prior.cpu().numpy()

        npz_path = os.path.join(out_dir, "goat_prior.npz")
        np.savez(npz_path, log_prior=log_prior_np, L=log_prior_np.shape[0])

    finally:
        if m_moved:
            m.to(mod_dev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_full", action="store_true", help="Run full training/eval (GPU recommended).")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels.")
    args = parser.parse_args()

    tok = CharTokenizer()

    if args.run_full:
        cfg = TrainConfig()
        passkey_lengths = [512]
        niah_lengths =    [512, 1024, 2048, 4096, 8192, 16384]
        niah_depths = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_eval_batches = 10
    else:
        cfg = TrainConfig(
            d_model=192, n_layers=2, n_heads=4, mlp_ratio=3.0,
            batch_size=8, lr=5e-4, warmup_steps=10, max_steps=50,
            train_context_len=256, device="cpu",
            use_amp=False,
            use_data_parallel=False,
        )
        passkey_lengths = [64, 128, 256, 512]
        niah_lengths = [64, 128, 256, 512]
        niah_depths = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_eval_batches = 2

    device = torch.device(cfg.device)
    set_seed(cfg.seed, deterministic=args.deterministic)

    variants = ["goat", "rope", "alibi", "sinus", "rope_pi"]

    models: Dict[str, nn.Module] = {}
    for v in variants:
        print(f"Training passkey variant: {v}")
        m = train_model(v, tok, cfg, deterministic=args.deterministic)
        # Move trained model to CPU to avoid VRAM accumulation across variants
        m = m.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        models[v] = m

    cfg_niah = replace(cfg, train_context_len=min(cfg.train_context_len, 256), span_loss_only=True)
    niah_models: Dict[str, nn.Module] = {}
    for v in variants:
        print(f"Training NIAH variant: {v}")
        m_niah = train_model_niah(v, tok, cfg_niah, deterministic=args.deterministic)
        m_niah = m_niah.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        niah_models[v] = m_niah

    df_pass = eval_passkey_across_lengths(models, tok, passkey_lengths, key_len=8, batch_size=4, n_batches=n_eval_batches, device=cfg.device)
    save_passkey_results(df_pass, cfg.train_context_len)
    print("Saved passkey_results.json and passkey_results.csv")

    niah_results = eval_niah_grid(niah_models, tok, niah_lengths, niah_depths, needle_len=12, batch_size=2, n_batches=n_eval_batches, device=cfg.device)
    try:
        save_niah_results(niah_results, niah_lengths, niah_depths)
        print("Saved niah_results.npz")
    except Exception as e:
        print("Could not save NIAH results:", e)

    if "goat" in models:
        try:
            save_goat_prior(models["goat"], L=min(cfg.train_context_len, 512), out_dir=".", device=cfg.device)
            print("Saved goat_prior.npz")
        except Exception as e:
            print("Could not save GOAT prior:", e)
    else:
        print("Skipping GOAT prior (no GOAT model).")


if __name__ == "__main__":
    main()
