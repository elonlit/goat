#!/usr/bin/env python3
"""Train and benchmark GOAT vs RoPE on InstaDeepAI/human_reference_genome."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def hf_load_dataset(*args, **kwargs):
    from datasets import load_dataset
    if len(args) >= 1 and isinstance(args[0], str) and args[0].lower() == "instadeepai/human_reference_genome":
        kwargs.setdefault("trust_remote_code", True)
    return load_dataset(*args, **kwargs)

from goat import GoatAttention

def import_symbol(path: str):
    mod_name, sym_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no CUDA


def human_bytes(n: float) -> str:
    if not np.isfinite(n):
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while n >= 1024 and u < len(units) - 1:
        n /= 1024
        u += 1
    return f"{n:.2f} {units[u]}"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_int(n: int) -> str:
    return f"{n:,}"


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class DNACharTokenizer:
    def __init__(self, alphabet: str = "ACGTN"):
        alphabet = "".join(sorted(set(alphabet)))
        if "N" not in alphabet:
            alphabet += "N"
        self.alphabet = alphabet

        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(alphabet)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.unk_id = self.stoi["N"]

        lut = np.full((256,), self.unk_id, dtype=np.uint8)
        for ch, idx in self.stoi.items():
            lut[ord(ch)] = idx
            lut[ord(ch.lower())] = idx
        self._lut = lut

    def encode(self, s: str) -> np.ndarray:
        b = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8)
        return self._lut[b].astype(np.int64, copy=False)

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "N") for i in ids)


class RandomCropGenomeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer: DNACharTokenizer,
        seq_len: int,
        crop_start_max: int = 200,
        deterministic: bool = False,
        stable_seed: Optional[int] = None,
    ):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.seq_len = seq_len
        self.crop_start_max = crop_start_max
        self.deterministic = deterministic
        self.stable_seed = stable_seed

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.ds[int(idx)]
        s: str = ex["sequence"]
        need = self.seq_len + 1
        if len(s) < need:
            s = (s + "N" * need)[:need]

        if self.deterministic:
            start = 0
        else:
            if self.crop_start_max <= 0:
                max_start = len(s) - need
            else:
                max_start = min(self.crop_start_max, len(s) - need)
            if max_start > 0:
                if self.stable_seed is not None:
                    start = stable_start_for_idx(int(idx), int(max_start), int(self.stable_seed))
                else:
                    start = random.randint(0, max_start)
            else:
                start = 0

        chunk = s[start : start + need]
        ids_np = self.tok.encode(chunk)  # (need,)
        ids = torch.from_numpy(ids_np).to(torch.long)

        x = ids[:-1].contiguous()
        y = ids[1:].contiguous()
        return x, y


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def stable_start_for_idx(idx: int, max_start: int, seed: int) -> int:
    if max_start <= 0:
        return 0
    x = (idx ^ seed) & 0xFFFFFFFF
    x ^= (x >> 16) & 0xFFFFFFFF
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15) & 0xFFFFFFFF
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16) & 0xFFFFFFFF
    return int(x % (max_start + 1))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached.index_select(0, positions)
        sin = self.sin_cached.index_select(0, positions)
        return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    cos = cos.view(1, 1, cos.size(0), cos.size(1)).to(q1.dtype)
    sin = sin.view(1, 1, sin.size(0), sin.size(1)).to(q1.dtype)

    q1 = (q1 * cos) + (_rotate_half(q1) * sin)
    k1 = (k1 * cos) + (_rotate_half(k1) * sin)

    q = torch.cat([q1, q2], dim=-1)
    k = torch.cat([k1, k2], dim=-1)
    return q, k


class RoPECausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.rotary_dim = self.head_dim
        self.rope = RotaryEmbedding(rotary_dim=self.rotary_dim, max_seq_len=max_seq_len, base=rope_base)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos = torch.arange(L, device=x.device, dtype=torch.long)
        cos, sin = self.rope(pos)
        q, k = apply_rope(q, k, cos, sin, rotary_dim=self.rotary_dim)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(y)


class GoatCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        training_seq_len: int,
        goat_pos_rank: int = 2,
        goat_abs_rank: int = 4,
        goat_prior_init: str = "seeded",
        goat_import: str = "goat_attention:GoatAttention",
    ):
        super().__init__()
        GoatAttentionCls = GoatAttention
        if GoatAttentionCls is None:
            GoatAttentionCls = import_symbol(goat_import)

        self.attn = GoatAttentionCls.for_gpt(
            embed_dim=d_model,
            num_heads=n_heads,
            kv_num_heads=n_heads,
            dropout=dropout,
            pos_rank=goat_pos_rank,
            abs_rank=goat_abs_rank,
            prior_init=goat_prior_init,
            training_seq_len=training_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False, is_causal=True)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, attn: nn.Module, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float,
        dropout: float,
        max_seq_len: int,
        attn_type: str,
        goat_kwargs: Dict[str, Any],
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        blocks: List[nn.Module] = []
        for _ in range(n_layers):
            if attn_type == "rope":
                attn = RoPECausalSelfAttention(
                    d_model, n_heads, dropout, max_seq_len=max_seq_len, rope_base=rope_base
                )
            elif attn_type == "goat":
                attn = GoatCausalSelfAttention(
                    d_model,
                    n_heads,
                    dropout,
                    training_seq_len=max_seq_len,
                    **goat_kwargs,
                )
            else:
                raise ValueError(f"Unknown attn_type={attn_type!r}")
            blocks.append(TransformerBlock(d_model, attn, mlp_ratio, dropout))

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
            )
        return logits, loss


def _has_new_sdpa_api() -> bool:
    return hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel")


@contextlib.contextmanager
def sdpa_force_backends(enable_flash: bool, enable_mem_efficient: bool, enable_math: bool):
    if not torch.cuda.is_available():
        yield
        return

    if _has_new_sdpa_api():
        backends = []
        if enable_flash:
            backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
        if enable_mem_efficient:
            backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
        if enable_math:
            backends.append(torch.nn.attention.SDPBackend.MATH)
        if not backends:
            raise ValueError("No SDPA backends enabled")
        with torch.nn.attention.sdpa_kernel(backends):
            yield
    elif hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        ):
            yield
    else:
        if not enable_math:
            raise RuntimeError("SDPA backend selection not available in this torch build")
        yield


@contextlib.contextmanager
def patch_torch_sdp_kernel_for_goat(enable_flash: bool, enable_mem_efficient: bool, enable_math: bool):
    if not torch.cuda.is_available():
        yield
        return

    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        raise RuntimeError("torch.backends.cuda.sdp_kernel not available")

    orig = torch.backends.cuda.sdp_kernel

    def _wrapped_sdp_kernel(*args, **kwargs):
        return orig(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        )

    torch.backends.cuda.sdp_kernel = _wrapped_sdp_kernel  # type: ignore
    try:
        yield
    finally:
        torch.backends.cuda.sdp_kernel = orig  # type: ignore


@torch.no_grad()
def eval_lm(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    autocast_fn,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_fn():
            logits, loss = model(x, y)
        assert loss is not None

        B, L = y.shape
        tokens = B * L
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())

    mean_nll = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    bpb = mean_nll / math.log(2.0)
    return {
        "nll": mean_nll,
        "ppl": ppl,
        "bpb": bpb,
        "acc": acc,
        "tokens": total_tokens,
    }


def benchmark_full_step(
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    autocast_fn,
    iters: int = 30,
    warmup: int = 10,
) -> Dict[str, float]:
    prev_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    with torch.enable_grad():
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            with autocast_fn():
                _, loss = model(x, y)
            assert loss is not None
            loss.backward()
        _cuda_sync()

        t0 = time.perf_counter()
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            with autocast_fn():
                _, loss = model(x, y)
            assert loss is not None
            loss.backward()
        _cuda_sync()
        t1 = time.perf_counter()

    model.zero_grad(set_to_none=True)
    model.train(prev_training)

    dt = (t1 - t0) / iters
    tokens = batch_size * seq_len
    return {
        "ms_per_iter": float(dt * 1000.0),
        "tokens_per_sec": float(tokens / max(dt, 1e-12)),
        "tokens_per_iter": float(tokens),
    }


@torch.no_grad()
def benchmark_attention_only(
    attn_module: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    d_model: int,
    autocast_fn,
    iters: int = 100,
    warmup: int = 20,
) -> Dict[str, float]:
    attn_module.eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    for _ in range(warmup):
        with autocast_fn():
            _ = attn_module(x)
    _cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast_fn():
            _ = attn_module(x)
    _cuda_sync()
    t1 = time.perf_counter()

    dt = (t1 - t0) / iters
    tokens = batch_size * seq_len
    return {
        "ms_per_iter": dt * 1000.0,
        "tokens_per_sec": tokens / max(dt, 1e-12),
        "tokens_per_iter": tokens,
    }


def try_flash_only_for_rope(
    attn: RoPECausalSelfAttention,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    d_model: int,
    autocast_fn,
) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    attn.eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    try:
        with sdpa_force_backends(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            with autocast_fn():
                _ = attn(x)
        _cuda_sync()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def try_flash_only_for_goat(
    attn: GoatCausalSelfAttention,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    d_model: int,
    autocast_fn,
) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    attn.eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    try:
        with patch_torch_sdp_kernel_for_goat(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            with autocast_fn():
                _ = attn(x)
        _cuda_sync()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def benchmark_attention_backends(
    attn_type: str,
    attn_module: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    d_model: int,
    autocast_fn,
    iters: int = 80,
    warmup: int = 20,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    def _ctx(backend: str):
        if backend == "all":
            return contextlib.nullcontext()
        if backend == "flash":
            return sdpa_force_backends(True, False, False)
        if backend == "mem_efficient":
            return sdpa_force_backends(False, True, False)
        if backend == "math":
            return sdpa_force_backends(False, False, True)
        raise ValueError(backend)

    def _ctx_goat(backend: str):
        if backend == "all":
            return contextlib.nullcontext()
        if backend == "flash":
            return patch_torch_sdp_kernel_for_goat(True, False, False)
        if backend == "mem_efficient":
            return patch_torch_sdp_kernel_for_goat(False, True, False)
        if backend == "math":
            return patch_torch_sdp_kernel_for_goat(False, False, True)
        raise ValueError(backend)

    for b in ["all", "flash", "mem_efficient", "math"]:
        try:
            ctx = _ctx_goat(b) if attn_type == "goat" else _ctx(b)
            with ctx:
                results[b] = benchmark_attention_only(
                    attn_module,
                    device=device,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    d_model=d_model,
                    autocast_fn=autocast_fn,
                    iters=iters,
                    warmup=warmup,
                )
        except Exception as e:
            results[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return results


@dataclass
class TrainConfig:
    attn_type: str
    seed: int
    out_dir: str
    dataset_config: str
    max_train_examples: int
    max_val_examples: int
    max_test_examples: int
    seq_len: int
    batch_size: int
    num_workers: int
    crop_start_max: int

    max_steps: int
    eval_interval: int
    patience_evals: int
    eval_batches: int
    test_batches: int
    eval_prefix_only: bool
    test_prefix_only: bool
    eval_shuffle: bool
    test_shuffle: bool
    eval_stable_crops: bool
    print_eval_token_hist: bool
    eval_token_hist_batches: int
    check_split_overlap: bool
    overlap_max_examples: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: float
    dropout: float
    rope_base: float
    goat_import: str
    goat_pos_rank: int
    goat_abs_rank: int
    goat_prior_init: str
    lr: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int
    precision: str
    compile: bool


def make_autocast_fn(precision: str, device: torch.device):
    precision = precision.lower().strip()

    @contextlib.contextmanager
    def _ctx():
        if device.type != "cuda":
            yield
            return

        if precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
            return
        if precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
            return
        if precision == "fp32":
            yield
            return
        if precision == "auto":
            if torch.cuda.is_bf16_supported():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yield
            return
        raise ValueError(f"Unknown precision={precision!r}")

    return _ctx


def maybe_make_grad_scaler(precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    precision = precision.lower().strip()
    if precision == "fp16" or (precision == "auto" and not torch.cuda.is_bf16_supported()):
        return torch.cuda.amp.GradScaler()
    return None


def lr_schedule(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def train_and_eval(cfg: TrainConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    autocast_fn = make_autocast_fn(cfg.precision, device)
    scaler = maybe_make_grad_scaler(cfg.precision, device)

    eval_batches = None if cfg.eval_batches <= 0 else cfg.eval_batches
    test_batches = None if cfg.test_batches <= 0 else cfg.test_batches

    ds = hf_load_dataset("InstaDeepAI/human_reference_genome", cfg.dataset_config)
    train_split = ds["train"]
    val_split = ds["validation"]
    test_split = ds["test"]

    if cfg.max_train_examples > 0:
        train_split = train_split.shuffle(seed=cfg.seed).select(
            range(min(cfg.max_train_examples, len(train_split)))
        )
    if cfg.max_val_examples > 0:
        val_split = val_split.select(range(min(cfg.max_val_examples, len(val_split))))
    if cfg.max_test_examples > 0:
        test_split = test_split.select(range(min(cfg.max_test_examples, len(test_split))))

    tokenizer = DNACharTokenizer("ACGTN")
    vocab_size = tokenizer.vocab_size

    @torch.no_grad()
    def quick_token_hist(loader, vocab_size: int, n_batches: int = 50):
        counts = torch.zeros(vocab_size, dtype=torch.long)
        total = 0
        for i, (_x, y) in enumerate(loader):
            if i >= n_batches:
                break
            yy = y.reshape(-1)
            counts += torch.bincount(yy, minlength=vocab_size).cpu()
            total += int(yy.numel())
        return counts, total

    def _chunk_for_split(
        split, idx: int, need: int, crop_start_max: int, prefix_only: bool, stable_seed: Optional[int]
    ) -> str:
        ex = split[int(idx)]
        s: str = ex["sequence"]
        if len(s) < need:
            s = (s + "N" * need)[:need]
        if prefix_only:
            start = 0
        else:
            if crop_start_max <= 0:
                max_start = len(s) - need
            else:
                max_start = min(crop_start_max, len(s) - need)
            if max_start > 0 and stable_seed is not None:
                start = stable_start_for_idx(int(idx), int(max_start), int(stable_seed))
            elif max_start > 0:
                # Fall back to prefix if we can't reproduce the crop.
                start = 0
            else:
                start = 0
        return s[start : start + need]

    train_ds = RandomCropGenomeDataset(
        train_split,
        tokenizer,
        seq_len=cfg.seq_len,
        crop_start_max=cfg.crop_start_max,
        deterministic=False,
    )

    eval_stable_seed = int(cfg.seed) + 1337 if cfg.eval_stable_crops else None
    val_ds = RandomCropGenomeDataset(
        val_split,
        tokenizer,
        seq_len=cfg.seq_len,
        crop_start_max=cfg.crop_start_max,
        deterministic=bool(cfg.eval_prefix_only),
        stable_seed=None if cfg.eval_prefix_only else eval_stable_seed,
    )
    test_ds = RandomCropGenomeDataset(
        test_split,
        tokenizer,
        seq_len=cfg.seq_len,
        crop_start_max=cfg.crop_start_max,
        deterministic=bool(cfg.test_prefix_only),
        stable_seed=None if cfg.test_prefix_only else eval_stable_seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )

    val_ds_for_loader = val_ds
    if cfg.eval_shuffle:
        seed = int(cfg.seed) + 202
        rs = np.random.RandomState(seed)
        if eval_batches is not None:
            k = min(len(val_ds), int(eval_batches) * int(cfg.batch_size))
            idxs = rs.choice(len(val_ds), size=k, replace=False).tolist() if k > 0 else []
        else:
            idxs = rs.permutation(len(val_ds)).tolist()
        val_ds_for_loader = torch.utils.data.Subset(val_ds, idxs)
    val_loader = torch.utils.data.DataLoader(
        val_ds_for_loader,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, min(cfg.num_workers, 2)),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )

    test_ds_for_loader = test_ds
    if cfg.test_shuffle:
        seed = int(cfg.seed) + 303
        rs = np.random.RandomState(seed)
        if test_batches is not None:
            k = min(len(test_ds), int(test_batches) * int(cfg.batch_size))
            idxs = rs.choice(len(test_ds), size=k, replace=False).tolist() if k > 0 else []
        else:
            idxs = rs.permutation(len(test_ds)).tolist()
        test_ds_for_loader = torch.utils.data.Subset(test_ds, idxs)
    test_loader = torch.utils.data.DataLoader(
        test_ds_for_loader,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, min(cfg.num_workers, 2)),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
    )

    # Quick diagnostics (optional): helps catch "all-Ns / trivial prefixes" and leakage/duplication.
    if cfg.print_eval_token_hist:
        hb = int(cfg.eval_token_hist_batches)
        counts, total = quick_token_hist(val_loader, vocab_size=vocab_size, n_batches=hb)
        fracs = (counts.float() / max(total, 1)).tolist()
        labels = [tokenizer.itos[i] for i in range(vocab_size)]
        pairs = list(zip(labels, counts.tolist(), fracs))
        pairs.sort(key=lambda t: t[1], reverse=True)
        print("\n=== VAL label histogram (top tokens) ===")
        for ch, c, f in pairs:
            if c == 0:
                continue
            print(f"  {ch}: {c} ({100.0*f:.2f}%)")
        print(f"  total_labels: {total}")
    if cfg.check_split_overlap:
        import hashlib

        need = int(cfg.seq_len) + 1
        max_n = int(cfg.overlap_max_examples)
        train_n = min(max_n, len(train_split))
        val_n = min(max_n, len(val_split))
        ss = eval_stable_seed

        def _h(s: str) -> str:
            return hashlib.md5(s.encode("ascii", "ignore")).hexdigest()

        train_hashes = set()
        for i in range(train_n):
            chunk = _chunk_for_split(
                train_split,
                i,
                need=need,
                crop_start_max=cfg.crop_start_max,
                prefix_only=False,
                stable_seed=ss,
            )
            train_hashes.add(_h(chunk))

        overlap = 0
        for i in range(val_n):
            chunk = _chunk_for_split(
                val_split,
                i,
                need=need,
                crop_start_max=cfg.crop_start_max,
                prefix_only=bool(cfg.eval_prefix_only),
                stable_seed=ss,
            )
            if _h(chunk) in train_hashes:
                overlap += 1

        print(f"[overlap] train={train_n} val={val_n} overlap={overlap}")

    goat_kwargs = {
        "goat_import": cfg.goat_import,
        "goat_pos_rank": cfg.goat_pos_rank,
        "goat_abs_rank": cfg.goat_abs_rank,
        "goat_prior_init": cfg.goat_prior_init,
    }

    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        max_seq_len=cfg.seq_len,
        attn_type=cfg.attn_type,
        goat_kwargs=goat_kwargs,
        rope_base=cfg.rope_base,
    ).to(device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_step = -1
    best_path = os.path.join(cfg.out_dir, f"best_{cfg.attn_type}.pt")
    no_improve = 0

    step_times: List[float] = []
    tokens_per_step = cfg.batch_size * cfg.seq_len

    train_iter = iter(train_loader)
    pbar = tqdm(range(cfg.max_steps), desc=f"train[{cfg.attn_type}]", dynamic_ncols=True)
    for step in pbar:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        lr = lr_schedule(step, cfg.max_steps, cfg.warmup_steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        t0 = time.perf_counter()

        model.train()
        opt.zero_grad(set_to_none=True)

        with autocast_fn():
            _, loss = model(x, y)
        assert loss is not None

        if scaler is None:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        else:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

        _cuda_sync()
        dt = time.perf_counter() - t0
        step_times.append(dt)

        window = step_times[-50:]
        tps = (tokens_per_step * len(window)) / max(sum(window), 1e-12)

        pbar.set_postfix(loss=float(loss.item()), lr=lr, tok_s=tps)

        history.append(
            {
                "step": int(step),
                "train_loss": float(loss.item()),
                "lr": float(lr),
                "step_time_s": float(dt),
                "tokens_per_sec": float(tps),
            }
        )

        # Eval
        if (step + 1) % cfg.eval_interval == 0 or (step + 1) == cfg.max_steps:
            val_metrics = eval_lm(
                model, val_loader, device, autocast_fn, max_batches=eval_batches
            )
            history[-1].update(
                {
                    "val_nll": float(val_metrics["nll"]),
                    "val_ppl": float(val_metrics["ppl"]),
                    "val_bpb": float(val_metrics["bpb"]),
                    "val_acc": float(val_metrics["acc"]),
                }
            )

            if val_metrics["nll"] < best_val - 1e-6:
                best_val = float(val_metrics["nll"])
                best_step = int(step)
                no_improve = 0
                torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, best_path)
            else:
                no_improve += 1

            if no_improve >= cfg.patience_evals:
                pbar.write(
                    f"[{cfg.attn_type}] Early stop: no val improvement for {cfg.patience_evals} evals."
                )
                break

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    val_metrics = eval_lm(model, val_loader, device, autocast_fn, max_batches=eval_batches)
    test_metrics = eval_lm(model, test_loader, device, autocast_fn, max_batches=test_batches)

    attn_only_module = model.blocks[0].attn

    full_step_bench = benchmark_full_step(
        model,
        device=device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        vocab_size=vocab_size,
        autocast_fn=autocast_fn,
        iters=30,
        warmup=10,
    )
    attn_only_bench = benchmark_attention_only(
        attn_only_module,
        device=device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        d_model=cfg.d_model,
        autocast_fn=autocast_fn,
        iters=100,
        warmup=20,
    )

    if cfg.attn_type == "rope":
        flash_ok, flash_msg = try_flash_only_for_rope(
            attn_only_module, device, cfg.seq_len, cfg.batch_size, cfg.d_model, autocast_fn  # type: ignore
        )
    else:
        flash_ok, flash_msg = try_flash_only_for_goat(
            attn_only_module, device, cfg.seq_len, cfg.batch_size, cfg.d_model, autocast_fn  # type: ignore
        )

    backend_benches = benchmark_attention_backends(
        cfg.attn_type,
        attn_only_module,
        device=device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        d_model=cfg.d_model,
        autocast_fn=autocast_fn,
        iters=80,
        warmup=20,
    )
    train_tps_vals = [r.get("tokens_per_sec") for r in history if "tokens_per_sec" in r]
    train_tps_vals = [float(v) for v in train_tps_vals if v is not None and np.isfinite(float(v))]
    train_tps_median = float(np.median(train_tps_vals[-200:])) if train_tps_vals else float("nan")

    peak_mem_alloc = float("nan")
    peak_mem_reserved = float("nan")
    if device.type == "cuda":
        peak_mem_alloc = float(torch.cuda.max_memory_allocated(device=device))
        peak_mem_reserved = float(torch.cuda.max_memory_reserved(device=device))

    summary: Dict[str, Any] = {
        "attn_type": cfg.attn_type,
        "param_count": int(count_parameters(model)),
        "converged_val_nll": float(best_val),
        "converged_step": int(best_step),
        "final_val_nll": float(val_metrics["nll"]),
        "final_val_ppl": float(val_metrics["ppl"]),
        "final_val_bpb": float(val_metrics["bpb"]),
        "final_val_acc": float(val_metrics["acc"]),
        "final_test_nll": float(test_metrics["nll"]),
        "final_test_ppl": float(test_metrics["ppl"]),
        "final_test_bpb": float(test_metrics["bpb"]),
        "final_test_acc": float(test_metrics["acc"]),
        "train_tokens_per_step": int(tokens_per_step),
        "train_tokens_per_sec_median": float(train_tps_median),
        "bench_full_ms_per_iter": float(full_step_bench["ms_per_iter"]),
        "bench_full_tokens_per_sec": float(full_step_bench["tokens_per_sec"]),
        "bench_attn_ms_per_iter": float(attn_only_bench["ms_per_iter"]),
        "bench_attn_tokens_per_sec": float(attn_only_bench["tokens_per_sec"]),
        "flash_only_ok": bool(flash_ok),
        "flash_only_msg": str(flash_msg),
        "peak_cuda_mem_alloc_bytes": float(peak_mem_alloc),
        "peak_cuda_mem_reserved_bytes": float(peak_mem_reserved),
    }

    json_path = os.path.join(cfg.out_dir, f"summary_{cfg.attn_type}.json")
    with open(json_path, "w") as f:
        payload = {
            "cfg": dataclasses.asdict(cfg),
            "summary": summary,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "bench_full_step": full_step_bench,
            "bench_attn_only": attn_only_bench,
            "bench_attn_backends": backend_benches,
        }
        json.dump(payload, f, indent=2)

    hist_path = os.path.join(cfg.out_dir, f"history_{cfg.attn_type}.csv")
    with open(hist_path, "w", newline="") as f:
        keys: set[str] = set()
        for r in history:
            keys.update(r.keys())
        fieldnames = sorted(keys)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in history:
            w.writerow(r)

    make_reviewer_artifacts(
        model=model,
        tokenizer=tokenizer,
        val_split=val_split,
        device=device,
        autocast_fn=autocast_fn,
        out_dir=cfg.out_dir,
        tag=cfg.attn_type,
    )

    if cfg.attn_type == "goat":
        try:
            save_goat_prior(
                model,
                device=device,
                out_path=os.path.join(cfg.out_dir, "goat_log_prior.npy"),
                L=min(256, cfg.seq_len),
            )
        except Exception as e:
            with open(os.path.join(cfg.out_dir, "goat_prior_error.txt"), "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")

    with open(os.path.join(cfg.out_dir, f"attn_backend_bench_{cfg.attn_type}.json"), "w") as f:
        json.dump(backend_benches, f, indent=2)

    return summary, history


@torch.no_grad()
def generate_dna(
    model: GPT,
    tokenizer: DNACharTokenizer,
    prompt: str,
    device: torch.device,
    autocast_fn,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.from_numpy(ids).to(device=device, dtype=torch.long).unsqueeze(0)  # (1,L)

    for _ in range(max_new_tokens):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len :]
        with autocast_fn():
            logits, _ = model(x, labels=None)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            v, idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(1, idx, v)
            next_logits = mask
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x.squeeze(0).tolist())


@torch.no_grad()
def make_reviewer_artifacts(
    model: GPT,
    tokenizer: DNACharTokenizer,
    val_split,
    device: torch.device,
    autocast_fn,
    out_dir: str,
    tag: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(123)
    indices = [rng.randrange(0, len(val_split)) for _ in range(3)]

    samples: List[Tuple[str, str]] = []
    for idx in indices:
        s = val_split[int(idx)]["sequence"]
        prompt = s[:200]
        gen = generate_dna(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            autocast_fn=autocast_fn,
            max_new_tokens=200,
            temperature=1.0,
            top_k=0,
        )
        samples.append((prompt, gen))

    txt_path = os.path.join(out_dir, f"generated_samples_{tag}.txt")
    with open(txt_path, "w") as f:
        for i, (prompt, seq) in enumerate(samples):
            f.write(f"=== SAMPLE {i} ({tag}) ===\n")
            f.write(f"PROMPT (first 200 bp):\n{prompt}\n\n")
            f.write(f"PROMPT+GEN (400 bp total):\n{seq}\n\n")

    hist = {ch: samples[0][1].count(ch) for ch in tokenizer.alphabet}
    hist_json_path = os.path.join(out_dir, f"generated_base_hist_{tag}.json")
    with open(hist_json_path, "w") as f:
        json.dump(
            {
                "tag": tag,
                "alphabet": tokenizer.alphabet,
                "counts": hist,
                "sequence_length": int(len(samples[0][1])),
                "note": "Base-count histogram data.",
            },
            f,
            indent=2,
        )


@torch.no_grad()
def save_goat_prior(model: GPT, device: torch.device, out_path: str, L: int = 256) -> None:
    """Save GOAT prior data to NPY/JSON."""
    goat_wrapper = model.blocks[0].attn
    if not hasattr(goat_wrapper, "attn"):
        raise RuntimeError("First block does not appear to be GOAT.")
    goat_attn = goat_wrapper.attn
    if not hasattr(goat_attn, "compute_log_prior"):
        raise RuntimeError("GoatAttention does not expose compute_log_prior.")

    log_prior = goat_attn.compute_log_prior(L=L, device=device, dtype=torch.float32, is_causal=True)
    log_prior = log_prior.detach().cpu().numpy()

    npy_path = os.path.splitext(out_path)[0] + ".npy"
    np.save(npy_path, log_prior)
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "L": int(L),
                "shape": list(log_prior.shape),
                "dtype": str(log_prior.dtype),
                "npy_path": os.path.basename(npy_path),
                "is_causal": True,
                "note": "Raw GOAT positional log-prior data.",
            },
            f,
            indent=2,
        )


def save_curves_data(histories: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    """Save validation curves data to JSON."""
    curves_data: Dict[str, Dict[str, List[float]]] = {}
    for name, hist in histories.items():
        steps = [float(r["step"]) for r in hist if "val_nll" in r]
        vals = [float(r["val_nll"]) for r in hist if "val_nll" in r]
        if steps:
            curves_data[name] = {"steps": steps, "val_nll": vals}

    with open(out_path, "w") as f:
        json.dump({"curves": curves_data}, f, indent=2)


def save_throughput_data(summaries: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """Save throughput data to JSON."""
    names = list(summaries.keys())
    vals = []
    for n in names:
        v = float(summaries[n].get("train_tokens_per_sec_median", float("nan")))
        if not np.isfinite(v):
            v = float(summaries[n].get("bench_full_tokens_per_sec", float("nan")))
        vals.append(v)

    with open(out_path, "w") as f:
        json.dump({"models": names, "tokens_per_sec": vals}, f, indent=2)


def save_combined_data(out_dir: str, histories: Dict[str, List[Dict[str, Any]]], summaries: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    val_curves: Dict[str, Dict[str, List[float]]] = {}
    for name, hist in histories.items():
        steps = [float(r["step"]) for r in hist if "val_nll" in r]
        vals = [float(r["val_nll"]) for r in hist if "val_nll" in r]
        val_curves[name] = {"steps": steps, "val_nll": vals}

    names = list(summaries.keys())
    vals = []
    for n in names:
        v = float(summaries[n].get("train_tokens_per_sec_median", float("nan")))
        if not np.isfinite(v):
            v = float(summaries[n].get("bench_full_tokens_per_sec", float("nan")))
        vals.append(v)

    payload = {
        "val_nll_curves": val_curves,
        "throughput": {"models": names, "tokens_per_sec": vals},
    }
    with open(os.path.join(out_dir, "combined_data.json"), "w") as f:
        json.dump(payload, f, indent=2)


def write_report_md(out_dir: str, summaries: Dict[str, Dict[str, Any]]) -> None:
    def fmt(x: Any, digits: int = 4) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, bool):
            return "✅" if x else "❌"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(float(x)):
                return "n/a"
            return f"{float(x):.{digits}f}"
        return str(x)

    rows = []
    for name, s in summaries.items():
        rows.append(
            {
                "model": name,
                "params": format_int(int(s.get("param_count", 0))),
                "converged_val_nll": fmt(s.get("converged_val_nll")),
                "final_test_nll": fmt(s.get("final_test_nll")),
                "final_test_bpb": fmt(s.get("final_test_bpb")),
                "flash_only": fmt(s.get("flash_only_ok")),
                "train_tok_s": fmt(s.get("train_tokens_per_sec_median"), digits=1),
                "attn_tok_s": fmt(s.get("bench_attn_tokens_per_sec"), digits=1),
                "peak_alloc": human_bytes(float(s.get("peak_cuda_mem_alloc_bytes", float("nan")))),
            }
        )

    md = []
    md.append("# GOAT vs RoPE – human_reference_genome\n")
    md.append("## Summary\n")
    md.append(
        "| Model | Params | Converged val NLL | Final test NLL | Final test bits/base | Flash-only SDPA | Train tok/s | Attn tok/s | Peak CUDA alloc |\n"
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|"
    )
    for r in rows:
        md.append(
            f"| {r['model']} | {r['params']} | {r['converged_val_nll']} | {r['final_test_nll']} | {r['final_test_bpb']} | {r['flash_only']} | {r['train_tok_s']} | {r['attn_tok_s']} | {r['peak_alloc']} |"
        )

    md.append("\n## Artifacts\n")
    md.append("- Validation curves data: `val_nll_curves.json`")
    md.append("- Throughput data: `throughput_tokens_per_sec.json`")
    md.append("- Combined data: `combined_data.json`")
    md.append("- Samples: `rope/generated_samples_rope.txt`, `goat/generated_samples_goat.txt`")
    md.append("- Base histograms data: `rope/generated_base_hist_rope.json`, `goat/generated_base_hist_goat.json`")
    md.append("- GOAT prior data (if available): `goat/goat_log_prior.npy`, `goat/goat_log_prior.json`\n")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", type=str, default=f"runs/goat_vs_rope_{now_ts()}")
    p.add_argument("--dataset_config", type=str, default="6kbp", choices=["6kbp", "12kbp"])
    p.add_argument(
        "--max_train_examples",
        type=int,
        default=0,
        help="If >0, trains on a shuffled subset of this many examples.",
    )
    p.add_argument(
        "--max_val_examples",
        type=int,
        default=0,
        help="If >0, evaluates on at most this many validation examples.",
    )
    p.add_argument(
        "--max_test_examples",
        type=int,
        default=0,
        help="If >0, evaluates on at most this many test examples.",
    )

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--crop_start_max",
        type=int,
        default=200,
        help="If <=0 random-crop anywhere; if >0, crop start is limited to [0,crop_start_max].",
    )

    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--patience_evals", type=int, default=5)
    p.add_argument(
        "--eval_batches", type=int, default=50, help="Limit validation batches per eval (speed)."
    )
    p.add_argument(
        "--test_batches", type=int, default=50, help="Limit test batches for final report (speed)."
    )
    p.add_argument(
        "--eval_prefix_only",
        action="store_true",
        help="If set, validation crops always start at position 0 (prefix-only; can be biased).",
    )
    p.add_argument(
        "--test_prefix_only",
        action="store_true",
        help="If set, test crops always start at position 0 (prefix-only; can be biased).",
    )
    p.add_argument(
        "--eval_shuffle",
        type=int,
        default=1,
        choices=[0, 1],
        help="Shuffle validation loader (recommended if eval_batches>0).",
    )
    p.add_argument(
        "--test_shuffle",
        type=int,
        default=1,
        choices=[0, 1],
        help="Shuffle test loader (recommended if test_batches>0).",
    )
    p.add_argument(
        "--eval_stable_crops",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use deterministic per-example random crop starts for val/test (repeatable).",
    )
    p.add_argument(
        "--print_eval_token_hist",
        action="store_true",
        help="Print label-token histogram for evaluated validation batches (debug trivial eval).",
    )
    p.add_argument(
        "--eval_token_hist_batches",
        type=int,
        default=50,
        help="How many validation batches to use for --print_eval_token_hist.",
    )
    p.add_argument(
        "--check_split_overlap",
        action="store_true",
        help="Check train/val overlap of evaluated windows (debug leakage/duplicates).",
    )
    p.add_argument(
        "--overlap_max_examples",
        type=int,
        default=20000,
        help="Max examples per split to use for --check_split_overlap.",
    )

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rope_base", type=float, default=10000.0)

    # GOAT knobs
    p.add_argument("--goat_import", type=str, default="goat:GoatAttention")
    p.add_argument("--goat_pos_rank", type=int, default=2)
    p.add_argument("--goat_abs_rank", type=int, default=8)
    p.add_argument("--goat_prior_init", type=str, default="seeded", choices=["seeded", "uniform"])

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)

    # Precision / perf
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=123)

    # Which to run
    p.add_argument("--run_rope", action="store_true")
    p.add_argument("--run_goat", action="store_true")

    return p


def print_env_info() -> None:
    print("=== Environment ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass
        # These accessors vary by torch version.
        for name in ["flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"]:
            if hasattr(torch.backends.cuda, name):
                try:
                    print(f"{name}:", getattr(torch.backends.cuda, name)())
                except Exception:
                    pass
        print("new sdpa api:", _has_new_sdpa_api())
    print("===================")


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print_env_info()

    if not args.run_rope and not args.run_goat:
        args.run_rope = True
        args.run_goat = True

    summaries: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, List[Dict[str, Any]]] = {}

    common = dict(
        seed=args.seed,
        out_dir=args.out_dir,
        dataset_config=args.dataset_config,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        max_test_examples=args.max_test_examples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_start_max=args.crop_start_max,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        patience_evals=args.patience_evals,
        eval_batches=args.eval_batches,
        test_batches=args.test_batches,
        eval_prefix_only=args.eval_prefix_only,
        test_prefix_only=args.test_prefix_only,
        eval_shuffle=bool(args.eval_shuffle),
        test_shuffle=bool(args.test_shuffle),
        eval_stable_crops=bool(args.eval_stable_crops),
        print_eval_token_hist=args.print_eval_token_hist,
        eval_token_hist_batches=args.eval_token_hist_batches,
        check_split_overlap=args.check_split_overlap,
        overlap_max_examples=args.overlap_max_examples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        rope_base=args.rope_base,
        goat_import=args.goat_import,
        goat_pos_rank=args.goat_pos_rank,
        goat_abs_rank=args.goat_abs_rank,
        goat_prior_init=args.goat_prior_init,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        precision=args.precision,
        compile=args.compile,
    )

    if args.run_rope:
        cfg = TrainConfig(attn_type="rope", **common)
        cfg.out_dir = os.path.join(args.out_dir, "rope")
        summ, hist = train_and_eval(cfg)
        summaries["RoPE"] = summ
        histories["RoPE"] = hist

    if args.run_goat:
        cfg = TrainConfig(attn_type="goat", **common)
        cfg.out_dir = os.path.join(args.out_dir, "goat")
        summ, hist = train_and_eval(cfg)
        summaries["GOAT"] = summ
        histories["GOAT"] = hist

    # Save data
    if histories:
        save_curves_data(histories, out_path=os.path.join(args.out_dir, "val_nll_curves.json"))
    if summaries:
        save_throughput_data(summaries, out_path=os.path.join(args.out_dir, "throughput.json"))
        write_report_md(args.out_dir, summaries)
    if histories or summaries:
        save_combined_data(args.out_dir, histories=histories, summaries=summaries)

    print("\n=== Summary ===")
    for name, s in summaries.items():
        print(
            f"[{name}] params={format_int(int(s['param_count']))} "
            f"converged_val_nll={s['converged_val_nll']:.4f} "
            f"test_nll={s['final_test_nll']:.4f} "
            f"tok/s(train)={s['train_tokens_per_sec_median'] if np.isfinite(s['train_tokens_per_sec_median']) else float('nan'):.1f} "
            f"tok/s(attn)={s['bench_attn_tokens_per_sec']:.1f} "
            f"flash_ok={bool(s['flash_only_ok'])} "
            f"peak_alloc={human_bytes(s['peak_cuda_mem_alloc_bytes'])}"
        )

    with open(os.path.join(args.out_dir, "summary_all.json"), "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)

    print(f"\nArtifacts written to: {args.out_dir}")
    print("Key files:")
    print("  - summary_all.json")
    print("  - report.md")
    print("  - combined_data.json")
    print("  - val_nll_curves.json")
    print("  - throughput.json")
    print("  - rope/summary_rope.json, goat/summary_goat.json")
    print("  - rope/generated_samples_rope.txt, goat/generated_samples_goat.txt")
    print("  - rope/generated_base_hist_rope.json, goat/generated_base_hist_goat.json")
    print("  - goat/goat_log_prior.npy, goat/goat_log_prior.json")


if __name__ == "__main__":
    main()
