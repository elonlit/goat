"""
GOAT attention module.

This file contains the full PyTorch implementation for GOAT.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional, Tuple, Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._alibi import _alibi_slopes
from ._fourier import (
    _fourier_features,
    _fourier_features_axis,
    _infer_hw_from_length,
    _validate_spatial_shape,
)
from ._math import _inv_softplus
from ._mlp import _SmallMLP

__all__ = ["GoatAttention"]


class _GoatAttentionBase(nn.Module):
    """
    Multi-head attention with a learned position-dependent bias and an optional
    key-only “sink” bias.

    Practical notes:
    - Supports GQA/MQA via `kv_num_heads`.
    - Supports KV caching for incremental decoding.
    - Supports 1D or 2D positional structure (e.g., language vs ViT-style grids).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        # Spectral prior hyperparams
        pos_rank: int = 8,               # R (number of Fourier frequencies)
        abs_rank: int = 8,               # M (absolute Fourier frequencies for sink)
        pos_base: float = 10_000.0,
        abs_base: float = 10_000.0,
        enable_key_bias: bool = True,    # enable u(j) sink term
        init_scale_prior: float = 1e-3,
        training_seq_len: Optional[int] = None,  # for stable normalization in u(j)
        # Initialization mode for the prior parameters.
        # - "seeded": sensible non-zero defaults
        # - "uniform": start close to neutral, still learnable
        prior_init: str = "seeded",
        # Prior gate (optional): a learnable mechanism to dial the prior up/down.
        prior_gate: str = "none",        # "none" | "query_norm"
        # GQA / MQA
        kv_num_heads: Optional[int] = None,
        fuse_qkv: bool = False,
        # Positional structure
        pos_encoding: str = "1d",        # "auto" | "1d" | "2d"
        has_cls_token_default: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Basic dims
        self.embed_dim = embed_dim
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.head_dim = embed_dim // num_heads

        self.kv_num_heads = int(num_heads if kv_num_heads is None else kv_num_heads)
        if self.kv_num_heads <= 0:
            raise ValueError("kv_num_heads must be >= 1.")
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError("num_heads must be a multiple of kv_num_heads for GQA/MQA.")
        self._qkv_group_size = self.num_heads // self.kv_num_heads
        self.kv_embed_dim = self.kv_num_heads * self.head_dim

        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.R = int(pos_rank)
        self.M = int(abs_rank)
        if self.R < 0:
            raise ValueError("pos_rank must be >= 0.")
        self.enable_key_bias = bool(enable_key_bias)
        self.init_scale_prior = float(init_scale_prior)
        self.training_seq_len = int(training_seq_len) if training_seq_len is not None else None

        self.prior_init = str(prior_init).lower().strip()
        if self.prior_init not in ("seeded", "uniform"):
            raise ValueError(f"prior_init must be 'seeded' or 'uniform', got {prior_init!r}.")

        self.pos_dim = (2 * self.R) + (1 if self.enable_key_bias else 0)
        if self.head_dim <= self.pos_dim:
            raise ValueError(
                f"head_dim={self.head_dim} too small for pos_dim={self.pos_dim}. "
                f"Increase embed_dim/num_heads or reduce pos_rank."
            )
        self.content_dim = self.head_dim - self.pos_dim

        self._can_fuse_qkv = (
            fuse_qkv and kdim is None and vdim is None and
            self.kdim == self.embed_dim and self.vdim == self.embed_dim
        )
        if self._can_fuse_qkv:
            self.in_proj = nn.Linear(
                self.embed_dim,
                self.embed_dim + 2 * self.kv_embed_dim,
                bias=bias,
                **factory_kwargs,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.in_proj = None
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(self.kdim, self.kv_embed_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(self.vdim, self.kv_embed_dim, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)

        if self.add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, self.kv_embed_dim, **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty(1, 1, self.kv_embed_dim, **factory_kwargs))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.lambda_sym = nn.Parameter(
            torch.empty(self.kv_num_heads, self.R, device=device, dtype=torch.float32)
        )
        self.lambda_asym = nn.Parameter(
            torch.empty(self.kv_num_heads, self.R, device=device, dtype=torch.float32)
        )

        k_idx = torch.arange(self.R, device=device, dtype=torch.float32)
        m_idx = torch.arange(self.M, device=device, dtype=torch.float32)
        rel_freqs = pos_base ** (-k_idx / max(self.R - 1, 1))
        abs_freqs = abs_base ** (-m_idx / max(self.M - 1, 1))
        self.register_buffer("rel_freqs_base", rel_freqs)
        self.register_buffer("abs_freqs_base", abs_freqs)

        mlp_kwargs = {"device": device, "dtype": torch.float32}
        if self.enable_key_bias:
            self._abs_extra_dim = 2  # [t_norm, log_t_norm]
            self.key_bias = _SmallMLP(
                in_dim=(2 * self.M) + self._abs_extra_dim,
                out_dim=1,
                init_scale=init_scale_prior,
                **mlp_kwargs,
            )
            self.register_buffer("goat_prior_scale", torch.tensor(1.0), persistent=False)

            self.sink_gain = nn.Parameter(torch.ones((self.kv_num_heads,), device=device, dtype=torch.float32))
            self.sink_bump = nn.Parameter(torch.zeros((self.kv_num_heads,), device=device, dtype=torch.float32))

            self.recency_slope_raw = nn.Parameter(
                torch.empty((self.kv_num_heads,), device=device, dtype=torch.float32)
            )
        else:
            self.register_module("key_bias", None)
            self.register_parameter("recency_slope_raw", None)
            self._abs_extra_dim = 0
            self.register_buffer("goat_prior_scale", torch.tensor(1.0), persistent=False)
            self.register_parameter("sink_gain", None)
            self.register_parameter("sink_bump", None)

        self.prior_gate_scale_raw = nn.Parameter(
            torch.full((self.num_heads,), -10.0, device=device, dtype=torch.float32)
        )
        self.prior_gate_bias = nn.Parameter(
            torch.full((self.num_heads,), 10.0, device=device, dtype=torch.float32)
        )

        self.max_cache_entries = 8
        self._rel_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()
        self._abs_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()
        self._rel2d_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()
        self._abs2d_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()

        self.pos_encoding: str = pos_encoding
        self.has_cls_token_default: bool = has_cls_token_default

        self.prior_gate = str(prior_gate).lower().strip()
        if self.prior_gate not in ("none", "query_norm"):
            raise ValueError(f"prior_gate must be 'none' or 'query_norm', got {prior_gate!r}.")

        self._reset_parameters()

    def set_goat_prior_scale(self, scale: float) -> None:
        if not hasattr(self, "goat_prior_scale"):
            return
        self.goat_prior_scale.fill_(float(scale))

    def set_prior_gate(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in ("none", "query_norm"):
            raise ValueError(f"mode must be 'none' or 'query_norm', got {mode!r}.")
        self.prior_gate = mode

    def _reset_parameters(self) -> None:
        if self.in_proj is not None:
            nn.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                nn.init.zeros_(self.in_proj.bias)
        if self.q_proj is not None:
            nn.init.xavier_uniform_(self.q_proj.weight)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
        if self.k_proj is not None:
            nn.init.xavier_uniform_(self.k_proj.weight)
            if self.k_proj.bias is not None:
                nn.init.zeros_(self.k_proj.bias)
        if self.v_proj is not None:
            nn.init.xavier_uniform_(self.v_proj.weight)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        if self.add_bias_kv:
            nn.init.normal_(self.bias_k, mean=0.0, std=1e-3)
            nn.init.normal_(self.bias_v, mean=0.0, std=1e-3)

        self._reset_prior_parameters()

    def _reset_prior_parameters(self) -> None:
        """
        Initialize ONLY the GOAT prior parameters according to self.prior_init.

        - "seeded": non-zero defaults
        - "uniform": starts close to neutral, but still learnable
        """
        mode = getattr(self, "prior_init", "seeded")
        if mode not in ("seeded", "uniform"):
            raise ValueError(f"Unknown prior_init={mode!r}")

        if mode == "uniform":
            nn.init.zeros_(self.lambda_sym)
            nn.init.zeros_(self.lambda_asym)
        else:  # "seeded"
            nn.init.normal_(self.lambda_sym, mean=0.0, std=1e-2)
            nn.init.zeros_(self.lambda_asym)

        if self.enable_key_bias:
            with torch.no_grad():
                nn.init.normal_(self.key_bias.fc1.weight, mean=0.0, std=self.init_scale_prior)
                nn.init.zeros_(self.key_bias.fc1.bias)
                nn.init.normal_(self.key_bias.fc2.weight, mean=0.0, std=self.init_scale_prior)
                nn.init.zeros_(self.key_bias.fc2.bias)

                self.sink_gain.fill_(1.0)
                self.sink_bump.zero_()
                self.goat_prior_scale.fill_(1.0)

                if mode == "uniform":
                    self.key_bias.fc2.weight.zero_()
                    self.key_bias.fc2.bias.zero_()

                    self.recency_slope_raw.fill_(-30.0)
                else:  # "seeded"
                    slopes = _alibi_slopes(
                        self.kv_num_heads,
                        device=self.recency_slope_raw.device,
                        dtype=torch.float32,
                    )
                    self.recency_slope_raw.copy_(_inv_softplus(slopes))
        else:
            # If key bias is disabled, still ensure schedule scalar is sane.
            with torch.no_grad():
                self.goat_prior_scale.fill_(1.0)

    def to(self, *args, **kwargs):
        # Keep the sink MLP in fp32 for stability, regardless of model dtype.
        ret = super().to(*args, **kwargs)
        if self.key_bias is not None:
            self.key_bias.to(dtype=torch.float32, device=next(self.parameters()).device)
        return ret


class GoatAttention(_GoatAttentionBase):
    """
    GOAT attention (preferred name).

    This class contains the full implementation.
    """

    def _cache_put(self, cache: "OrderedDict[Any, torch.Tensor]", key: Any, tensor: torch.Tensor) -> None:
        cache[key] = tensor
        cache.move_to_end(key)
        while len(cache) > getattr(self, "max_cache_entries", 8):
            cache.popitem(last=False)

    def _get_rel_feats(self, L: int, dtype: torch.dtype, device: torch.device, offset: int = 0) -> torch.Tensor:
        key = (L, offset, torch.float32, device)
        cached = self._rel_cache.get(key)
        if cached is None:
            freqs = self.rel_freqs_base.to(device=device, dtype=torch.float32)
            feats32 = _fourier_features(L, freqs, offset=offset)
            self._cache_put(self._rel_cache, key, feats32)
            cached = feats32
        return cached.to(dtype)

    def _get_abs_feats(self, L: int, dtype: torch.dtype, device: torch.device, offset: int = 0) -> torch.Tensor:
        key = (L, offset, torch.float32, device)
        cached = self._abs_cache.get(key)
        if cached is None:
            freqs = self.abs_freqs_base.to(device=device, dtype=torch.float32)
            feats32 = _fourier_features(L, freqs, offset=offset)
            self._cache_put(self._abs_cache, key, feats32)
            cached = feats32
        return cached.to(dtype)

    def _get_rel_feats_2d(
        self,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
        has_cls: bool,
        offset_h: int = 0,
        offset_w: int = 0,
    ) -> torch.Tensor:
        key = (H, W, offset_h, offset_w, torch.float32, device, bool(has_cls))
        cached = self._rel2d_cache.get(key)
        if cached is None:
            Rx = self.R // 2
            Ry = self.R - Rx
            fy = self.rel_freqs_base[:Ry].to(device=device, dtype=torch.float32)   # y gets first Ry
            fx = self.rel_freqs_base[Ry:Ry + Rx].to(device=device, dtype=torch.float32)  # x gets remaining Rx
            Fx = _fourier_features_axis(W, fx, offset=offset_w)  # (W, 2Rx)
            Fy = _fourier_features_axis(H, fy, offset=offset_h)  # (H, 2Ry)
            # Important: downstream expects features grouped by halves (first half, then second half).
            Fy_cos, Fy_sin = Fy.chunk(2, dim=-1)  # (H, Ry), (H, Ry)
            Fx_cos, Fx_sin = Fx.chunk(2, dim=-1)  # (W, Rx), (W, Rx)
            F_cos = torch.cat(
                [
                    Fy_cos[:, None, :].expand(H, W, Ry),
                    Fx_cos[None, :, :].expand(H, W, Rx),
                ],
                dim=-1,
            )  # (H, W, R)
            F_sin = torch.cat(
                [
                    Fy_sin[:, None, :].expand(H, W, Ry),
                    Fx_sin[None, :, :].expand(H, W, Rx),
                ],
                dim=-1,
            )  # (H, W, R)
            F = torch.cat([F_cos, F_sin], dim=-1).reshape(H * W, 2 * self.R)
            if has_cls:
                zeros = torch.zeros(1, 2 * self.R, device=device, dtype=torch.float32)
                out = torch.cat([zeros, F], dim=0)
            else:
                out = F
            self._cache_put(self._rel2d_cache, key, out)
            cached = out
        return cached.to(dtype)

    def _get_abs_feats_2d(
        self,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
        has_cls: bool,
        offset_h: int = 0,
        offset_w: int = 0,
    ) -> torch.Tensor:
        key = (H, W, offset_h, offset_w, torch.float32, device, bool(has_cls))
        cached = self._abs2d_cache.get(key)
        if cached is None:
            Mx = self.M // 2
            My = self.M - Mx
            fy = self.abs_freqs_base[:My].to(device=device, dtype=torch.float32)   # y gets first My
            fx = self.abs_freqs_base[My:My + Mx].to(device=device, dtype=torch.float32)  # x gets remaining Mx
            Fx = _fourier_features_axis(W, fx, offset=offset_w)
            Fy = _fourier_features_axis(H, fy, offset=offset_h)
            # Keep the same [cos_all, sin_all] convention for consistency.
            Fy_cos, Fy_sin = Fy.chunk(2, dim=-1)  # (H, My), (H, My)
            Fx_cos, Fx_sin = Fx.chunk(2, dim=-1)  # (W, Mx), (W, Mx)
            F_cos = torch.cat(
                [
                    Fy_cos[:, None, :].expand(H, W, My),
                    Fx_cos[None, :, :].expand(H, W, Mx),
                ],
                dim=-1,
            )  # (H, W, M)
            F_sin = torch.cat(
                [
                    Fy_sin[:, None, :].expand(H, W, My),
                    Fx_sin[None, :, :].expand(H, W, Mx),
                ],
                dim=-1,
            )  # (H, W, M)
            F = torch.cat([F_cos, F_sin], dim=-1).reshape(H * W, 2 * self.M)
            if has_cls:
                zeros = torch.zeros(1, 2 * self.M, device=device, dtype=torch.float32)
                out = torch.cat([zeros, F], dim=0)
            else:
                out = F
            self._cache_put(self._abs2d_cache, key, out)
            cached = out
        return cached.to(dtype)

    def clear_feature_caches(self) -> None:
        """
        Clear all positional feature caches (1D + 2D).
        """
        self._rel_cache.clear()
        self._abs_cache.clear()
        self._rel2d_cache.clear()
        self._abs2d_cache.clear()

    def _augment_abs_features(
        self,
        abs_fourier: torch.Tensor,   # (S, 2M) in dtype
        pos_offset: int,
    ) -> torch.Tensor:
        """
        Build stable absolute-position features for the sink MLP:
          [fourier(abs), t_norm, log_t_norm]
        """
        S = abs_fourier.size(0)
        device = abs_fourier.device

        # absolute positions (float32 for stability)
        t = (pos_offset + torch.arange(S, device=device, dtype=torch.float32)).view(S, 1)

        # normalize by training length if provided; else by current length
        denom = float(self.training_seq_len if self.training_seq_len is not None else max(S - 1, 1))
        t_norm = t / denom

        # log position normalized to ~[0,1]
        log_t = torch.log1p(t)
        log_denom = math.log1p(denom)
        log_t_norm = log_t / (log_denom if log_denom > 0 else 1.0)

        extra = torch.cat([t_norm, log_t_norm], dim=-1)  # (S,2)
        return torch.cat([abs_fourier.to(torch.float32), extra], dim=-1)  # (S, 2M+2) fp32

    # Prior visualization helper
    @torch.no_grad()
    def compute_log_prior(
        self,
        L: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        spatial_shape: Optional[Tuple[int, int]] = None,
        has_cls_token: Optional[bool] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Construct the GOAT positional log-prior matrix of shape (L, L),
        averaged over attention heads, using only the spectral prior +
        sink term (no content logits).
        """
        dev = device
        dt = dtype

        has_cls = self.has_cls_token_default if has_cls_token is None else bool(has_cls_token)

        # Positional features: 1D by default, or true 2D if spatial_shape is provided.
        if spatial_shape is not None:
            H_sp, W_sp = int(spatial_shape[0]), int(spatial_shape[1])
            if not _validate_spatial_shape(L, has_cls, H_sp, W_sp):
                raise ValueError(
                    f"compute_log_prior: L={L} incompatible with spatial_shape=({H_sp},{W_sp}) "
                    f"and has_cls_token={has_cls}."
                )
            rel = (
                self._get_rel_feats_2d(H_sp, W_sp, dt, dev, has_cls, offset_h=0, offset_w=0)
                if self.R > 0 else None
            )  # (L, 2R) or None
            abs_k = self._get_abs_feats_2d(H_sp, W_sp, dt, dev, has_cls, offset_h=0, offset_w=0) # (L, 2M)
        else:
            # 1D relative and absolute features
            rel = self._get_rel_feats(L, dt, dev, offset=0) if self.R > 0 else None  # (L, 2R) or None
            abs_k = self._get_abs_feats(L, dt, dev, offset=0) # (L, 2M)

        R = self.R
        N = 1  # single synthetic batch for visualization

        if R > 0:
            assert rel is not None
            # Split relative features into two halves.
            cos_rel, sin_rel = rel.chunk(2, dim=-1)  # (L, R) each

            # Spectral factors per KV head (same parametrization as in forward).
            lam_sym = self.lambda_sym.to(device=dev, dtype=dt)    # (H_kv, R)
            lam_asym = self.lambda_asym.to(device=dev, dtype=dt)  # (H_kv, R)

            # Repeat across Q heads if using GQA/MQA so each Q head in a group
            # shares the same spectral parameters as its KV head.
            if self.kv_num_heads == self.num_heads:
                A_heads = lam_sym                                 # (H, R)
                B_heads = lam_asym                                # (H, R)
            else:
                A_heads = lam_sym.repeat_interleave(self._qkv_group_size, dim=0)   # (H, R)
                B_heads = lam_asym.repeat_interleave(self._qkv_group_size, dim=0)  # (H, R)

            # Broadcast Fourier features for queries/keys
            cos_q = cos_rel.view(1, 1, L, R).expand(N, self.num_heads, L, R)
            sin_q = sin_rel.view(1, 1, L, R).expand(N, self.num_heads, L, R)
            cos_k = cos_rel.view(1, 1, L, R).expand(N, self.kv_num_heads, L, R)
            sin_k = sin_rel.view(1, 1, L, R).expand(N, self.kv_num_heads, L, R)

            # Reshape spectral weights for broadcasting: (1, H, 1, R)
            A = A_heads.view(1, self.num_heads, 1, R)
            B = B_heads.view(1, self.num_heads, 1, R)

            # Build query-side position features.
            q_pos_0 = (A * cos_q) + (B * sin_q)
            q_pos_1 = (A * sin_q) - (B * cos_q)
            q_pos = torch.cat([q_pos_0, q_pos_1], dim=-1)         # (N, H, L, 2R)

            # Keys use the raw relative features.
            k_pos = torch.cat([cos_k, sin_k], dim=-1)             # (N, H_kv, L, 2R)
        else:
            q_pos = torch.empty((N, self.num_heads, L, 0), device=dev, dtype=dt)
            k_pos = torch.empty((N, self.kv_num_heads, L, 0), device=dev, dtype=dt)

        # Sink term u(j): key-only lane
        if self.enable_key_bias:
            abs_in = self._augment_abs_features(abs_k, pos_offset=0)  # (L, 2M+2) fp32
            u_shared = self.key_bias(abs_in).squeeze(-1)  # (L,) fp32
            t = torch.arange(L, device=dev, dtype=torch.float32)  # (L,)
            is_bos = (t == 0).to(dtype=torch.float32)  # (L,)
            prior_scale = self.goat_prior_scale.to(device=dev, dtype=torch.float32)
            # Base sink: shared u(j) + explicit BOS bump
            u_h = prior_scale * (
                self.sink_gain.to(device=dev, dtype=torch.float32).view(self.kv_num_heads, 1) * u_shared.view(1, L)
                + self.sink_bump.to(device=dev, dtype=torch.float32).view(self.kv_num_heads, 1) * is_bos.view(1, L)
            )  # (H_kv,L)

            # Recency term is only meaningful under causal masking (key-linear == lag-linear up to a row-constant).
            if is_causal:
                slopes = F.softplus(self.recency_slope_raw.to(device=dev, dtype=torch.float32))  # (H_kv,)
                u_h = u_h + prior_scale * (slopes.view(self.kv_num_heads, 1) * t.view(1, L))

            phi_bias = torch.ones((N, self.num_heads, L, 1), dtype=dt, device=dev)
            psi_bias = u_h.to(dt).view(1, self.kv_num_heads, L, 1).expand(N, self.kv_num_heads, L, 1)
            q_pos = torch.cat([q_pos, phi_bias], dim=-1)
            k_pos = torch.cat([k_pos, psi_bias], dim=-1)

        # Align KV heads to Q heads when using GQA/MQA
        if self.kv_num_heads != self.num_heads:
            group = self.num_heads // self.kv_num_heads
            idx = (torch.arange(self.num_heads, device=dev) // group).to(torch.long)
            k_pos = k_pos.index_select(1, idx)                              # (N, H, L, Dp)

        # Drop synthetic batch dimension and compute per-head logits
        qh = q_pos.squeeze(0)                                               # (H, L, Dp)
        kh = k_pos.squeeze(0)                                               # (H, L, Dp)
        kh_T = kh.transpose(1, 2).contiguous()                              # (H, Dp, L)
        log_prior_heads = torch.matmul(qh, kh_T)                            # (H, L, L)

        # Average over heads → (L, L)
        log_prior = log_prior_heads.mean(dim=0)
        return log_prior

    @staticmethod
    def _merge_masks(
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        num_heads: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
        kv_num_heads: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Merge attn_mask (2D/3D/4D) and key_padding_mask into a single additive mask
        of shape (B*H, tgt_len, src_len) for scaled_dot_product_attention.

        Boolean masks are treated as {True = -inf, False = 0}.
        Float masks are assumed additive.
        """
        add_mask = None

        # attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                B, h, T, S = attn_mask.shape
                if S != src_len:
                    raise ValueError(f"4D attn_mask src dim S={S} doesn't match src_len={src_len}.")
                if T == 1:
                    attn_mask = attn_mask.expand(B, h, tgt_len, S)
                    T = tgt_len
                elif T != tgt_len:
                    raise ValueError(f"4D attn_mask tgt dim T={T} must be 1 or tgt_len={tgt_len}.")

                if h == 1:
                    attn_mask = attn_mask.expand(B, num_heads, T, S)
                elif h == num_heads:
                    pass
                elif kv_num_heads is not None and h == kv_num_heads:
                    # Mask is per KV-head (GQA/MQA). Repeat each KV-head mask for its query-head group.
                    group = num_heads // kv_num_heads
                    attn_mask = attn_mask.repeat_interleave(group, dim=1)
                else:
                    valid_h = f"1, {num_heads}"
                    if kv_num_heads is not None:
                        valid_h += f", or {kv_num_heads}"
                    raise ValueError(f"4D attn_mask second dim h={h} must be {valid_h}, got {h}.")
                attn_mask = attn_mask.reshape(B * num_heads, T, S)
            elif attn_mask.dim() == 3 and attn_mask.size(0) == batch_size:
                if attn_mask.size(1) != tgt_len or attn_mask.size(2) != src_len:
                    raise ValueError(
                        f"3D attn_mask shape {(attn_mask.size(1), attn_mask.size(2))} "
                        f"doesn't match {(tgt_len, src_len)}."
                    )
                attn_mask = attn_mask.unsqueeze(1).expand(batch_size, num_heads, tgt_len, src_len)
                attn_mask = attn_mask.reshape(batch_size * num_heads, tgt_len, src_len)
            elif (
                attn_mask.dim() == 3 and
                kv_num_heads is not None and
                attn_mask.size(0) == batch_size * kv_num_heads and
                kv_num_heads != num_heads
            ):
                # (B*H_kv, T, S) -> (B*H, T, S) for GQA/MQA
                if attn_mask.size(1) != tgt_len or attn_mask.size(2) != src_len:
                    raise ValueError(
                        f"3D attn_mask shape {(attn_mask.size(1), attn_mask.size(2))} "
                        f"doesn't match {(tgt_len, src_len)}."
                    )
                group = num_heads // kv_num_heads
                attn_mask = attn_mask.view(batch_size, kv_num_heads, tgt_len, src_len)
                attn_mask = attn_mask.repeat_interleave(group, dim=1)
                attn_mask = attn_mask.reshape(batch_size * num_heads, tgt_len, src_len)

            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    am = torch.zeros((tgt_len, src_len), device=device, dtype=dtype)
                    am = am.masked_fill(attn_mask, float("-inf"))
                    add_mask = am.unsqueeze(0)
                elif attn_mask.dim() == 3:
                    if attn_mask.size(-2) != tgt_len or attn_mask.size(-1) != src_len:
                        raise ValueError("Boolean attn_mask 3D must be (B*H, T, S) or (1, T, S).")
                    am = torch.zeros_like(attn_mask, dtype=dtype)
                    am = am.masked_fill(attn_mask, float("-inf"))
                    add_mask = am
                else:
                    raise ValueError("Boolean attn_mask must be 2D or 3D.")
            else:
                am = attn_mask.to(device=device, dtype=dtype)
                if am.dim() == 2:
                    add_mask = am.unsqueeze(0)
                elif am.dim() == 3 and am.size(0) == batch_size:
                    am = am.unsqueeze(1).expand(batch_size, num_heads, tgt_len, src_len)
                    add_mask = am.reshape(batch_size * num_heads, tgt_len, src_len)
                elif am.dim() == 3:
                    add_mask = am
                else:
                    raise ValueError("attn_mask must be 2D or 3D after reshaping.")

        # key_padding_mask: (B, S) bool, True = ignore
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=device)
            if kpm.dtype != torch.bool:
                kpm = kpm.to(torch.bool)
            kpm_add = torch.zeros((batch_size, 1, 1, src_len), device=device, dtype=dtype)
            kpm_add = kpm_add.masked_fill(kpm[:, None, None, :], float("-inf"))
            kpm_add = kpm_add.expand(batch_size, num_heads, 1, src_len).reshape(batch_size * num_heads, 1, src_len)
            add_mask = kpm_add if add_mask is None else (add_mask + kpm_add)

        if add_mask is not None:
            if add_mask.dim() == 3:
                if add_mask.size(0) not in (1, batch_size * num_heads):
                    add_mask = add_mask.expand(batch_size * num_heads, tgt_len, src_len)
                else:
                    add_mask = add_mask.expand(-1, tgt_len, src_len)
            else:
                raise AssertionError("add_mask must be 3D at this point.")

        return add_mask

    @classmethod
    def for_gpt(cls, embed_dim, num_heads, kv_num_heads=None, **kw):
        """
        Factory for GPT-style decoder (causal, KV cache, GQA/MQA).
        """
        defaults = {
            "batch_first": True,
            "pos_rank": 0,
            "abs_rank": 8,
            "enable_key_bias": True,
            "prior_init": "seeded",
            "prior_gate": "none",
            "pos_encoding": "1d",
            "fuse_qkv": True,
            "training_seq_len": 2048,
        }
        merged = {**defaults, **kw}
        return cls(embed_dim=embed_dim, num_heads=num_heads, kv_num_heads=kv_num_heads, **merged)

    @classmethod
    def for_vit(cls, embed_dim, num_heads, **kw):
        """
        Factory for ViT-style attention (2D patches, CLS token).
        """
        defaults = {
            "batch_first": True,
            "pos_rank": 8,
            "abs_rank": 8,
            "enable_key_bias": True,
            "prior_init": "uniform",
            "prior_gate": "none",
            "pos_encoding": "2d",
            "has_cls_token_default": True,
        }
        merged = {**defaults, **kw}
        return cls(embed_dim=embed_dim, num_heads=num_heads, **merged)

    @classmethod
    def for_bert(cls, embed_dim, num_heads, **kw):
        """
        Factory for BERT-style encoder (bidirectional, 1D positions).
        """
        defaults = {
            "batch_first": True,
            "pos_rank": 8,
            "abs_rank": 8,
            "enable_key_bias": True,
            "prior_init": "uniform",
            "prior_gate": "none",
            "pos_encoding": "1d",
        }
        merged = {**defaults, **kw}
        return cls(embed_dim=embed_dim, num_heads=num_heads, **merged)

    @classmethod
    def from_torch_mha(cls, mha: nn.MultiheadAttention, **overrides):
        """
        Wrap an existing torch.nn.MultiheadAttention with GOAT + sink priors.
        Copies Q/K/V/out weights when dimensions match.
        """
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        dropout = mha.dropout
        bias = (mha.in_proj_bias is not None)
        batch_first = getattr(mha, "batch_first", False)
        kdim = getattr(mha, "kdim", None)
        vdim = getattr(mha, "vdim", None)
        add_bias_kv = (getattr(mha, "bias_k", None) is not None) or (getattr(mha, "bias_v", None) is not None)
        add_zero_attn = getattr(mha, "add_zero_attn", False)

        m = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            kdim=kdim,
            vdim=vdim,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kv_num_heads=num_heads,   # 1:1 copy
            fuse_qkv=False,
            **overrides,
        )

        with torch.no_grad():
            if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
                W = mha.in_proj_weight  # (3E, E)
                qW, kW, vW = W[:embed_dim], W[embed_dim:2*embed_dim], W[2*embed_dim:]
                m.q_proj.weight.copy_(qW)
                m.k_proj.weight.copy_(kW)
                m.v_proj.weight.copy_(vW)
                if mha.in_proj_bias is not None:
                    b = mha.in_proj_bias  # (3E,)
                    qb, kb, vb = b[:embed_dim], b[embed_dim:2*embed_dim], b[2*embed_dim:]
                    if m.q_proj.bias is not None: m.q_proj.bias.copy_(qb)
                    if m.k_proj.bias is not None: m.k_proj.bias.copy_(kb)
                    if m.v_proj.bias is not None: m.v_proj.bias.copy_(vb)
            else:
                if hasattr(mha, "q_proj_weight") and mha.q_proj_weight is not None:
                    m.q_proj.weight.copy_(mha.q_proj_weight)
                if hasattr(mha, "k_proj_weight") and mha.k_proj_weight is not None:
                    m.k_proj.weight.copy_(mha.k_proj_weight)
                if hasattr(mha, "v_proj_weight") and mha.v_proj_weight is not None:
                    m.v_proj.weight.copy_(mha.v_proj_weight)
                if hasattr(mha, "in_proj_bias") and mha.in_proj_bias is not None and m.q_proj.bias is not None:
                    b = mha.in_proj_bias
                    qb, kb, vb = b[:embed_dim], b[embed_dim:2*embed_dim], b[2*embed_dim:]
                    m.q_proj.bias.copy_(qb); m.k_proj.bias.copy_(kb); m.v_proj.bias.copy_(vb)

            m.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None and m.out_proj.bias is not None:
                m.out_proj.bias.copy_(mha.out_proj.bias)

            if getattr(mha, "bias_k", None) is not None and getattr(m, "bias_k", None) is not None:
                m.bias_k.copy_(mha.bias_k)
            if getattr(mha, "bias_v", None) is not None and getattr(m, "bias_v", None) is not None:
                m.bias_v.copy_(mha.bias_v)

        return m

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        cls_attn_only: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        spatial_shape: Optional[Tuple[int, int]] = None,
        has_cls_token: Optional[bool] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_present_kv: bool = False,
        position_offset_q: Optional[Union[int, Tuple[int, int]]] = None,
        position_offset_k: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            query, key, value: (L, N, E) or (N, L, E) depending on batch_first.
            key_padding_mask: (N, S) bool, True = ignore.
            attn_mask: bool or additive float; supports 2D/3D/4D HF-style masks.
            is_causal: use causal masking.
            spatial_shape: (H, W) for 2D layouts (ViT).
            has_cls_token: override has_cls_token_default.
            past_key_value: KV cache for streaming: (k_past, v_past).
            position_offset_q, position_offset_k: global positions (for long contexts).

        Returns:
            attn_output: same shape as query (batch_first respected).
            attn_weights (optional): (N, L, S) or (N, H, L, S).
            present_key_value (optional): if return_present_kv=True.
        """
        if self.batch_first:
            query = query.transpose(0, 1)  # (L, N, E)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        L, N, _E = query.shape
        S, N_k, _K = key.shape
        if N != N_k:
            raise ValueError("Batch sizes of query and key must match.")

        # HF-style attention_mask convenience
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is not None:
            if attention_mask.dim() == 2 and key_padding_mask is None and attn_mask is None:
                # 2D [B, S] pad mask (1 = keep, 0 = pad)
                key_padding_mask = (attention_mask == 0)
            elif attn_mask is None:
                attn_mask = attention_mask

        # Infer past length for offsets
        S_past_infer = 0
        if past_key_value is not None:
            k_past = past_key_value[0]
            S_past_infer = int(k_past.size(2))

        if position_offset_q is None:
            position_offset_q = S_past_infer if is_causal else 0
        if position_offset_k is None:
            position_offset_k = S_past_infer if is_causal else 0

        same_obj = (query is key) and (key is value)
        same_ptr = (query.data_ptr() == key.data_ptr() == value.data_ptr())

        # Q/K/V projections
        if self.in_proj is not None:
            if not (same_obj or same_ptr):
                raise RuntimeError(
                    "GoatAttention with fuse_qkv=True only supports self-attention. "
                    "Use fuse_qkv=False for cross-attention."
                )
            qkv = self.in_proj(query)  # (L, N, E + 2*kvE)
            q_lin, k_lin, v_lin = torch.split(
                qkv,
                [self.embed_dim, self.kv_embed_dim, self.kv_embed_dim],
                dim=-1,
            )
            q, k, v = q_lin, k_lin, v_lin
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Optional bias_kv / zero_attn tokens
        appended = 0
        if self.add_bias_kv:
            bias_k = self.bias_k.expand(1, N, -1)
            bias_v = self.bias_v.expand(1, N, -1)
            k = torch.cat([k, bias_k], dim=0)
            v = torch.cat([v, bias_v], dim=0)
            S += 1
            appended += 1
            if key_padding_mask is not None:
                pad_col = torch.zeros((N, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([key_padding_mask, pad_col], dim=1)

        if self.add_zero_attn:
            zero = torch.zeros((1, N, self.kv_embed_dim), dtype=k.dtype, device=k.device)
            k = torch.cat([k, zero], dim=0)
            v = torch.cat([v, zero], dim=0)
            S += 1
            appended += 1
            if key_padding_mask is not None:
                pad_col = torch.zeros((N, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([key_padding_mask, pad_col], dim=1)

        # Reshape to (N, H_q/H_kv, L/S, D)
        q = q.view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()       # (N, H, L, D)
        k = k.view(S, N, self.kv_num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()    # (N, H_kv, S, D)
        v = v.view(S, N, self.kv_num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()    # (N, H_kv, S, D)

        D_c = self.content_dim
        q_content = q[..., :D_c]
        k_content = k[..., :D_c]

        # Use full V projection as payload; only Q/K are structurally split
        v_total = v

        device, dtype = q.device, q.dtype
        R = self.R

        # 1D vs 2D positional routing
        has_cls = self.has_cls_token_default if has_cls_token is None else bool(has_cls_token)
        use_2d = False
        H_sp = W_sp = None
        mode = getattr(self, "pos_encoding", "auto")

        if mode in ("auto", "2d"):
            if spatial_shape is not None:
                H_sp, W_sp = int(spatial_shape[0]), int(spatial_shape[1])
                S_struct = S - appended
                if _validate_spatial_shape(L, has_cls, H_sp, W_sp) and _validate_spatial_shape(S_struct, has_cls, H_sp, W_sp):
                    use_2d = True
            else:
                # Infer spatial shape from the KEY sequence length S (the context),
                # then require that the query length L is also grid-aligned.
                S_struct = S - appended
                hw = _infer_hw_from_length(S_struct, has_cls)
                if hw is not None and _validate_spatial_shape(L, has_cls, hw[0], hw[1]):
                    H_sp, W_sp = hw
                    use_2d = True

        if mode == "2d" and not use_2d:
            raise ValueError(
                "pos_encoding='2d' but spatial_shape is missing or incompatible. "
                "Pass spatial_shape=(H, W) and set has_cls_token correctly."
            )

        def _as_2d_offset(off, name: str) -> Tuple[int, int]:
            if off is None:
                return (0, 0)
            if isinstance(off, (tuple, list)):
                if len(off) != 2:
                    raise ValueError(f"{name} must be an int or (offset_h, offset_w).")
                return (int(off[0]), int(off[1]))
            off_i = int(off)
            if off_i != 0:
                raise ValueError(f"{name} must be (offset_h, offset_w) in 2D mode (got {off_i}).")
            return (0, 0)

        def _as_1d_offset(off, name: str) -> int:
            if isinstance(off, (tuple, list)):
                raise ValueError(f"{name} must be an int in 1D mode (got {off}).")
            return int(off)

        if use_2d:
            oq_h, oq_w = _as_2d_offset(position_offset_q, "position_offset_q")
            ok_h, ok_w = _as_2d_offset(position_offset_k, "position_offset_k")
            # Keep 1D offsets for causal masking / bookkeeping paths that are sequence-index based.
            position_offset_q_1d = 0
            position_offset_k_1d = 0
            if is_causal and ((oq_h, oq_w) != (0, 0) or (ok_h, ok_w) != (0, 0)):
                raise ValueError(
                    "Non-zero 2D position offsets are ambiguous under causal masking. "
                    "Use position_offset_q/k=0 (or None) when is_causal=True in 2D mode."
                )
        else:
            position_offset_q = _as_1d_offset(position_offset_q, "position_offset_q")
            position_offset_k = _as_1d_offset(position_offset_k, "position_offset_k")
            position_offset_q_1d = int(position_offset_q)
            position_offset_k_1d = int(position_offset_k)

        # Relative Fourier features (queries & keys) + absolute features (keys) for sink
        if use_2d:
            rel_q = (
                self._get_rel_feats_2d(
                    H_sp, W_sp, dtype, device, has_cls,
                    offset_h=oq_h,
                    offset_w=oq_w,
                )
                if R > 0 else None
            )  # (L, 2R) or None
            rel_k = (
                self._get_rel_feats_2d(
                    H_sp, W_sp, dtype, device, has_cls,
                    offset_h=ok_h,
                    offset_w=ok_w,
                )
                if R > 0 else None
            )  # (S, 2R) or None
            abs_k = self._get_abs_feats_2d(
                H_sp, W_sp, dtype, device, has_cls,
                offset_h=ok_h,
                offset_w=ok_w,
            )  # (S, 2M)

            if appended > 0:
                if rel_k is not None:
                    rel_k = torch.cat([rel_k, rel_k.new_zeros(appended, rel_k.size(-1))], dim=0)
                abs_k = torch.cat([abs_k, abs_k.new_zeros(appended, abs_k.size(-1))], dim=0)
        else:
            rel_q = self._get_rel_feats(L, dtype, device, offset=position_offset_q) if R > 0 else None  # (L, 2R) or None
            rel_k = self._get_rel_feats(S, dtype, device, offset=position_offset_k) if R > 0 else None  # (S, 2R) or None
            abs_k = self._get_abs_feats(S, dtype, device, offset=position_offset_k)  # (S, 2M)

        if appended > 0 and not use_2d:
            if rel_k is not None:
                rel_k = rel_k.clone()
                rel_k[-appended:, :] = 0.0
            abs_k = abs_k.clone()
            abs_k[-appended:] = 0.0

        if R > 0:
            assert rel_q is not None and rel_k is not None
            lam_sym_fp32 = self.lambda_sym.to(device=device, dtype=torch.float32)    # (H_kv, R)
            lam_asym_fp32 = self.lambda_asym.to(device=device, dtype=torch.float32)  # (H_kv, R)
            lam_sym_fp32 = lam_sym_fp32.clamp(min=-1e6, max=1e6)
            lam_asym_fp32 = lam_asym_fp32.clamp(min=-1e6, max=1e6)

            lam_sym = lam_sym_fp32.to(dtype=dtype)     # (H_kv, R)
            lam_asym = lam_asym_fp32.to(dtype=dtype)   # (H_kv, R)

            if self.kv_num_heads == self.num_heads:
                A_heads = lam_sym                                      # (H, R)
                B_heads = lam_asym                                     # (H, R)
            else:
                A_heads = lam_sym.repeat_interleave(self._qkv_group_size, dim=0)   # (H, R)
                B_heads = lam_asym.repeat_interleave(self._qkv_group_size, dim=0)  # (H, R)

            # Split relative features into two halves.
            cos_q, sin_q = rel_q.chunk(2, dim=-1)  # (L, R)
            cos_k, sin_k = rel_k.chunk(2, dim=-1)  # (S, R)

            cos_q_full = cos_q.view(1, 1, L, R).expand(N, self.num_heads, L, R)
            sin_q_full = sin_q.view(1, 1, L, R).expand(N, self.num_heads, L, R)
            cos_k_full = cos_k.view(1, 1, S, R).expand(N, self.kv_num_heads, S, R)
            sin_k_full = sin_k.view(1, 1, S, R).expand(N, self.kv_num_heads, S, R)

            # Build positional subspace (spectral part) via rotated queries:
            # A = symmetric cos-term weight, B = antisymmetric sin-term weight.
            A = A_heads.view(1, self.num_heads, 1, R)
            B = B_heads.view(1, self.num_heads, 1, R)

            q_pos_0 = (A * cos_q_full) + (B * sin_q_full)
            q_pos_1 = (A * sin_q_full) - (B * cos_q_full)
            q_pos = torch.cat([q_pos_0, q_pos_1], dim=-1)    # (N, H, L, 2R)

            # Keys use the raw relative features.
            k_pos = torch.cat([cos_k_full, sin_k_full], dim=-1)  # (N, H_kv, S, 2R)
        else:
            q_pos = q.new_empty((N, self.num_heads, L, 0))
            k_pos = k.new_empty((N, self.kv_num_heads, S, 0))

        use_gate = (getattr(self, "prior_gate", "none") == "query_norm")
        g: Optional[torch.Tensor] = None

        if use_gate:
            qc_norm = q_content.to(torch.float32).pow(2).sum(dim=-1).sqrt()  # (N,H,L) fp32
            scale = F.softplus(self.prior_gate_scale_raw).view(1, self.num_heads, 1)  # (1,H,1)
            bias = self.prior_gate_bias.view(1, self.num_heads, 1)                    # (1,H,1)
            g = torch.sigmoid(bias - scale * qc_norm).to(dtype)                       # (N,H,L)

            if q_pos.size(-1) > 0:
                q_pos = q_pos * g.unsqueeze(-1)

        if self.enable_key_bias:
            abs_in = self._augment_abs_features(abs_k, pos_offset=int(position_offset_k_1d))  # (S,2M+2) fp32
            u_shared = self.key_bias(abs_in).squeeze(-1)  # (S,) fp32

            t = (int(position_offset_k_1d) + torch.arange(S, device=device, dtype=torch.float32))  # (S,) fp32
            is_bos = (t == 0).to(dtype=torch.float32)  # (S,)
            prior_scale = self.goat_prior_scale.to(device=device, dtype=torch.float32)
            u_h = prior_scale * (
                self.sink_gain.view(self.kv_num_heads, 1) * u_shared.view(1, S)
                + self.sink_bump.view(self.kv_num_heads, 1) * is_bos.view(1, S)
            )  # (H_kv,S)

            if is_causal:
                slopes = F.softplus(self.recency_slope_raw)  # (H_kv,) fp32
                u_h = u_h + prior_scale * (slopes.view(self.kv_num_heads, 1) * t.view(1, S))

            if appended > 0:
                u_h[:, -appended:] = 0.0

            if use_gate:
                assert g is not None
                phi_bias = g.unsqueeze(-1)  # (N,H,L,1)
            else:
                phi_bias = q_pos.new_ones((N, self.num_heads, L, 1))
            psi_bias = u_h.to(dtype).view(1, self.kv_num_heads, S, 1).expand(N, self.kv_num_heads, S, 1)
            q_pos = torch.cat([q_pos, phi_bias], dim=-1)
            k_pos = torch.cat([k_pos, psi_bias], dim=-1)

        D_total = self.head_dim
        q_total = q.new_empty((N, self.num_heads, L, D_total))
        k_total = k.new_empty((N, self.kv_num_heads, S, D_total))

        q_total[..., :D_c].copy_(q_content / math.sqrt(D_c))
        k_total[..., :D_c].copy_(k_content)
        q_total[..., D_c:].copy_(q_pos)
        k_total[..., D_c:].copy_(k_pos)

        present_key_value = None
        S_past = 0
        if past_key_value is not None:
            k_past, v_past = past_key_value
            if k_past.dim() != 4 or v_past.dim() != 4:
                raise ValueError("past_key_value must be (k, v) with shape (N, kv_heads, S_past, D).")
            if k_past.shape[:2] != (N, self.kv_num_heads) or v_past.shape[:2] != (N, self.kv_num_heads):
                raise ValueError("past_key_value head dims must be (N, kv_num_heads, ...).")
            if k_past.shape[-1] != self.head_dim or v_past.shape[-1] != self.head_dim:
                raise ValueError("past_key_value last dim must be head_dim.")

            k_total = torch.cat([k_past, k_total], dim=2)  # (N, H_kv, S_total, D)
            v_total = torch.cat([v_past, v_total], dim=2)
            S_past = k_past.size(2)
            S = k_total.size(2)

            if key_padding_mask is not None:
                pad = torch.zeros((N, S_past), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        if use_cache or return_present_kv:
            present_key_value = (k_total, v_total)

        if self.kv_num_heads != self.num_heads:
            k_total = k_total.repeat_interleave(self._qkv_group_size, dim=1)  # (N, H, S, D)
            v_total = v_total.repeat_interleave(self._qkv_group_size, dim=1)

        merged_mask = self._merge_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_size=N,
            num_heads=self.num_heads,
            tgt_len=L,
            src_len=S,
            device=device,
            dtype=dtype,
            kv_num_heads=self.kv_num_heads,
        )
        if merged_mask is not None and merged_mask.dtype != dtype:
            merged_mask = merged_mask.to(dtype)

        # Keep extreme positive mask values from producing NaNs.
        if merged_mask is not None:
            finite = torch.isfinite(merged_mask)
            too_pos = merged_mask > 80.0
            merged_mask = torch.where(
                finite & too_pos,
                torch.tensor(80.0, device=device, dtype=dtype),
                merged_mask,
            )

        # SDPA forbids is_causal=True when attn_mask is not None.
        use_internal_causal = (
            is_causal
            and (S_past == 0)
            and (appended == 0)
            and (merged_mask is None)
        )

        if is_causal and not use_internal_causal:
            i_idx = torch.arange(L, device=device).view(L, 1)  # query index within chunk
            j_idx = torch.arange(S, device=device).view(1, S)  # key index within [past | current]
            pos_delta = int(position_offset_q_1d) - int(position_offset_k_1d)
            allowed = j_idx <= (S_past + pos_delta + i_idx)
            if appended > 0:
                allowed[..., S - appended:] = True
            causal_add = torch.zeros((N * self.num_heads, L, S), dtype=dtype, device=device)
            causal_add = causal_add.masked_fill(~allowed, float("-inf"))
            merged_mask = causal_add if merged_mask is None else (merged_mask + causal_add)

        # Run SDPA.
        # IMPORTANT: keep 4D (B,H,L,D) tensors so Flash/mem-efficient/cuDNN kernels are eligible.
        # Query scaling is adjusted to match this module's logits.
        q_in = (q_total * math.sqrt(D_total)).contiguous()  # (N, H, L, D_total)
        k_in = k_total.contiguous()  # (N, H, S, D_total)
        v_in = v_total.contiguous()  # (N, H, S, D_total)

        attn_mask_4d = None
        if merged_mask is not None:
            # merged_mask is (1, L, S) or (N*H, L, S). SDPA wants broadcastable to (N, H, L, S).
            if merged_mask.size(0) == 1:
                attn_mask_4d = merged_mask.reshape(1, 1, L, S)
            else:
                attn_mask_4d = merged_mask.reshape(N, self.num_heads, L, S)

        attn_output = F.scaled_dot_product_attention(
            q_in,
            k_in,
            v_in,
            attn_mask=attn_mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_internal_causal,
        )  # (N, H, L, D_total)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(N, L, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        attn_weights_out: Optional[torch.Tensor] = None
        if need_weights:
            q_total_bh = q_total.reshape(N * self.num_heads, L, D_total)
            k_total_bh = k_total.reshape(N * self.num_heads, S, D_total)

            if cls_attn_only:
                q_cls = q_total_bh[:, 0:1, :]  # (B*H, 1, D)
                logits = torch.bmm(q_cls, k_total_bh.transpose(1, 2))  # (B*H, 1, S)
                if merged_mask is not None:
                    logits = logits + merged_mask[:, 0:1, :]
                if is_causal:
                    j_idx = torch.arange(S, device=device).view(1, S)
                    pos_delta = int(position_offset_q_1d) - int(position_offset_k_1d)
                    causal = j_idx <= (S_past + pos_delta + 0)
                    if appended > 0:
                        causal[..., S - appended:] = True
                    logits = logits.masked_fill(~causal, float("-inf"))
                weights = F.softmax(logits, dim=-1)  # (B*H, 1, S)
                weights = weights.view(N, self.num_heads, 1, S)
                attn_weights_out = weights.mean(dim=1) if average_attn_weights else weights
            else:
                logits = torch.bmm(q_total_bh, k_total_bh.transpose(1, 2))
                if merged_mask is not None:
                    logits = logits + merged_mask
                if is_causal:
                    i_idx = torch.arange(L, device=device).view(L, 1)
                    j_idx = torch.arange(S, device=device).view(1, S)
                    pos_delta = int(position_offset_q_1d) - int(position_offset_k_1d)
                    causal = j_idx <= (S_past + pos_delta + i_idx)
                    if appended > 0:
                        causal[..., S - appended:] = True
                    logits = logits.masked_fill(~causal, float("-inf"))
                weights = F.softmax(logits, dim=-1)
                weights = weights.view(N, self.num_heads, L, S)
                attn_weights_out = weights.mean(dim=1) if average_attn_weights else weights

        if return_present_kv:
            return attn_output, attn_weights_out, present_key_value
        else:
            return attn_output, attn_weights_out

