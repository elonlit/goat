from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def _fourier_features(length: int, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    [cos(ω_k t), sin(ω_k t)] for t = offset .. offset + length - 1

    Args:
        length: sequence length L
        freqs: (K,) tensor of frequencies ω_k
        offset: starting position index

    Returns:
        (L, 2K) tensor
    """
    t = offset + torch.arange(length, device=freqs.device, dtype=freqs.dtype)  # (L,)
    phases = t[:, None] * freqs[None, :]  # (L, K)
    feats = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)  # (L, 2K)
    return feats


@torch.no_grad()
def _fourier_features_axis(n: int, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Axis-wise Fourier features for 2D grids.

    Args:
        n: number of positions along one axis
        freqs: (K,) tensor of frequencies
        offset: starting coordinate index

    Returns:
        (n, 2K) tensor of [cos, sin] features
    """
    coord = offset + torch.arange(n, device=freqs.device, dtype=freqs.dtype)[:, None]  # (n, 1)
    phases = coord * freqs[None, :]  # (n, K)
    return torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)  # (n, 2K)


def _validate_spatial_shape(tokens: int, has_cls: bool, H: int, W: int) -> bool:
    """
    Check that number of tokens matches H×W (plus optional CLS).
    """
    n_patches = tokens - (1 if has_cls else 0)
    return (H > 0) and (W > 0) and (H * W == n_patches)


def _infer_hw_from_length(tokens: int, has_cls: bool) -> Optional[Tuple[int, int]]:
    """
    Try to infer H×W from token count (assuming square grid).
    """
    n_patches = tokens - (1 if has_cls else 0)
    if n_patches <= 0:
        return None
    s = int(round(n_patches**0.5))
    if s * s == n_patches:
        return (s, s)
    return None


