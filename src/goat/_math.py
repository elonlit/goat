from __future__ import annotations

import torch


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse of softplus for y>0: x = log(exp(y) - 1).
    """
    y = torch.clamp(y, min=1e-6)
    return torch.log(torch.expm1(y))


