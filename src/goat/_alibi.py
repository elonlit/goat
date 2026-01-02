from __future__ import annotations

import math

import torch


def _alibi_slopes(n_heads: int, *, device=None, dtype=None) -> torch.Tensor:
    """
    Returns ALiBi slopes for `n_heads` as a 1D tensor (n_heads,).

    This is the widely used reference algorithm from the ALiBi paper's ecosystem.
    """
    if dtype is None:
        dtype = torch.float32

    def _get_slopes_power_of_2(n: int):
        start = 2 ** (-2 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if float(math.log2(n_heads)).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest_power_of_2)
        slopes_extra = _alibi_slopes(2 * closest_power_of_2, device=device, dtype=dtype)[0::2]
        slopes = slopes + slopes_extra[: (n_heads - closest_power_of_2)].tolist()

    return torch.tensor(slopes, device=device, dtype=dtype)


