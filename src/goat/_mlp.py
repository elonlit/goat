from __future__ import annotations

import torch
import torch.nn as nn


class _SmallMLP(nn.Module):
    """
    Tiny MLP used for the absolute-position sink term u(j).

    Architecture: in_dim -> hidden -> out_dim with SiLU activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 2,
        init_scale: float = 1e-3,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden = min(in_dim * hidden_mult, max(out_dim * 2, 4))

        self.fc1 = nn.Linear(in_dim, hidden, bias=True, **factory_kwargs)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True, **factory_kwargs)
        self.act = nn.SiLU()

        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


