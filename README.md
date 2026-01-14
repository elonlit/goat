## GOAT Attention

[![PyPI](https://img.shields.io/pypi/v/goat-attention.svg)](https://pypi.org/project/goat-attention/)
![Python](https://img.shields.io/pypi/pyversions/goat-attention.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Generalized Optimal Transport Attention with Trainable Priors (**GOAT**), available as a PyTorch multi-head attention module.

> **Install name:** `goat-attention` (PyPI) · **Import name:** `goat`

<p align="center">
  <img src="figs/goated.png" alt="GOAT Attention" width="320" />
</p>

## Installation

- **From PyPI (recommended)**:

```bash
uv add goat-attention
```

- **pip**:

```bash
pip install goat-attention
```

- **From source (editable)**:

```bash
uv pip install -e .
```

- **From source (editable, pip)**:

```bash
pip install -e .
```

## Quickstart

```python
import torch
from goat import GoatAttention

B, L, S, E, H = 2, 5, 7, 64, 8
xq = torch.randn(B, L, E)
xk = torch.randn(B, S, E)
xv = torch.randn(B, S, E)

attn = GoatAttention(
    embed_dim=E,
    num_heads=H,
    batch_first=True,
    pos_rank=2,
    abs_rank=4,
    enable_key_bias=True,
)

out, weights = attn(xq, xk, xv, is_causal=False, need_weights=True)
print(out.shape, None if weights is None else weights.shape)
```

## Key Parameters

GOAT uses spectral (Fourier) features to model positional relationships. The two main hyperparameters are:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `pos_rank` | Number of Fourier frequencies for **relative** position encoding. Controls how finely the model can distinguish between different relative distances. Higher values = more expressive distance modeling. | 2–16 |
| `abs_rank` | Number of Fourier frequencies for **absolute** position encoding, used by the learned "sink" bias `u(j)`. Controls position-dependent attention patterns (e.g., attending to sequence start). | 2–8 |

**Guidelines:**
- For **language models** (GPT-style): `pos_rank=4, abs_rank=4` is a good starting point
- For **vision transformers** (ViT): `pos_rank=16, abs_rank=2` works well with 2D positional encoding
- For **long-context tasks**: Higher `pos_rank` may help capture fine-grained positional patterns
- Set `enable_key_bias=False` to disable the sink term entirely (removes `abs_rank` dependency)

## CLI

After installation:

```bash
goat info
goat smoke
```

## Documentation

See `docs/`:

- [`docs/index.md`](docs/index.md)
- [`docs/usage.md`](docs/usage.md)
- [`docs/api.md`](docs/api.md)
- [`docs/development.md`](docs/development.md)

## Development

```bash
uv pip install -e ".[dev]"
pytest
```

## License

MIT (see `LICENSE`).

## Citation

If you find GOAT useful, please cite:

```bibtex
@misc{goat,
  title         = {You Need Better Attention Priors},
  author        = {Litman, Elon and ...},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

