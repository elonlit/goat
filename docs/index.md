## GOAT Attention

This project packages **GOAT-style attention** as a reusable PyTorch module: `goat.GoatAttention`.

- **Main feature**: spectral relative prior + optional key-only sink term (see class docstring).
- **Intended use**: drop-in attention replacement in transformer blocks (GPT-style, ViT-style, etc.).

### Install

```bash
uv pip install -e .
```

### Quickstart

```python
import torch
from goat import GoatAttention

attn = GoatAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 5, 64)
y, _ = attn(x, x, x, is_causal=False)
print(y.shape)
```


