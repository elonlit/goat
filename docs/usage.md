## Usage

### Importing

```python
from goat import GoatAttention
```

### GPT-style (causal + optional KV-cache)

`GoatAttention` supports causal attention and a KV-cache interface. For convenience:

- `GoatAttention.for_gpt(...)`

### ViT-style (2D positional structure)

`GoatAttention` can route positional logic as 1D/2D/auto:

- `GoatAttention.for_vit(...)`

When using 2D structure, provide `spatial_shape=(H, W)` and whether a CLS token is present.

### CLI smoke test

```bash
goat smoke
```


