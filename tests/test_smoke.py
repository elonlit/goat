from __future__ import annotations

import torch

from goat import GoatAttention


def test_smoke_1d_shapes():
    torch.manual_seed(0)
    B, L, S, E, H = 2, 5, 7, 64, 8
    xq = torch.randn(B, L, E)
    xk = torch.randn(B, S, E)
    xv = torch.randn(B, S, E)

    m = GoatAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.0,
        batch_first=True,
        pos_rank=2,
        abs_rank=4,
        enable_key_bias=True,
    )
    out, attn = m(xq, xk, xv, is_causal=False, need_weights=True)
    assert out.shape == (B, L, E)
    assert attn is not None
    assert attn.shape == (B, L, S)


def test_smoke_gpt_cache_roundtrip_shapes():
    torch.manual_seed(0)
    B, L_full, E, Hq = 2, 6, 64, 8
    kvH = 2
    x = torch.randn(B, L_full, E)
    gpt = GoatAttention.for_gpt(
        embed_dim=E, num_heads=Hq, kv_num_heads=kvH, pos_rank=2, abs_rank=4
    )

    y_full, _ = gpt(x, x, x, is_causal=True)
    assert y_full.shape == (B, L_full, E)

    L0 = 4
    y0, _, kv = gpt(
        x[:, :L0],
        x[:, :L0],
        x[:, :L0],
        is_causal=True,
        use_cache=True,
        return_present_kv=True,
        position_offset_q=0,
        position_offset_k=0,
    )
    assert y0.shape == (B, L0, E)
    assert isinstance(kv, tuple) and len(kv) == 2

    x1 = x[:, L0 : L0 + 1]
    y1, _, kv2 = gpt(
        x1,
        x1,
        x1,
        is_causal=True,
        past_key_value=kv,
        use_cache=True,
        return_present_kv=True,
        position_offset_q=L0,
        position_offset_k=L0,
    )
    assert y1.shape == (B, 1, E)
    assert isinstance(kv2, tuple) and len(kv2) == 2


def test_smoke_vit_2d_shapes():
    torch.manual_seed(0)
    B, Ht, Wt, E, Hv = 2, 4, 4, 96, 8
    L = Ht * Wt + 1
    x = torch.randn(B, L, E)

    vit = GoatAttention.for_vit(embed_dim=E, num_heads=Hv, pos_rank=4)
    out, attn = vit(
        x,
        x,
        x,
        is_causal=False,
        spatial_shape=(Ht, Wt),
        has_cls_token=True,
        need_weights=True,
    )
    assert out.shape == (B, L, E)
    assert attn is not None
    assert attn.shape == (B, L, L)


