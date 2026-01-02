from __future__ import annotations

import argparse

from . import __version__


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"goat-attention {__version__}")
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("torch: <unavailable>")
        print(f"reason: {type(e).__name__}: {e}")
        return 0
    import torch as _torch

    print(f"torch {_torch.__version__}")
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    try:
        import torch
        from .attention import GoatAttention
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Smoke test requires a working PyTorch installation.\n"
            f"Import error: {type(e).__name__}: {e}\n"
        ) from e

    torch.manual_seed(0)

    # 1D test
    B, L, S, E, H = 2, 5, 7, 64, 8
    xq = torch.randn(B, L, E)
    xk = torch.randn(B, S, E)
    xv = torch.randn(B, S, E)

    m = GoatAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.1,
        batch_first=True,
        pos_rank=2,
        abs_rank=4,
        enable_key_bias=True,
        kv_num_heads=None,
    )
    out, attn = m(xq, xk, xv, is_causal=False, need_weights=True)
    print("1D out:", tuple(out.shape), "attn:", None if attn is None else tuple(attn.shape))

    # GQA + KV-cache sanity
    B, L_full, E, Hq = 2, 6, 64, 8
    kvH = 2
    x = torch.randn(B, L_full, E)
    gpt = GoatAttention.for_gpt(
        embed_dim=E, num_heads=Hq, kv_num_heads=kvH, pos_rank=2, abs_rank=4
    )

    y_full, _ = gpt(x, x, x, is_causal=True)
    print("GPT full:", tuple(y_full.shape))

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
    print("GPT streaming y0:", tuple(y0.shape), "y1:", tuple(y1.shape), "kv:", tuple(kv2[0].shape))

    # ViT-style 2D test
    B, Ht, Wt, E_vit, Hv = 2, 4, 4, 96, 8
    L_vit = Ht * Wt + 1
    x_img = torch.randn(B, L_vit, E_vit)
    vit = GoatAttention.for_vit(embed_dim=E_vit, num_heads=Hv, pos_rank=4)
    out_img, attn_img = vit(
        x_img,
        x_img,
        x_img,
        is_causal=False,
        spatial_shape=(Ht, Wt),
        has_cls_token=True,
        need_weights=True,
    )
    print("ViT out:", tuple(out_img.shape), "attn:", tuple(attn_img.shape))

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goat", description="GOAT Attention utilities")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Print version information")
    info.set_defaults(func=_cmd_info)

    smoke = sub.add_parser("smoke", help="Run a small smoke test on CPU")
    smoke.set_defaults(func=_cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


