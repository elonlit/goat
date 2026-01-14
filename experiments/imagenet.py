#!/usr/bin/env python
"""Train ViT on ImageNet comparing GOAT vs baseline (APE)."""

from __future__ import annotations

import argparse
import json
import os
import random

from typing import Optional

import numpy as np
from PIL import Image
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.datasets as datasets
from tqdm import tqdm

from goat import GoatAttention

try:
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
except Exception:  # pragma: no cover
    Mixup = None
    SoftTargetCrossEntropy = None
    LabelSmoothingCrossEntropy = None


def ddp_init():
    if not (dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device, False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    elif torch.cuda.is_available():
        raise RuntimeError("LOCAL_RANK not set - launch with torchrun")
    else:
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device, True


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        n = len(self.dataset)
        return (n + self.num_replicas - 1 - self.rank) // self.num_replicas


def _sync_scaler_found_inf_(scaler: torch.cuda.amp.GradScaler, optimizer: torch.optim.Optimizer):
    if not (dist.is_available() and dist.is_initialized()):
        return
    per_opt = getattr(scaler, "_per_optimizer_states", None)
    if isinstance(per_opt, dict):
        st = per_opt.get(id(optimizer), None)
        if isinstance(st, dict):
            found = st.get("found_inf_per_device", None)
            if isinstance(found, dict):
                for t in found.values():
                    if torch.is_tensor(t):
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)
                return
            if torch.is_tensor(found):
                dist.all_reduce(found, op=dist.ReduceOp.MAX)
                return

    found2 = getattr(scaler, "_found_inf_per_device", None)
    if isinstance(found2, dict):
        for t in found2.values():
            if torch.is_tensor(t):
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return


def _global_mixup_ddp(
    mixup_fn,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (dist.is_available() and dist.is_initialized()) or world_size <= 1:
        return mixup_fn(images, labels)

    # Gather (same shapes across ranks).
    imgs_g = [torch.empty_like(images) for _ in range(world_size)]
    lbls_g = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(imgs_g, images)
    dist.all_gather(lbls_g, labels)

    if rank == 0:
        all_images = torch.cat(imgs_g, dim=0)
        all_labels = torch.cat(lbls_g, dim=0)
        all_images, all_labels = mixup_fn(all_images, all_labels)
        chunks_img = list(all_images.chunk(world_size, dim=0))
        chunks_lbl = list(all_labels.chunk(world_size, dim=0))
        lbl_dim = chunks_lbl[0].shape[1] if chunks_lbl[0].dim() == 2 else 1
    else:
        chunks_img = None
        chunks_lbl = None
        lbl_dim = 0

    lbl_dim_t = torch.tensor([lbl_dim], device=images.device, dtype=torch.long)
    dist.broadcast(lbl_dim_t, src=0)
    lbl_dim = lbl_dim_t.item()

    recv_images = torch.empty_like(images)
    recv_labels = torch.empty((images.shape[0], lbl_dim), device=images.device, dtype=torch.float32)

    dist.scatter(recv_images, scatter_list=chunks_img, src=0)
    dist.scatter(recv_labels, scatter_list=chunks_lbl, src=0)
    return recv_images, recv_labels


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden = max(1, int(hidden_features * 2 / 3))
        self.fc1 = nn.Linear(in_features, hidden * 2)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, need_weights: bool = False, cls_attn_only: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Dh)

        # SDPA expects (..., L, Dh)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        attn = None
        if need_weights:
            if cls_attn_only:
                logits = (q[:, :, 0:1, :] * self.scale) @ k.transpose(-2, -1)
                attn = logits.softmax(dim=-1)
            else:
                logits = (q * self.scale) @ k.transpose(-2, -1)
                attn = logits.softmax(dim=-1)
        return out, attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
        mlp_layer: str = "gelu",
        use_goat: bool = False,
        goat_kwargs=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        if use_goat:
            self.attn = GoatAttention.for_vit(
                embed_dim=dim, num_heads=num_heads, **(goat_kwargs or {})
            )
        else:
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=drop,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        mlp_layer = str(mlp_layer).lower()
        if mlp_layer == "gelu":
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, out_features=dim, drop=drop)
        elif mlp_layer == "swiglu":
            self.mlp = SwiGLUMlp(in_features=dim, hidden_features=mlp_hidden, out_features=dim, drop=drop)
        else:
            raise ValueError(f"Unknown mlp_layer={mlp_layer}. Use 'gelu' or 'swiglu'.")
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values > 0.0 else nn.Identity()
        self.use_goat = use_goat

    def forward(self, x, spatial_shape=None, need_weights=False, cls_attn_only: bool = False):
        shortcut = x
        x = self.norm1(x)

        if self.use_goat:
            attn_out, weights = self.attn(
                x, x, x,
                spatial_shape=spatial_shape,
                need_weights=need_weights,
                cls_attn_only=cls_attn_only,
            )
        else:
            attn_out, weights = self.attn(x, need_weights=need_weights, cls_attn_only=cls_attn_only)

        x = shortcut + self.drop_path(self.ls1(attn_out))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x, weights


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 0.0,
        patch_norm: bool = False,
        mlp_layer: str = "gelu",
        use_goat=False,
        goat_kwargs=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        norm_layer = (nn.LayerNorm if patch_norm else None)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if not use_goat:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
            _trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    init_values=init_values,
                    mlp_layer=mlp_layer,
                    use_goat=use_goat,
                    goat_kwargs=goat_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        if use_goat:
            for blk in self.blocks:
                attn = getattr(blk, "attn", None)
                if isinstance(attn, GoatAttention):
                    attn._reset_parameters()
                    if getattr(attn, "key_bias", None) is not None:
                        scale = float(getattr(attn, "init_scale_prior", 1e-3))
                        with torch.no_grad():
                            nn.init.normal_(attn.key_bias.fc1.weight, mean=0.0, std=scale)
                            nn.init.zeros_(attn.key_bias.fc1.bias)
                            nn.init.normal_(attn.key_bias.fc2.weight, mean=0.0, std=scale)
                            nn.init.zeros_(attn.key_bias.fc2.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, H, W):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and H == W:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = h0 = int(N ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(H, W),
            mode='bicubic',
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, return_last_attn: bool = False, attn_full: bool = False):
        B, C, H_img, W_img = x.shape
        H_grid = H_img // self.patch_embed.patch_size
        W_grid = W_img // self.patch_embed.patch_size

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.interpolate_pos_encoding(x, H_grid, W_grid)
        x = self.pos_drop(x)

        last_weights = None
        for i, blk in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            capture = return_last_attn and is_last
            x, w = blk(
                x,
                spatial_shape=(H_grid, W_grid),
                need_weights=capture,
                cls_attn_only=(capture and (not attn_full)),
            )
            if capture:
                last_weights = w

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        if return_last_attn:
            return x, last_weights
        return x


def build_vit(use_goat: bool, goat_kwargs=None) -> VisionTransformer:
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_values=0.0,
        patch_norm=False,
        mlp_layer="gelu",
        use_goat=use_goat,
        goat_kwargs=goat_kwargs,
    )
    return model


def prepare_model(
    model: nn.Module,
    device: torch.device,
    model_name: str = "Model",
    *,
    ddp: bool,
    local_rank: int,
) -> nn.Module:
    model = model.to(device)
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    return model


def count_trainable_params(model: nn.Module) -> int:
    model = unwrap_model(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    model = unwrap_model(model)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    new_state = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(new_state, strict=True)
    return model


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, max_retries: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self._warned = set()

    def __getitem__(self, index: int):
        last_err = None
        idx = index
        for _ in range(self.max_retries):
            try:
                return super().__getitem__(idx)
            except (OSError, ValueError, RuntimeError) as e:
                last_err = e
                path, _ = self.samples[idx]
                if path not in self._warned:
                    print(f"[SafeImageFolder] Skipping unreadable image: {path} ({type(e).__name__}: {e})")
                    self._warned.add(path)
                idx = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"SafeImageFolder: failed to fetch a valid sample after {self.max_retries} retries") from last_err


def get_imagenet_loaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    drop_last: bool = False,
    sampler_seed: int = 0,
):
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SafeImageFolder(train_path, transform=train_transform)
    val_dataset = SafeImageFolder(val_path, transform=val_transform)

    pin = (device.type == "cuda")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=sampler_seed,
        drop_last=False,
    )
    val_sampler = DistributedEvalSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=bool(drop_last),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler


@torch.no_grad()
def evaluate_on_loader(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       *,
                       use_amp: bool = False,
                       desc: str = "Val",
                       rank: int = 0) -> (float, float):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)

    for images, labels in tqdm(loader, desc=desc, leave=False, disable=(rank != 0)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(use_amp) and device.type == "cuda"):
            outputs = model(images)
        loss = criterion(outputs, labels)  # summed over batch

        loss_sum += loss.detach().to(torch.float64)
        total += labels.numel()
        correct += (outputs.argmax(dim=1) == labels).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum / total.to(torch.float64)).item()
    top1 = (100.0 * correct.to(torch.float64) / total.to(torch.float64)).item()
    return top1, avg_loss


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                train_sampler: DistributedSampler,
                device: torch.device,
                epochs: int,
                lr: float,
                weight_decay: float,
                warmup_epochs: int,
                model_name: str,
                save_path: str,
                *,
                mixup_fn=None,
                label_smoothing: float = 0.1,
                use_amp: bool = False,
                rank: int = 0,
                world_size: int = 1):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_l = n.lower()
        if (
            p.ndim == 1
            or n.endswith(".bias")
            or ("norm" in n_l)
            or ("pos_embed" in n_l)
            or ("cls_token" in n_l)
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=(0.9, 0.999),
    )

    warmup_epochs = int(warmup_epochs)
    warmup_epochs = max(0, warmup_epochs)
    cosine_epochs = max(1, epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)

    if mixup_fn is not None:
        train_criterion = SoftTargetCrossEntropy()
    else:
        if float(label_smoothing) > 0.0:
            train_criterion = LabelSmoothingCrossEntropy(smoothing=float(label_smoothing))
        else:
            train_criterion = nn.CrossEntropyLoss()

    use_amp = bool(use_amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_correct = torch.zeros((), device=device, dtype=torch.long)
        epoch_total = torch.zeros((), device=device, dtype=torch.long)

        base_lr = float(lr)
        if epoch < warmup_epochs:
            warmup_lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        pbar = tqdm(train_loader, desc=f"[{model_name}] Train epoch {epoch+1}/{epochs}", disable=(rank != 0))
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            hard_labels = labels
            if mixup_fn is not None:
                images, labels = _global_mixup_ddp(
                    mixup_fn,
                    images,
                    labels,
                    rank=rank,
                    world_size=world_size,
                )
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = train_criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _sync_scaler_found_inf_(scaler, optimizer)
            scaler.step(optimizer)
            scaler.update()

            bs = hard_labels.size(0)
            epoch_loss_sum += loss.detach().to(torch.float64) * bs
            epoch_total += bs
            epoch_correct += (outputs.argmax(dim=1) == hard_labels).sum()

            if rank == 0:
                pbar.set_postfix({
                    "loss": f"{(epoch_loss_sum / epoch_total.to(torch.float64)).item():.4f}",
                    "acc": f"{(100.0 * epoch_correct.to(torch.float64) / epoch_total.to(torch.float64)).item():.2f}%"
                })

        if epoch >= warmup_epochs:
            scheduler.step()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_correct, op=dist.ReduceOp.SUM)

        train_loss = (epoch_loss_sum / epoch_total.to(torch.float64)).item()
        train_acc = (100.0 * epoch_correct.to(torch.float64) / epoch_total.to(torch.float64)).item()

        val_acc, val_loss = evaluate_on_loader(
            model, val_loader, device,
            use_amp=use_amp,
            desc=f"[{model_name}] Val epoch {epoch+1}/{epochs}",
            rank=rank,
        )

        if rank == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if (val_acc > best_acc) and is_main_process(rank):
            best_acc = val_acc
            torch.save(unwrap_model(model).state_dict(), save_path)
            print(f"[{model_name}] New best val_acc={best_acc:.2f}%, saved to {save_path}")

    if rank == 0:
        print(f"[{model_name}] Training done. Best val_acc={best_acc:.2f}% (ckpt: {save_path})")


def evaluate_at_resolution(model: nn.Module,
                           val_path: str,
                           resolution: int,
                           *,
                           batch_size: int,
                           device: torch.device,
                           use_amp: bool = False,
                           num_workers: int = 4,
                           rank: int = 0,
                           world_size: int = 1) -> float:
    resize_side = round(resolution * 256.0 / 224.0)
    transform = T.Compose([
        T.Resize(resize_side),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    bs = batch_size
    if bs <= 0:
        raise ValueError("batch_size must be >= 1.")

    ddp = dist.is_available() and dist.is_initialized() and world_size > 1

    while True:
        dataset = SafeImageFolder(val_path, transform=transform)
        sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model.eval()
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = torch.zeros((), device=device, dtype=torch.long)
        oom_any = torch.zeros((), device=device, dtype=torch.int32)
        err: Optional[RuntimeError] = None
        try:
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=f"Eval @ {resolution}px", disable=(rank != 0)):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=bool(use_amp) and device.type == "cuda"):
                        outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    total += labels.numel()
                    correct += (preds == labels).sum()
        except RuntimeError as e:
            msg = str(e).lower()
            if device.type == "cuda" and ("out of memory" in msg or "cuda error" in msg):
                oom_any.fill_(1)
                err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

        # DDP-safe OOM backoff: all ranks decide together whether to retry.
        if ddp:
            dist.all_reduce(oom_any, op=dist.ReduceOp.MAX)
        if oom_any.item() != 0:
            if bs == 1:
                # If any rank OOMs at batch=1, we can't recover via batch backoff.
                if err is not None:
                    raise err
                raise RuntimeError(f"OOM during eval @ {resolution}px even with batch_size=1")
            prev = bs
            bs = max(1, bs // 2)
            if rank == 0:
                print(f"[evaluate_at_resolution] OOM at {resolution}px with batch_size={prev}; retrying at bs={bs}")
            continue

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        acc = (100.0 * correct.to(torch.float64) / total.to(torch.float64)).item()
        return acc


def get_center_prior(model: nn.Module,
                     device: torch.device,
                     H_grid: int = 32,
                     W_grid: int = 32) -> np.ndarray:
    L = H_grid * W_grid + 1

    vit = unwrap_model(model)

    attn_module = vit.blocks[0].attn
    spatial = (H_grid, W_grid)
    if hasattr(attn_module, "pos_encoding"):
        pe = str(getattr(attn_module, "pos_encoding")).lower()
        if pe == "1d":
            spatial = None

    log_prior = attn_module.compute_log_prior(
        L,
        device=device,
        spatial_shape=spatial,
        has_cls_token=True,
    )  # (L, L)

    center_idx = 1 + (H_grid // 2) * W_grid + (W_grid // 2)
    prior_map = log_prior[center_idx, 1:].reshape(H_grid, W_grid)
    return prior_map.detach().cpu().numpy()


def get_attn_map(model: nn.Module,
                 image_tensor: torch.Tensor,
                 device: torch.device,
                 use_amp: bool = False) -> np.ndarray:
    vit = unwrap_model(model)

    vit.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=bool(use_amp) and device.type == "cuda"):
            logits, weights = vit(image_tensor.to(device, non_blocking=True), return_last_attn=True)

        if weights is None:
            raise RuntimeError("Model did not return attention weights. "
                               "Make sure return_last_attn=True is respected.")

        if weights.dim() == 4:
            weights = weights.mean(dim=1)
        if weights.dim() == 3:
            cls_row = weights[0, 0]
        elif weights.dim() == 2:
            cls_row = weights[0]
        else:
            raise RuntimeError(f"Unexpected attention weight shape: {tuple(weights.shape)}")

        cls_attn = cls_row[1:]
        _, _, H_img, W_img = image_tensor.shape
        patch = vit.patch_embed.patch_size
        H_grid = H_img // patch
        W_grid = W_img // patch
        cls_attn_grid = cls_attn.reshape(H_grid, W_grid)

    return cls_attn_grid.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Single-script ImageNet-1k pipeline for GOAT vs baseline ViT: "
                    "train both, then produce data for 3-panel figure (A/B/C)."
    )
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to ImageNet training folder (ImageFolder layout).")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to ImageNet validation folder (ImageFolder layout).")

    # Checkpoint paths (can be left None; will default inside output_dir)
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="Path to baseline ViT checkpoint (for saving/loading).")
    parser.add_argument("--goat_ckpt", type=str, default=None,
                        help="Path to GOAT ViT checkpoint (for saving/loading).")

    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to a demo image (e.g., bird/dog) for Panels B & C.")
    parser.add_argument("--output_dir", type=str, default="./goat_imagenet_runs",
                        help="Directory to save checkpoints and panel data.")

    parser.add_argument("--out_json", type=str, default="panel_a_extrapolation.json",
                        help="Output JSON file name for Panel A results (saved inside output_dir).")
    parser.add_argument("--out_npz", type=str, default="panel_b_c_viz.npz",
                        help="Output NPZ file name for Panel B/C arrays (saved inside output_dir).")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Base batch size for training and for resolutions < 800.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers.")
    # Modern supervised ViT recipes typically use 300-400 epochs.
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per model.")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW.")
    # Mixup/CutMix + label smoothing (timm-style)
    parser.add_argument("--no_mixup", action="store_true",
                        help="Disable Mixup/CutMix. (By default we enable the modern recipe.)")
    parser.add_argument("--mixup", type=float, default=0.8,
                        help="Mixup alpha.")
    parser.add_argument("--cutmix", type=float, default=1.0,
                        help="CutMix alpha.")
    parser.add_argument("--mixup_prob", type=float, default=1.0,
                        help="Probability of applying mixup or cutmix.")
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5,
                        help="Probability of switching to CutMix when both are enabled.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing (used inside Mixup labels, or as CE smoothing when mixup is disabled).")

    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[224, 384, 512],
                        help="List of test-time resolutions for Panel A.")
    parser.add_argument("--goat_pos_rank", type=int, default=16,
                        help="pos_rank passed into GoatAttention.for_vit for GOAT.")
    parser.add_argument("--goat_abs_rank", type=int, default=2,
                        help="abs_rank passed into GoatAttention.for_vit for GOAT.")
    parser.add_argument("--goat_pos_encoding", type=str, default="2d",
                        choices=["1d", "2d"],
                        help="pos_encoding passed into GoatAttention.for_vit for GOAT.")
    # Modern ViT knobs (affect BOTH baseline and GOAT wrappers via the ViT blocks)
    parser.add_argument("--modern_vit", action="store_true",
                        help="Opt-in preset for a more SOTA-ish ViT (SwiGLU, patch_norm, LayerScale, etc.). "
                             "Use this when training new checkpoints; old checkpoints may not be compatible.")
    parser.add_argument("--drop_path", type=float, default=0.1,
                        help="Stochastic depth rate (timm-style).")
    parser.add_argument("--drop", type=float, default=0.0,
                        help="Dropout rate for tokens/MLP/proj (baseline) and MLP/proj + attention dropout (GOAT wrapper).")
    parser.add_argument("--layer_scale", type=float, default=0.0,
                        help="LayerScale init value (0 disables). Typical values: 1e-5.")
    pn = parser.add_mutually_exclusive_group()
    pn.add_argument("--patch_norm", dest="patch_norm", action="store_true",
                    help="Apply LayerNorm after patch embedding.")
    pn.add_argument("--no_patch_norm", dest="patch_norm", action="store_false",
                    help="Disable patch embedding LayerNorm.")
    pn.set_defaults(patch_norm=None)

    parser.add_argument("--mlp_layer", type=str, default=None, choices=["gelu", "swiglu"],
                        help="MLP type inside transformer blocks.")
    parser.add_argument("--highres_panel_c", type=int, default=896,
                        help="Resolution (pixels) used for high-res attention maps in Panel C.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--skip_training", action="store_true",
                        help="If set, skip training and only run eval/viz using existing ckpts.")
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp mixed precision (CUDA only).")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 for matmul/conv on Ampere+ (often faster; usually safe for ViTs).")

    args = parser.parse_args()

    if (not args.no_mixup) or (float(args.label_smoothing) > 0.0):
        if Mixup is None or SoftTargetCrossEntropy is None or LabelSmoothingCrossEntropy is None:
            raise RuntimeError("timm is required for mixup/label smoothing")

    if args.base_ckpt is None:
        args.base_ckpt = os.path.join(args.output_dir, "baseline_vit_best.pth")
    if args.goat_ckpt is None:
        args.goat_ckpt = os.path.join(args.output_dir, "goat_vit_best.pth")

    rank, world_size, local_rank, device, ddp_enabled = ddp_init()
    main_proc = is_main_process(rank)

    if main_proc:
        os.makedirs(args.output_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    seed_all(args.seed)

    if args.modern_vit:
        if args.mlp_layer is None:
            args.mlp_layer = "swiglu"
        if args.patch_norm is None:
            args.patch_norm = True
        if args.layer_scale == 0.0:
            args.layer_scale = 1e-5
    else:
        if args.mlp_layer is None:
            args.mlp_layer = "gelu"
        if args.patch_norm is None:
            args.patch_norm = False

    if main_proc:
        print(f"DDP enabled: {ddp_enabled}  rank={rank}  world_size={world_size}  local_rank={local_rank}")
        print(f"Using device: {device}")
        print(f"Baseline ckpt: {args.base_ckpt}")
        print(f"GOAT ckpt:     {args.goat_ckpt}")
        print(f"ViT config: mlp_layer={args.mlp_layer} pos_embed=APE(learned) "
              f"pooling=CLS-token patch_norm={args.patch_norm} layer_scale={args.layer_scale}")

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_amp = bool(args.amp) and (device.type == "cuda")

    global_bs = int(args.batch_size)
    if global_bs % world_size != 0:
        raise ValueError(f"--batch_size (global) must be divisible by world_size={world_size}. Got {global_bs}.")
    local_bs = global_bs // world_size

    mixup_enabled = (
        (not args.no_mixup)
        and (float(args.mixup_prob) > 0.0)
        and (float(args.mixup) > 0.0 or float(args.cutmix) > 0.0)
    )
    if mixup_enabled and (local_bs % 2 != 0):
        raise ValueError("mixup/cutmix requires even local batch size")

    if not args.skip_training:
        if main_proc:
            print("\n=== Building ImageNet train/val loaders for training (224x224) ===")
        train_loader, val_loader, train_sampler = get_imagenet_loaders(
            args.train_path,
            args.val_path,
            local_bs,
            args.num_workers,
            device,
            rank=rank,
            world_size=world_size,
            drop_last=mixup_enabled,
            sampler_seed=args.seed,
        )

        if main_proc:
            print("=== Training baseline ViT ===")
        seed_all(args.seed)
        model_base_train = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            init_values=args.layer_scale,
            patch_norm=args.patch_norm,
            mlp_layer=args.mlp_layer,
            use_goat=False,
        )
        model_base_train = prepare_model(
            model_base_train,
            device,
            model_name="Baseline (train)",
            ddp=ddp_enabled,
            local_rank=local_rank,
        )
        n_params_base = count_trainable_params(model_base_train)
        if main_proc:
            print(f"[Baseline] Trainable parameters: {n_params_base:,} "
                  f"({n_params_base/1e6:.2f}M)")
        train_model(
            model_base_train,
            train_loader,
            val_loader,
            train_sampler,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            model_name="Baseline",
            save_path=args.base_ckpt,
            mixup_fn=(
                None
                if args.no_mixup
                else (
                    Mixup(
                        mixup_alpha=float(args.mixup),
                        cutmix_alpha=float(args.cutmix),
                        cutmix_minmax=None,
                        prob=float(args.mixup_prob),
                        switch_prob=float(args.mixup_switch_prob),
                        mode="batch",
                        label_smoothing=float(args.label_smoothing),
                        num_classes=1000,
                    )
                    if Mixup is not None
                    else None
                )
            ),
            label_smoothing=float(args.label_smoothing),
            use_amp=use_amp,
            rank=rank,
            world_size=world_size,
        )

        del model_base_train
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if main_proc:
            print("=== Training GOAT ViT ===")
        seed_all(args.seed)
        goat_kwargs = {
            "pos_rank": args.goat_pos_rank,
            "abs_rank": args.goat_abs_rank,
            "pos_encoding": args.goat_pos_encoding,
            "dropout": args.drop,  # GOAT attention dropout
        }
        model_goat_train = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            init_values=args.layer_scale,
            patch_norm=args.patch_norm,
            mlp_layer=args.mlp_layer,
            use_goat=True,
            goat_kwargs=goat_kwargs,
        )
        model_goat_train = prepare_model(
            model_goat_train,
            device,
            model_name="GOAT (train)",
            ddp=ddp_enabled,
            local_rank=local_rank,
        )
        n_params_goat = count_trainable_params(model_goat_train)
        if main_proc:
            print(f"[GOAT] Trainable parameters: {n_params_goat:,} "
                  f"({n_params_goat/1e6:.2f}M)")
        train_model(
            model_goat_train,
            train_loader,
            val_loader,
            train_sampler,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            model_name="GOAT",
            save_path=args.goat_ckpt,
            mixup_fn=(
                None
                if args.no_mixup
                else (
                    Mixup(
                        mixup_alpha=float(args.mixup),
                        cutmix_alpha=float(args.cutmix),
                        cutmix_minmax=None,
                        prob=float(args.mixup_prob),
                        switch_prob=float(args.mixup_switch_prob),
                        mode="batch",
                        label_smoothing=float(args.label_smoothing),
                        num_classes=1000,
                    )
                    if Mixup is not None
                    else None
                )
            ),
            label_smoothing=float(args.label_smoothing),
            use_amp=use_amp,
            rank=rank,
            world_size=world_size,
        )

        del model_goat_train
        del train_loader
        del val_loader
        del train_sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        if not os.path.isfile(args.base_ckpt):
            raise FileNotFoundError(f"--skip_training was set but baseline ckpt not found: {args.base_ckpt}")
        if not os.path.isfile(args.goat_ckpt):
            raise FileNotFoundError(f"GOAT ckpt not found: {args.goat_ckpt}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if main_proc:
        print("\n=== Panel A: Evaluating Baseline (APE + interpolation) ===")
        results = {"baseline": {}, "goat": {}}
    else:
        results = None

    model_base = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,          # keep eval deterministic
        drop_path_rate=0.0,
        init_values=args.layer_scale,
        patch_norm=args.patch_norm,
        mlp_layer=args.mlp_layer,
        use_goat=False,
    )
    model_base = load_checkpoint(model_base, args.base_ckpt, device)
    model_base = prepare_model(model_base, device, model_name="Baseline", ddp=False, local_rank=local_rank)

    for res in sorted(args.resolutions):
        bs = local_bs if res < 800 else min(local_bs, 8)
        if main_proc:
            print(f"\n[Baseline] Resolution {res} (local batch size {bs})")
        acc = evaluate_at_resolution(
            model_base, args.val_path, res,
            batch_size=bs,
            device=device,
            use_amp=use_amp,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )
        if main_proc:
            results["baseline"][res] = acc
            print(f"Baseline @ {res}: {acc:.2f}%")

    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if main_proc:
        print("\n=== Panel A: Evaluating GOAT ===")
    goat_kwargs = {
        "pos_rank": args.goat_pos_rank,
        "abs_rank": args.goat_abs_rank,
        "pos_encoding": args.goat_pos_encoding,
        "dropout": 0.0,  # eval
    }
    model_goat = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,          # keep eval deterministic
        drop_path_rate=0.0,
        init_values=args.layer_scale,
        patch_norm=args.patch_norm,
        mlp_layer=args.mlp_layer,
        use_goat=True,
        goat_kwargs=goat_kwargs,
    )
    model_goat = load_checkpoint(model_goat, args.goat_ckpt, device)
    model_goat = prepare_model(model_goat, device, model_name="GOAT", ddp=False, local_rank=local_rank)

    for res in sorted(args.resolutions):
        bs = local_bs if res < 800 else min(local_bs, 8)
        if main_proc:
            print(f"\n[GOAT] Resolution {res} (local batch size {bs})")
        acc = evaluate_at_resolution(
            model_goat, args.val_path, res,
            batch_size=bs,
            device=device,
            use_amp=use_amp,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )
        if main_proc:
            results["goat"][res] = acc
            print(f"GOAT @ {res}: {acc:.2f}%")

    if main_proc:
        json_path = os.path.join(args.output_dir, args.out_json)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved Panel A data to {json_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not main_proc:
        try:
            del model_goat
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        ddp_cleanup()
        return

    print("=== Panel B & C: Prior and attention data ===")

    def _snap_res_to_patch(res: int, patch: int, min_res: int = 224) -> int:
        res, patch = int(res), int(patch)
        if patch <= 0:
            return max(min_res, res)
        res = max(min_res, res)
        return (res // patch) * patch

    print("Extracting GOAT prior...")
    prior_map = get_center_prior(model_goat, device)

    def _try_panel_c_at_resolution(res_c: int):
        transform = T.Compose([
            T.Resize((res_c, res_c)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        raw_img = Image.open(args.img_path).convert("RGB")
        img_tensor = transform(raw_img).unsqueeze(0)  # (1, C, H, W)

        unwrap_model(model_goat).to(device)
        goat_attn = get_attn_map(model_goat, img_tensor, device, use_amp=use_amp)
        unwrap_model(model_goat).to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_base_c = None
        try:
            model_base_c = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,          # deterministic eval
                drop_path_rate=0.0,
                init_values=args.layer_scale,
                patch_norm=args.patch_norm,
                mlp_layer=args.mlp_layer,
                use_goat=False,
            )
            model_base_c = load_checkpoint(model_base_c, args.base_ckpt, device)
            model_base_c = prepare_model(model_base_c, device, model_name="Baseline", ddp=False, local_rank=local_rank)
            base_attn = get_attn_map(model_base_c, img_tensor, device, use_amp=use_amp)
        finally:
            if model_base_c is not None:
                del model_base_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return raw_img, goat_attn, base_attn

    _vit_goat = unwrap_model(model_goat)
    patch = getattr(_vit_goat.patch_embed, "patch_size", 16)
    res_c = _snap_res_to_patch(args.highres_panel_c, patch=patch, min_res=224)
    tried = set()
    data_store = {}
    while True:
        if res_c in tried:
            raise RuntimeError("Panel C fallback loop got stuck; please report this.")
        tried.add(res_c)
        try:
            raw_img, goat_attn, base_attn = _try_panel_c_at_resolution(res_c)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if device.type == "cuda" and ("out of memory" in msg or "cuda error" in msg):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_res = _snap_res_to_patch(round(res_c * 0.85), patch=patch, min_res=224)
                if next_res >= res_c:
                    next_res = _snap_res_to_patch(res_c - patch, patch=patch, min_res=224)
                if next_res < 224 or next_res <= 0:
                    raise RuntimeError(
                        f"Panel C OOM even after backoff down to {res_c}px. "
                        "Try a smaller --highres_panel_c or a larger GPU."
                    ) from e
                print(f"[Panel C] OOM at {res_c}px; retrying at {next_res}px")
                res_c = next_res
                continue
            raise

    data_store["panel_c_resolution"] = res_c
    data_store["original_image"] = np.array(raw_img.resize((res_c, res_c)))

    data_store["goat_prior_map"] = prior_map
    print("Extracting GOAT attention map...")
    data_store["goat_attn_map"] = goat_attn
    print("Extracting Baseline attention map...")
    data_store["base_attn_map"] = base_attn

    npz_path = os.path.join(args.output_dir, args.out_npz)
    np.savez(npz_path, **data_store)
    print(f"Saved to {npz_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    ddp_cleanup()
    return


if __name__ == "__main__":
    main()
