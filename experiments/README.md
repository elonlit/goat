# GOAT Experiments

Reproducibility scripts for GOAT experiments.

## Prerequisites

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Scripts

### 1. Synthetic Benchmarks (`synthetic.py`)

Synthetic benchmarks: Passkey Retrieval accuracy, Needle-in-a-Haystack (NIAH) data, and GOAT prior.

```bash
uv run python experiments/synthetic.py --run_full
```


---

### 2. Genome Benchmark (`genome.py`)

Benchmarks GOAT vs RoPE on the human reference genome dataset.

**Run both variants:**
```bash
uv run python experiments/genome.py --run_rope --run_goat
```

**Run RoPE only:**
```bash
uv run python experiments/genome.py --run_rope
```

**Run GOAT only:**
```bash
uv run python experiments/genome.py --run_goat
```

**With custom settings:**
```bash
uv run python experiments/genome.py \
    --run_rope --run_goat \
    --dataset_config 6kbp \
    --seq_len 1024 \
    --batch_size 8 \
    --max_steps 1000 \
    --out_dir runs/my_genome_experiment
```

**Key options:**
- `--dataset_config`: `6kbp` or `12kbp`
- `--seq_len`: Training sequence length
- `--max_steps`: Maximum training steps
- `--eval_interval`: Steps between evaluations
- `--precision`: `auto`, `bf16`, `fp16`, or `fp32`
- `--compile`: Enable torch.compile

**GOAT-specific options:**
- `--goat_pos_rank`: Positional Fourier rank (default: 2)
- `--goat_abs_rank`: Absolute Fourier rank for sink term (default: 8)
- `--goat_prior_init`: Prior initialization mode (`seeded` or `uniform`)

---

### 3. ImageNet ViT (`imagenet.py`)

Trains ViT models on ImageNet-1k comparing GOAT vs baseline (APE).

**Full training + evaluation:**
```bash
uv run python experiments/imagenet.py \
    --train_path /path/to/imagenet/train \
    --val_path /path/to/imagenet/val \
    --img_path /path/to/sample_image.jpg \
    --output_dir runs/imagenet
```

**Skip training (eval only with existing checkpoints):**
```bash
uv run python experiments/imagenet.py \
    --train_path /path/to/imagenet/train \
    --val_path /path/to/imagenet/val \
    --img_path /path/to/sample_image.jpg \
    --skip_training \
    --base_ckpt runs/imagenet/base_vit_best.pth \
    --goat_ckpt runs/imagenet/goat_vit_best.pth
```

**With modern ViT recipe:**
```bash
uv run python experiments/imagenet.py \
    --train_path /path/to/imagenet/train \
    --val_path /path/to/imagenet/val \
    --img_path /path/to/sample_image.jpg \
    --modern_vit \
    --amp --tf32
```

**Multi-GPU with torchrun:**
```bash
uv run torchrun --standalone --nproc_per_node=8 experiments/imagenet.py \
    --train_path /path/to/imagenet/train \
    --val_path /path/to/imagenet/val \
    --img_path /path/to/sample_image.jpg
```

**Key options:**
- `--epochs`: Training epochs (default: 300)
- `--batch_size`: Global batch size (default: 64)
- `--resolutions`: Test resolutions for extrapolation (default: 224 384 512)
- `--modern_vit`: Use modern ViT recipe (SwiGLU, LayerScale, etc.)
- `--amp`: Enable automatic mixed precision

**GOAT-specific options:**
- `--goat_pos_rank`: Positional Fourier rank (default: 16)
- `--goat_abs_rank`: Absolute Fourier rank for sink term (default: 2)
- `--goat_pos_encoding`: Positional encoding type (`1d` or `2d`, default: `2d`)

---

### 4. C4 Language Modeling (`train_c4.py`)

Trains 125M GPT-style causal LMs on C4 comparing RoPE, ALiBi, and GOAT.

**Train all variants:**
```bash
uv run python experiments/train_c4.py \
    --output_root runs/c4 \
    --run_name c4_125m \
    --variants rope alibi goat
```

**Train single variant:**
```bash
uv run python experiments/train_c4.py \
    --output_root runs/c4 \
    --variants goat
```

**Multi-GPU with torchrun:**
```bash
uv run torchrun --standalone --nproc_per_node=8 experiments/train_c4.py \
    --output_root runs/c4 \
    --variants rope alibi goat
```

**Custom training settings:**
```bash
uv run python experiments/train_c4.py \
    --output_root runs/c4 \
    --seq_len_train 2048 \
    --total_tokens 4000000000 \
    --micro_batch_size 4 \
    --grad_accum 4 \
    --precision bf16 \
    --gradient_checkpointing
```

**Key options:**
- `--variants`: Which variants to train (`rope`, `alibi`, `goat`)
- `--seq_len_train`: Training sequence length (default: 2048)
- `--total_tokens`: Total tokens to train on (default: 4B)
- `--extrap_lengths`: Context lengths for extrapolation eval
- `--precision`: `bf16`, `fp16`, or `fp32`
- `--gradient_checkpointing`: Reduce memory usage

**GOAT-specific options:**
- `--goat_pos_rank`: Positional Fourier rank (default: 4)
- `--goat_abs_rank`: Absolute Fourier rank for sink term (default: 4)
- `--goat_pos_base`: Positional base frequency (default: 10000.0)
- `--goat_abs_base`: Absolute base frequency (default: 10000.0)
- `--goat_disable_key_bias`: Disable the learned key bias (sink term)

