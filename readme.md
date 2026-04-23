# DNA MLM Scaling Laws

A reproducible pipeline for measuring scaling laws on a Masked-Language-Model trained on protein-coding DNA (CDS) sequences. Sweeps model size `N`, training tokens `D`, and compute `C = 6·N·D`; fits power laws; produces Chinchilla-style IsoFLOP / TTP plots.

---

## 1. What's in the box

| Component | Path | Role |
|---|---|---|
| Model | `src/model.py` | Encoder-only transformer (Pre-LN, RoPE, GELU FFN, MLM head) |
| Dataset | `src/dataset.py` | Tokeniser, MLM masking, FASTA loader, synthetic generator |
| Loss | `src/loss.py` | Cross-entropy over masked positions only |
| Trainer | `src/train.py` | AdamW + cosine LR, grad accumulation + clipping, JSON logs |
| Data download | `data_downloaders/download_ncbi_ftp.py` | Bulk CDS download from NCBI FTP |
| Setup | `scripts/setup_scaling_experiment.py` | Generates per-run configs |
| Orchestrator | `scripts/run_complete_scaling_analysis.sh` | Runs the full sweep |
| Smoke test | `scripts/test_scaling_orchestration.sh` | Tiny CPU-only end-to-end check |
| Plots | `scripts/plot_scaling.py`, `plot_iso_analysis.py`, `plot_ttp.py` | Analysis outputs |

## 2. Model architecture

- **Type**: encoder-only transformer, Pre-LN, RoPE positional encoding.
- **Per-layer**: multi-head self-attention → feed-forward (GELU) → residuals.
- **Head**: final LayerNorm → linear MLM head over a 7-token vocab (`[PAD]`, `[MASK]`, `A`, `C`, `G`, `T`, `N`).
- **Objective**: 15% of non-pad positions replaced with `[MASK]`; cross-entropy only over masked positions.
- **FLOPs accounting**: `C ≈ 6 · N · D` where `D` is real (non-pad) tokens seen.

### Parameter-count estimate

For `(d_model=d, n_heads=h, n_layers=L, d_ff=ff, vocab=V)`:

```
embeddings    = 2 · V · d
per_layer     = 4·d² + 2·d·ff + 4·d       # attention + FFN + 2 LNs
total         ≈ 2·V·d + L · per_layer
```

`scripts/setup_scaling_experiment.py` builds arch configs by solving this inversely for a target param count.

---

## 3. Installation

```bash
./setup.sh
```

`setup.sh` does three things in order:

1. Installs Python dependencies (`pip install -r requirements.txt`).
2. Makes all shell and Python scripts executable.
3. Downloads CDS FASTA data for all 33 species into `ncbi_ftp_output/` (~4.5 GB, 30–60 min depending on connection). If the folder is already populated it skips the download automatically.

To download only a subset of species instead, Ctrl-C during the download step and run manually:

```bash
python data_downloaders/download_ncbi_ftp.py --species human mouse rat
```

Dependencies: `torch`, `numpy`, `matplotlib`, `tqdm`. Python 3.10+.

---

## 4. Quick test (synthetic, CPU, ~5 min)

Sanity-checks every phase of the pipeline with 4 tiny models on random DNA:

```bash
./scripts/test_scaling_orchestration.sh
```

Produces `test_experiment/plots/*.png` and `test_experiment/logs/*.json`.

Override the knobs with env vars:

```bash
NUM_SYNTHETIC=500 MAX_SEQ_LEN=64 ./scripts/test_scaling_orchestration.sh
```

---

## 5. Full run

### Step 1 — install and download data

```bash
./setup.sh
```

This handles both dependency installation and data download. After it completes, `ncbi_ftp_output/` will contain 33 species FASTA files (~4.5 GB total).

### Step 2 — run the sweep

```bash
./scripts/run_complete_scaling_analysis.sh
```

The script does, in order:

1. `setup` — generates per-run JSON configs.
2. `split` — carves a fixed held-out eval FASTA (default 50M tokens).
3. `samples` — pre-builds token-budgeted FASTAs for Iso-Param.
4. `iso-token` — N sweep at full-data D.
5. `iso-param` — D sweep at fixed N.
6. `iso-flop` — per-budget N sweep at fixed C, with per-run FASTA samples.
7. `plots` — scaling exponents + IsoFLOP / TTP figures.

Skip any phase with `--skip-<name>` (e.g. `--skip-setup --skip-split` to resume). Completed runs are auto-skipped via `training_log_<name>.json` lookup, so a killed run can be resumed just by re-invoking the script. Cleanup intermediates with `--cleanup`.

---

## 6. Configuration reference

### 6a. Orchestrator env vars (`run_complete_scaling_analysis.sh`)

| Var | Default | Meaning |
|---|---|---|
| `EXPERIMENT_DIR` | `scaling_experiment` | where logs/checkpoints/plots/configs live |
| `DATA_DIR` | `ncbi_ftp_output` | raw FASTA source |
| `EVAL_TOKENS` | `50000000` | held-out eval set size in tokens |
| `ISO_TOKEN_POINTS` | `8` | N values in Phase 1 |
| `ISO_PARAM_MODEL_PARAMS` | `10000000` | fixed N used in Phase 2 |
| `ISO_FLOP_BUDGETS` | `1e15,3e15,1e16,3e16` | compute budgets (FLOPs) for Phase 3 |
| `ISO_FLOP_POINTS` | `5` | N values per Phase-3 budget |
| `BATCH_SIZE` | `32` | per-device micro-batch |
| `GRAD_ACCUM` | `1` | gradient-accumulation steps (effective batch = `BATCH_SIZE × GRAD_ACCUM`) |
| `MAX_SEQ_LEN` | `2048` | max DNA tokens per sequence |
| `EVAL_EVERY` | `500` | eval frequency in optimiser steps |
| `LR` | `1e-4` | AdamW learning rate |
| `SEED` | `42` | global seed |

Example: larger sweep, bigger batches via accumulation, 5 compute budgets:

```bash
ISO_TOKEN_POINTS=10 \
ISO_FLOP_POINTS=6 \
ISO_FLOP_BUDGETS="1e15,3e15,1e16,3e16,1e17" \
BATCH_SIZE=64 GRAD_ACCUM=4 \
./scripts/run_complete_scaling_analysis.sh
```

### 6b. `src/train.py` flags

| Flag | Default | Notes |
|---|---|---|
| `--config` | required | path to architecture JSON |
| `--data_path` | (synthetic) | training FASTA(s); multiple paths allowed |
| `--eval_data_path` | 10% split | **recommended**: external held-out FASTA |
| `--num_synthetic` | 10000 | synthetic sequences (only if `--data_path` absent) |
| `--min_seq_len` / `--max_seq_len` | 64 / 2048 | length filter / truncation |
| `--batch_size` | 32 | |
| `--gradient_accumulation_steps` | 1 | |
| `--grad_clip` | 1.0 | max grad norm |
| `--learning_rate` | 1e-4 | AdamW |
| `--weight_decay` | 0.01 | |
| `--num_epochs` | 1 | **spec requires 1** |
| `--warmup_frac` | 0.05 | warmup as a fraction of total optimiser steps |
| `--max_steps` | None | hard cap in optimiser steps |
| `--mask_prob` | 0.15 | MLM mask ratio |
| `--eval_every` | 500 | eval cadence in optimiser steps |
| `--save_dir` / `--log_dir` | `checkpoints` / `logs` | |
| `--seed` | 42 | |
| `--num_workers` | 2 | DataLoader workers |
| `--device` | auto | `cuda` or `cpu`; auto-detects |
| `--no_progress` | off | hide tqdm bars |

### 6c. Architecture config JSON

A per-run config looks like:

```json
{
  "d_model":   256,
  "n_heads":   4,
  "n_layers":  4,
  "d_ff":      1024,
  "dropout":   0.1,
  "vocab_size": 7,
  "max_seq_len": 2048
}
```

`setup_scaling_experiment.py` generates these automatically; you only hand-write one if you're running a single ad-hoc training.

---

## 7. Recommended model scales

The repo is capped at **500 M params** (spec). The default Iso-Token sweep spans **100 K → 200 M** across 8 log-spaced points — this is a good baseline.

### Which scales to use

| Run scenario | Compute budget | Suggested `N` range | Sweep points |
|---|---|---|---|
| Tiny debug (CPU) | < 1e13 FLOPs | 50 K – 5 M | 4 |
| Small GPU run | 1e15 – 1e16 | 100 K – 50 M | 6 |
| **Default full run** | 1e15 – 1e16 | 100 K – 200 M | 8 |
| Rich sweep | 1e15 – 1e17 | 500 K – 500 M | 10 |

Rough **Chinchilla-optimal** N for a given compute budget C: `N_opt ≈ sqrt(C / 120)` tokens-per-parameter ≈ 20. That means:

| Compute budget `C` (FLOPs) | Chinchilla-optimal `N` | Optimal `D` | Wall time on one A100 (est.) |
|---|---|---|---|
| 1e15 | ~3 M | ~60 M tokens | ~10 min |
| 3e15 | ~5 M | ~100 M | ~30 min |
| 1e16 | ~10 M | ~170 M | ~1.5 hr |
| 3e16 | ~18 M | ~280 M | ~4 hr |
| 1e17 | ~30 M | ~560 M | ~12 hr |

For **Phase 3 (IsoFLOP)**, at each compute budget the N-sweep should bracket the optimum by ~10× on each side (so that the parabola fit has a clear valley). The defaults already do this via log-spaced point selection around `sqrt(C/120)`.

For **Phase 2 (Iso-Param)**, use a mid-range model (`ISO_PARAM_MODEL_PARAMS`, default 10 M) where D can vary over ~1.5 decades without hitting the "way under-trained" regime at small D.

### Heuristics

- Keep `head_dim = d_model / n_heads ≥ 32`.
- `d_ff = 4 · d_model` is the standard ratio used here.
- For runs > 100 M params, bump `GRAD_ACCUM` so the effective batch stays ≥ 256.
- If OOM, drop `BATCH_SIZE` first, then `MAX_SEQ_LEN`, then fall back to gradient accumulation.

---

## 8. Outputs

After a full run:

```
scaling_experiment/
├── configs/                        per-run architecture JSONs
├── data_samples/
│   ├── split/{train_full,eval}.fasta    fixed held-out eval set
│   ├── iso_param/sample_*.fasta         D-sweep FASTAs
│   └── iso_flop/<run>.fasta             token-budgeted samples
├── checkpoints/<phase>/                 best_model_<run>.pth
├── logs/training_log_<run>.json         one per run
└── plots/
    ├── loss_vs_params.png               \
    ├── loss_vs_tokens.png                |  from plot_scaling.py
    ├── loss_vs_flops.png                /
    ├── scaling_exponents.json           α, β, γ in L ∝ N^α, D^β, C^γ
    ├── iso_token_analysis.png           \
    ├── iso_param_analysis.png            |  from plot_iso_analysis.py
    ├── iso_flop_analysis.png             |
    ├── loss_landscape_2d.png            /
    ├── ttp_analysis.png                 4-panel: IsoFLOP curves, N_opt(C), D_opt(C), loss vs D/N
    └── ttp_summary.json                 bucket sizes + parabola vertices
```

### Training log schema

Each `training_log_<run>.json`:

```json
{
  "run_name": "iso_token_model_10m",
  "config": { ...architecture... },
  "num_parameters": 10000128,
  "final_tokens_seen": 170000000,
  "final_flops": 1.02e+16,
  "final_eval_loss": 1.842,
  "best_eval_loss": 1.841,
  "total_opt_steps": 5312,
  "wall_seconds": 5240.7,
  "log": [
    { "step": 500, "tokens_seen": 32000000, "train_loss": 2.1, "eval_loss": 2.05,
      "eval_perplexity": 7.77, "num_parameters": 10000128,
      "flops": 1.92e+15, "learning_rate": 1e-4, "phase": "periodic" },
    ...,
    { "phase": "final", ... }
  ]
}
```

---

## 9. GPU notes

- Device auto-detected; pass `--device cuda` or `--device cpu` to override.
- On CUDA, `pin_memory=True` + `.to(device, non_blocking=True)` are used automatically.
- `num_workers` defaults to 2 (Linux-friendly). Bump up if data loading is the bottleneck.
- Checkpoints are portable (saved as `state_dict`), so you can move them between CPU and GPU hosts.

---

## 10. One-off training

To train a single config outside the orchestrator:

```bash
# make a config
cat > my_arch.json <<'EOF'
{"d_model":256,"n_heads":4,"n_layers":4,"d_ff":1024,"dropout":0.1,"vocab_size":7,"max_seq_len":2048}
EOF

# train
python src/train.py \
  --config my_arch.json --run_name my_run \
  --data_path ncbi_ftp_output/human_CDS.fasta \
  --eval_data_path scaling_experiment/data_samples/split/eval.fasta \
  --batch_size 32 --num_epochs 1 \
  --save_dir checkpoints --log_dir logs
```

---

## 11. Design notes

- **Spec compliance**: exactly 1 epoch; no data reuse. Iso-FLOP knob is `D` (via pre-sampled FASTAs), not extra epochs. The trainer itself accepts `--num_epochs N` for flexibility, but both `run_complete_scaling_analysis.sh` and `test_scaling_orchestration.sh` always pass `--num_epochs 1` — the constraint is enforced by convention in the scripts, not by a hard code guard.
- **Fixed eval set**: one FASTA carved at the start of the pipeline; every run in every phase evaluates against it. This makes losses comparable across `(N, D)`.
- **Token-budgeted sampling**: per-run FASTAs are sized to hit a target token count, not a target sequence count, so iso-FLOP runs actually land on their compute budget.
- **Multi-budget IsoFLOP**: Phase 3 runs a full N-sweep at each of several `C` values, so `N_opt ~ C^a` and `D_opt ~ C^b` are real regressions, not one-point extrapolations.
- **Final-eval-only plots**: every plot script consumes one point per run (the guaranteed final eval), so cross-run fits are not contaminated by intermediate-checkpoint noise from differently-scheduled runs.
