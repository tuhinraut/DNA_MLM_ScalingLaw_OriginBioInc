# Scaling Laws for DNA Sequence Models — Take-Home

## Purpose

The goal of this assignment is to investigate **scaling laws** for a Masked
Language Model trained on DNA coding sequences.  Specifically:

- Does **test-set cross-entropy loss** decrease as a predictable power law
  when you scale up model parameters, training tokens, or total compute
  (FLOPs)?
- What are the fitted exponents for each scaling axis?
- Is there an **optimal token-to-parameter ratio (TTP)** — i.e. for a fixed
  compute budget, what balance of model size vs. data minimises loss?

These are the same questions studied in Kaplan et al. (2020) and
Hoffmann et al. ("Chinchilla", 2022), but applied to genomic sequences
instead of natural language.

To answer them you need to train many models at different scales and
collect their final eval losses.  This assignment tests your ability to
build the **experiment infrastructure** that makes that possible — a
complete, runnable pipeline that goes from raw data to final scaling
plots.

### Two phases

**Phase 1 — Code submission (this take-home).**
Fix bugs, implement TODOs, write the data pipeline and orchestration.
Everything should be ready to run end-to-end with a single command.
You can verify correctness with the synthetic data generator on CPU.

**Phase 2 — GPU run.**
After the code submission is reviewed, GPU access will be provided to
execute the full sweep and produce the final results: scaling plots
(cross-entropy vs. FLOPs / parameters / tokens) and the TTP analysis.

---

## Dataset

Choose a dataset of **protein-coding DNA sequences (CDS)**.  Common sources
include Ensembl, NCBI RefSeq, or GENCODE.  The choice of organism(s) and
the total data volume are up to you — just make sure there is enough data
to meaningfully train the range of model sizes you plan to sweep.

Implement `data_download.py` so that it:

1. Downloads the raw data.
2. Cleans / filters as needed.
3. Saves processed sequences in a file format of your choice (JSON lines,
   Parquet, plain text — whatever you prefer).

Then implement `load_sequences()` in `dataset.py` to read that format and
return a list of DNA strings.

`generate_synthetic_sequences()` is provided for fast debugging on CPU.

---

## Constraints

| Constraint | Value |
|---|---|
| Max model size | **≤ 500 M parameters** |
| Epochs | **Exactly 1** — no repeated data.  Each token is seen once. |
| FLOPs estimate | Use **C ≈ 6 × N × D** (N = trainable params, D = tokens). |

All other experiment-design decisions — FLOPs range, number of data points
per axis, evaluation split, learning rate schedule tuning, etc. — are yours
to make.  The number of points along each scaling axis should be a
**configurable parameter** in your orchestration, not hard-coded.

---

## File Descriptions

Each file has a mix of working code, **intentional bugs**, and **TODO
stubs**.  The descriptions below tell you what *should* be present when
everything is correct.

### `model.py` — MLM Transformer

Encoder-only Transformer for masked language modelling.

- **Architecture**: Pre-LayerNorm, Rotary Position Embeddings (RoPE),
  multi-head self-attention, feed-forward network with GELU activation,
  linear MLM prediction head.
- **Vocabulary**: `[PAD]`, `[MASK]`, A, C, G, T, N.
- **Key dimensions**: `d_model`, `n_heads`, `head_dim`, `d_ff`, `n_layers`.
- **Components**: `RotaryPositionEmbedding`, `rotate_half`,
  `apply_rotary_pos_emb`, `MultiHeadAttention`, `FeedForward`,
  `TransformerBlock`, `DNATransformerMLM`.
- **Weight initialisation**: embeddings, linear projections, layer norms,
  and the output head each have standard initialisation schemes (TODO).

### `dataset.py` — Tokenisation & Data Loading

- **Vocabulary**: `[PAD]`, `[MASK]`, A, C, G, T, N  (7 tokens).
- **MLM masking**: configurable mask probability.  Selected positions are
  replaced with `[MASK]`; all others are ignored in the loss.
- **Padding**: every sample is padded to `max_seq_len`; `attention_mask`
  distinguishes real tokens from padding.
- **`load_sequences`** (TODO): reads the processed data file produced by
  `data_download.py`.

### `loss.py` — MLM Loss

- Cross-entropy computed **only over masked positions** (labels = −100 at
  non-masked sites).
- Returns both the scalar loss and the count of masked tokens — needed for
  correct metric aggregation across batches with varying mask counts.

### `train.py` — Training Driver

- **Config**: model architecture is read from a **JSON file** passed via
  `--config` (you create your own configs).
- **Optimiser**: AdamW with configurable learning rate and weight decay.
- **LR schedule**: cosine annealing with linear warm-up.
- **Gradient accumulation**: configurable via
  `--gradient_accumulation_steps`.  Effective batch size =
  `batch_size × gradient_accumulation_steps`.  Clipping, optimiser step,
  and scheduler step should all happen at accumulation boundaries.
- **Token accounting**: only *real* (non-padding) tokens should be counted
  toward tokens-seen for accurate FLOPs and scaling analysis.
- **Evaluation**: periodic eval on held-out split tracking loss and
  perplexity (TODO).
- **Logging**: JSON log per run with step, losses, perplexity, tokens seen,
  parameter count, estimated FLOPs, learning rate.

### `data_download.py` — Data Pipeline (TODO)

- Downloads protein-coding DNA sequences from your chosen source.
- Processes and saves them in a format that `load_sequences()` can read.

### `plot_scaling.py` — Scaling Plots

- Reads training log JSONs from the `logs/` directory.
- Fits power laws in log-log space.
- Produces three plots: Loss vs. Parameters, Loss vs. Tokens, Loss vs.
  FLOPs.

### `plot_ttp.py` — TTP Analysis Plot (TODO)

- For different compute (FLOPs) iso-curves, plots loss as a function of
  TTP ratio (D / N) to show where the optimum lies at each scale.

---

## Model Configurations

**No pre-made configs are provided.**  `train.py` reads architecture
hyper-parameters from a JSON file:

```json
{
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 1024,
    "n_layers": 6
}
```

For a thorough scaling study you need configs spanning several orders of
magnitude in parameter count (up to the 500 M ceiling).

**Recommendation**: write a helper script that generates config JSONs,
estimates parameter counts and FLOPs (use **C ≈ 6 N D**), and orchestrates
the full sweep — launching `train.py` for each (config, data-size)
combination, then calling the plotting scripts to produce the final
outputs.  The number of scale points should be configurable.

---

## What You Need To Do

### Part 1 — Code Submission (this take-home)

**A. Bug fixes.**
Find and fix intentional bugs in `model.py`, `dataset.py`, and `loss.py`.
Briefly document each fix.

**B. Missing implementations.**

| File | Component |
|---|---|
| `model.py` | `DNATransformerMLM._init_weights` |
| `dataset.py` | `load_sequences` |
| `train.py` | `evaluate()` and training loop |
| `data_download.py` | Entire script |
| `plot_ttp.py` | IsoFLOP TTP analysis plot |

**C. Orchestration.**
Build **one executable** that does everything end-to-end:
generate configs → download data → train all runs → produce plots.
File simplicity and smart orchestration of the different scripts is
expected.  The sweep should be reproducible and configurable without
editing source code.

### Part 2 — GPU Run (after review)

Once approved, you will receive GPU access to execute the sweep and
deliver:

1. **Loss vs. Parameters** (log-log) with power-law fit.
2. **Loss vs. Training Tokens** (log-log) with power-law fit.
3. **Loss vs. Compute (FLOPs)** (log-log) with power-law fit.
4. **TTP analysis** — for several FLOPs budgets, loss vs. D / N.
5. Short write-up (1–2 pages) discussing fitted exponents and whether
   an optimal TTP exists for DNA sequence modelling.

---

## Quick Start

```bash
pip install -r requirements.txt

# Debug with synthetic data
echo '{"d_model":64,"n_heads":2,"d_ff":256,"n_layers":2}' > tiny.json
python train.py --config tiny.json --num_synthetic 5000
```

---

## TODO Checklist (repeated for reference)

- [ ] Bug fixes in `model.py`, `dataset.py`, `loss.py`
- [ ] `model.py` — `_init_weights`
- [ ] `dataset.py` — `load_sequences`
- [ ] `train.py` — `evaluate()` and training loop
- [ ] `data_download.py` — full implementation
- [ ] `plot_ttp.py` — TTP analysis plot
- [ ] Orchestration script — single entry point for the entire sweep
