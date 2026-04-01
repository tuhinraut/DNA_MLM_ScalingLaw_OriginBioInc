# DNA MLM Scaling Laws - Complete Package

Complete system for conducting scaling law analysis on DNA Masked Language Models.

## What's Included

### Core Training Code (`src/`)
- `train.py` - Main training script with MLM objective
- `model.py` - DNATransformerMLM with RoPE and Pre-LN
- `dataset.py` - DNASequenceDataset with tokenization and masking
- `loss.py` - MLMCrossEntropyLoss for masked positions

### Data Downloaders (`data_downloaders/`)
- `download_ncbi_ftp.py` - Bulk download from NCBI Genomes FTP (recommended)

### Experiment Scripts (`scripts/`)
- `setup_scaling_experiment.py` - Generate all configs
- `run_complete_scaling_analysis.sh` - **MASTER SCRIPT** - Run full-scale analysis
- `test_scaling_orchestration.sh` - **TEST SCRIPT** - Quick local test with synthetic data
- `plot_scaling.py` - Generate scaling law plots (loss vs params/tokens/FLOPs)
- `plot_ttp.py` - Chinchilla-style IsoFLOP analysis with parabolic fits (classic 3-panel figure)

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy biopython
```

### 2. Quick Local Test (Recommended First)

Before running the full analysis, test the pipeline locally with synthetic data:

```bash
# Run small-scale test with synthetic data (~10-30 minutes on CPU)
./scripts/test_scaling_orchestration.sh

# This creates:
# - test_experiment/logs/          # Training logs
# - test_experiment/checkpoints/ # Model checkpoints  
# - test_experiment/plots/         # Scaling law plots
# - test_experiment/results/       # Phase summaries
```

The test script:
- Uses 1,000 synthetic DNA sequences (no download needed)
- Trains 4 small models (100K to 4M parameters)
- Logs frequently (every 50 steps) for detailed curves
- Generates all plots automatically

### 3. Download Data (for Full Analysis)

```bash
# Recommended: NCBI FTP (fastest, most reliable)
python data_downloaders/download_ncbi_ftp.py

# Or download specific species:
python data_downloaders/download_ncbi_ftp.py --species human mouse rat
```

### 4. Run Complete Analysis

```bash
# One command runs everything:
./scripts/run_complete_scaling_analysis.sh

# Or with auto-cleanup:
./scripts/run_complete_scaling_analysis.sh --cleanup
```

## Three-Phase Analysis

### Phase 1: Iso-Token (Find Optimal Model Size)
- **Input**: Full dataset (~3.95B tokens)
- **Process**: Train 11 models (100K → 200M params) for 1 epoch each
- **Output**: Best model architecture

### Phase 2: Iso-Param (Find Ideal Token Count)
- **Input**: Best model from Phase 1
- **Process**: Train on 5 data samples (6.25%, 12.5%, 25%, 50%, 100%)
- **Output**: Optimal D (tokens) for fixed N (parameters)

### Phase 3: Iso-FLOP (Find Optimal N/D Allocation)
- **Input**: Fixed FLOPs budget (~1.34e+18)
- **Process**: Train 5 models with varying epochs (1→8)
- **Output**: Best model size / data trade-off

## Directory Structure After Setup

```
dna_mlm_scaling_package/
├── src/                    # Core training code
├── scripts/                # Orchestration scripts
├── data_downloaders/       # Data acquisition
├── experiments/            # Created during run
│   ├── data_samples/       # Random samples for Phase 2
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/               # Training logs
│   └── results/            # Phase summaries
├── config.json             # Master configuration
├── setup.sh                # One-time setup
└── README.md               # This file
```

## Usage Options

### Download Data Only
```bash
python data_downloaders/download_ncbi_ftp.py
```

### Setup Only (Generate Configs)
```bash
python scripts/setup_scaling_experiment.py
```

### Run Specific Phase
```bash
# Phase 1 only
./scripts/run_complete_scaling_analysis.sh --skip-samples --skip-param --skip-flop

# Phase 2 only (requires Phase 1 complete)
./scripts/run_complete_scaling_analysis.sh --skip-setup --skip-samples --skip-token --skip-flop
```

### Custom Data Path
```bash
# If data is elsewhere
export DATA_DIR=/path/to/your/fasta/files
./scripts/run_complete_scaling_analysis.sh
```

## Data Requirements

### Minimum (for testing)
- 1 species, ~50K sequences, ~100M tokens
- Can run small models (100K-4M params)

### Recommended (for full analysis)
- 33 species from NCBI FTP
- 1.89M sequences, ~3.95B tokens
- Supports all model sizes up to 200M params

### Data Sources

**NCBI FTP (Recommended)**
- 33 species available
- Pre-filtered protein-coding CDS
- Fast bulk download
- Script: `download_ncbi_ftp.py`


## Model Configurations

| Model | d_model | n_heads | n_layers | d_ff | Params |
|-------|---------|---------|----------|------|--------|
| 100K | 64 | 2 | 2 | 256 | ~100K |
| 400K | 128 | 4 | 2 | 512 | ~400K |
| 1M | 192 | 3 | 4 | 768 | ~1.8M |
| 4M | 256 | 4 | 4 | 1024 | ~3.2M |
| 7M | 384 | 6 | 4 | 1536 | ~7.1M |
| 12M | 512 | 8 | 4 | 2048 | ~12.6M |
| 25M | 512 | 8 | 8 | 2048 | ~25.2M |
| 36M | 768 | 12 | 6 | 3072 | ~42.5M |
| 48M | 768 | 12 | 8 | 3072 | ~56.7M |
| 100M | 1024 | 16 | 8 | 4096 | ~100.7M |
| 200M | 1280 | 16 | 10 | 5120 | ~196.7M |

## Expected Runtime

### Test Run (Synthetic Data, 4 Small Models)

| Hardware | Time | Details |
|----------|------|---------|
| CPU | 10-30 min | 4 models (100K-4M params), 1K synthetic sequences |
| GPU | 2-5 min | Same as above, much faster |

### Full Run (Real Data, 11 Models)

With GPU (V100/A100):
- **Phase 1**: 2-6 hours (11 models)
- **Phase 2**: 1-3 hours (5 samples)
- **Phase 3**: 2-4 hours (5 models)
- **Total**: 5-13 hours

With CPU:
- **Not recommended** for models > 4M params
- Would take 2-6 days

## Disk Space

### Test Run (Synthetic Data)

| Component | Size | Notes |
|-----------|------|-------|
| Test logs | ~1-2 MB | Frequent logging (every 50 steps) |
| Test checkpoints | ~10-20 MB | 4 small models |
| Test plots | ~500 KB | PNG images |
| **Total** | **~20 MB** | Negligible space |

### Full Run (Real Data)

| Component | Size | Notes |
|-----------|------|-------|
| Package itself | ~50 KB | Code only |
| Downloaded data | 4-8 GB | Depending on sources |
| Data samples | 7-8 GB | Generated during Phase 2 |
| Checkpoints | 200-500 MB | Saved per model |
| Logs | 50-100 MB | JSON metrics |
| **Total** | **12-16 GB** | With all data |

Cleanup: Use `--cleanup` flag or manually delete:
- `experiments/data_samples/` (7-8 GB)
- `experiments/checkpoints/` (200-500 MB)
- `test_experiment/` (test run data)

## Output Files

### Test Run Output
```
test_experiment/
├── logs/training_log_*.json       # Detailed training logs
├── checkpoints/{phase}/            # Model checkpoints
├── plots/
│   ├── loss_vs_params.png         # Scaling law plot
│   ├── loss_vs_tokens.png         # Scaling law plot
│   ├── loss_vs_flops.png          # Scaling law plot
│   ├── ttp_analysis.png           # Simple IsoFLOP plot
│   └── ttp_analysis_chinchilla.png # Chinchilla-style 3-panel plot
└── results/phase*.json            # Phase summaries
```

### Full Run Output
```
experiments/logs/training_log_{run_name}.json
experiments/checkpoints/{phase}/best_model_{run_name}.pth
experiments/logs/{run_name}.stdout
```

### Phase Summaries
```
experiments/results/phase1_iso_token.json    # Best model
experiments/results/phase2_iso_param.json     # Optimal token count
experiments/results/phase3_iso_flop.json      # N/D allocation
```

## Analyzing Results

### Recommended: Chinchilla-Style Analysis

```bash
# Generate the classic 3-panel Chinchilla figure (Figure 3 style)
python scripts/plot_ttp.py --log_dir experiments/logs --output_dir plots --style chinchilla

# Generate comprehensive Iso-analysis with 2D relationships
python scripts/plot_iso_analysis.py --log_dir experiments/logs --output_dir plots
```

### Test Run Results
```bash
# The test script generates plots automatically
# View generated plots:
ls test_experiment/plots/

# Regenerate with improved visualizations:
python scripts/plot_ttp.py --log_dir test_experiment/logs --output_dir test_experiment/plots --style chinchilla
python scripts/plot_iso_analysis.py --log_dir test_experiment/logs --output_dir test_experiment/plots
```

### Full Run Results
```bash
# View best model from Phase 1
cat experiments/results/phase1_iso_token.json

# View token count analysis
cat experiments/results/phase2_iso_param.json

# View FLOP allocation analysis
cat experiments/results/phase3_iso_flop.json

# Classic Chinchilla 3-panel figure with parabolic fits
python scripts/plot_ttp.py --log_dir experiments/logs --output_dir plots --style chinchilla

# 2D Iso-analysis (training curves by hyperparameter)
python scripts/plot_iso_analysis.py --log_dir experiments/logs --output_dir plots

# Legacy plots (basic scaling laws)
python scripts/plot_scaling.py --log_dir experiments/logs --output_dir plots
```

### Generated Plots

**`plot_ttp.py --style chinchilla` generates:**
- `ttp_analysis_chinchilla.png` - Classic 3-panel figure:
  - Left: IsoFLOP curves (loss vs N, parabolic fits with valleys)
  - Center: Optimal N vs FLOPs (power law fit)
  - Right: Optimal D vs FLOPs (power law fit)
  - Shows the key finding: N and D should scale equally (~C^0.5)

**`plot_iso_analysis.py` generates:**
- `iso_token_analysis.png` - Iso-Token: Training curves by model size, fixed data
- `iso_param_analysis.png` - Iso-Param: Training curves by data size, fixed model
- `loss_landscape_2d.png` - 2D loss landscape in (N, D) space with efficient frontier
- `scaling_summary.png` - Comprehensive dashboard with all analyses

## Customization

### Change Batch Size
Edit `scripts/run_complete_scaling_analysis.sh`:
```bash
BATCH_SIZE=16  # If OOM with 32
```

### Add/Remove Model Sizes
Edit `scripts/setup_scaling_experiment.py` to modify model configurations:
```bash
python scripts/setup_scaling_experiment.py
```

### Use Different Data
Modify data download scripts or provide custom FASTA files:
```bash
./scripts/run_complete_scaling_analysis.sh --data-dir /path/to/your/data
```

## Troubleshooting

### "Out of memory"
- Reduce `BATCH_SIZE` in run script
- Use gradient accumulation (modify train.py)
- Train smaller models only

### "Out of disk space"
- Use `--cleanup` flag
- Delete checkpoints after each phase
- Store data on external drive

### "Data download fails"
- Check network connectivity
- Use smaller species subset
- Verify NCBI FTP access

### "Phase 1 fails partway"
- Check logs: `tail -f experiments/logs/*.stdout`
- Resume with `--skip-setup --skip-samples` flags
- Check CUDA/GPU availability

## Scaling Law Theory

This package implements methodology from:
- **Hoffmann et al. (2022)**: "Training Compute-Optimal Large Language Models" (Chinchilla)
- **Kaplan et al. (2020)**: "Scaling Laws for Neural Language Models"

Key finding: Model size (N) and data (D) should scale equally (1:1 ratio) for compute-optimal training, unlike Kaplan's prediction.

Formula: `C ≈ 6 × N × D`

Where:
- C = FLOPs (compute)
- N = Model parameters
- D = Training tokens

## Citation

If you use this package for research:

```bibtex
@software{dna_mlm_scaling,
  title = {DNA MLM Scaling Laws: Complete Analysis Package},
  author = {Your Name},
  year = {2026},
  note = {Based on Chinchilla scaling law methodology}
}
```

## License

MIT License - See LICENSE file

## Support

- Review logs in `experiments/logs/`
- Verify data in downloaded folders
- Check `.gitignore` for excluded files

---

**Ready to start?** Run: `./setup.sh` then `./scripts/run_complete_scaling_analysis.sh`

**Note:** This repository includes a `.gitignore` file that excludes generated data, checkpoints, and logs from version control.
