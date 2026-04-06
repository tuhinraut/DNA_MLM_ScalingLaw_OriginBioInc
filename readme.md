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
# Run small-scale test with synthetic data
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
# Recommended: NCBI FTP (most reliable)
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
- **Input**: Fixed compute budget (~4.19e+16 FLOPs), based on 1M model with 100% data
- **Process**: Train 7 models (1M, 2M, 5M, 10M, 25M, 50M, 100M params) with variable data sampling
- **Data scaling**: Explicit fractions: 100%, 50%, 20%, 10%, 4%, 2%, 1%
- **Output**: Optimal N/D allocation for fixed compute

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

### Setup (Optional - Usually Automatic)

Setup runs **automatically** when you execute `run_complete_scaling_analysis.sh`. 
Manual setup is only needed if you want to inspect configs before running:

```bash
python scripts/setup_scaling_experiment.py
```

To skip auto-setup (e.g., when resuming a previous run):
```bash
./scripts/run_complete_scaling_analysis.sh --skip-setup
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
**Ready to start?** Run: `./scripts/run_complete_scaling_analysis.sh`

*Setup is now integrated - configs are generated automatically. Use `--skip-setup` to skip if resuming a previous run.*

**Note:** This repository includes a `.gitignore` file that excludes generated data, checkpoints, and logs from version control.
