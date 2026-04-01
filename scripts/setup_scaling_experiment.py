#!/usr/bin/env python3
"""Setup scaling law experiment with generated configs.

Creates folder structure and config files for:
- Phase 1: Iso-Token (vary model size, constant data)
- Phase 2: Iso-Param (constant model, vary data via random sampling)
- Phase 3: Iso-FLOP (constant compute, vary N/D ratio)
"""

import json
import os
from pathlib import Path
import math


def generate_model_config(d_model, n_heads, n_layers, d_ff=None, dropout=0.1):
    """Generate a model config dict."""
    if d_ff is None:
        d_ff = 4 * d_model
    
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "vocab_size": 7,
        "max_seq_len": 2048
    }


def estimate_params(config):
    """Estimate parameter count for a config."""
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = config["d_ff"]
    vocab_size = config.get("vocab_size", 7)
    
    embeddings = 2 * vocab_size * d_model
    attn_params = 4 * d_model * d_model
    ffn_params = 2 * d_model * d_ff
    layer_norm_params = 4 * d_model
    per_layer = attn_params + ffn_params + layer_norm_params
    transformer = n_layers * per_layer
    
    return embeddings + transformer


def setup_folder_structure(base_dir="scaling_experiment"):
    """Create folder structure for the experiment."""
    dirs = {
        "configs": {
            "iso_token": {},
            "iso_param": {},
            "iso_flop": {}
        },
        "logs": {},
        "checkpoints": {
            "iso_token": {},
            "iso_param": {},
            "iso_flop": {}
        },
        "results": {},
        "data_samples": {}  # Pre-sampled FASTA files
    }
    
    def create_dirs(path, structure):
        for name, sub in structure.items():
            new_path = path / name
            new_path.mkdir(parents=True, exist_ok=True)
            if sub:
                create_dirs(new_path, sub)
    
    base = Path(base_dir)
    create_dirs(base, dirs)
    print(f"Created folder structure at: {base.absolute()}")
    return base


def generate_iso_token_configs(config_dir):
    """Generate configs for Iso-Token analysis.
    
    Vary model size from 100K to ~200M params (Chinchilla optimal).
    Use consistent ratios: d_ff = 4*d_model, heads maintain head_dim >= 32.
    """
    configs = [
        # (d_model, n_heads, n_layers, name_suffix)
        (64, 2, 2, "100k"),      # ~100K
        (128, 4, 2, "400k"),     # ~400K
        (192, 3, 4, "1m"),       # ~1.8M
        (256, 4, 4, "4m"),       # ~3.2M
        (384, 6, 4, "7m"),       # ~7.1M
        (512, 8, 4, "12m"),      # ~12.6M
        (512, 8, 8, "25m"),      # ~25M
        (768, 12, 6, "36m"),     # ~42M
        (768, 12, 8, "48m"),     # ~57M
        (1024, 16, 8, "100m"),   # ~100M
        (1280, 16, 10, "200m"),  # ~200M (Chinchilla optimal for ~4B tokens)
    ]
    
    config_dir = Path(config_dir)
    generated = []
    
    for d_model, n_heads, n_layers, suffix in configs:
        config = generate_model_config(d_model, n_heads, n_layers)
        params = estimate_params(config)
        
        filename = f"model_{suffix}.json"
        filepath = config_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        generated.append({
            'file': filename,
            'params': params,
            'config': config
        })
        print(f"  {filename}: {params:,} params")
    
    return generated


def generate_iso_param_configs(config_dir, best_model_config):
    """Generate configs for Iso-Param analysis.
    
    All use the same model (best from Iso-Token), just varying data samples.
    We only need one config file since data sampling happens at runtime.
    """
    config_dir = Path(config_dir)
    
    # Copy the best model config for reference
    filepath = config_dir / "iso_param_model.json"
    with open(filepath, 'w') as f:
        json.dump(best_model_config, f, indent=2)
    
    params = estimate_params(best_model_config)
    print(f"  Iso-Param model: {params:,} params")
    
    return filepath


def generate_iso_flop_configs(config_dir):
    """Generate configs for Iso-FLOP analysis.
    
    Fixed compute budget based on 1M model with 100% data:
    C_fixed = 6 * N_1M * D_total
    
    For each model N, data is sampled as:
    D = C_fixed / (6 * N) = (N_1M / N) * D_total
    
    Larger models get less data, smaller models get more (via sampling).
    """
    # Total available tokens (from NCBI data)
    D_total = 3_950_000_000  # ~3.95B tokens
    
    # Reference: 1M model parameters
    N_ref = 1_770_000  # ~1.77M params (192, 3, 4 config)
    
    # Fixed compute budget: C = 6 * N_ref * D_total
    C_fixed = 6 * N_ref * D_total
    
    # Model sizes to test (same as Iso-Token, from 1M up to 200M)
    # We use models where D_required <= D_total (data fits in corpus)
    configs = [
        (192, 3, 4, "1m"),       # ~1.77M (reference, 100% data)
        (256, 4, 4, "4m"),       # ~3.2M (55.3% data)
        (384, 6, 4, "7m"),       # ~7.1M (24.9% data)
        (512, 8, 4, "12m"),      # ~12.6M (14.0% data)
        (512, 8, 8, "25m"),      # ~25.2M (7.0% data)
        (768, 12, 6, "36m"),     # ~42.5M (4.2% data)
        (768, 12, 8, "48m"),     # ~56.7M (3.1% data)
        (1024, 16, 8, "100m"),   # ~100.7M (1.8% data)
        (1280, 16, 10, "200m"),  # ~196.7M (0.9% data)
    ]
    
    config_dir = Path(config_dir)
    generated = []
    
    print(f"\n  Fixed Compute Budget: {C_fixed:.2e} FLOPs")
    print(f"  Reference: 1M model ({N_ref:,} params) with 100% data")
    print(f"  Total tokens available: {D_total:.2e}")
    print(f"  Data strategy: Sample fraction = N_ref / N_model\n")
    
    for d_model, n_heads, n_layers, suffix in configs:
        config = generate_model_config(d_model, n_heads, n_layers)
        N = estimate_params(config)
        
        # Calculate required data for fixed compute
        # C_fixed = 6 * N * D => D = C_fixed / (6 * N)
        D_required = C_fixed / (6 * N)
        
        # Data fraction relative to full corpus
        data_fraction = N_ref / N  # Simplifies to D_required / D_total
        data_pct = data_fraction * 100
        
        # All models use 1 epoch on sampled data (no repetition)
        epochs = 1
        
        # Store config with metadata
        config_with_meta = {
            **config,
            "_meta": {
                "target_flops": C_fixed,
                "params": N,
                "reference_params": N_ref,
                "data_fraction": data_fraction,
                "data_percent": round(data_pct, 2),
                "required_tokens": int(D_required),
                "epochs": epochs,
                "isoflop_mode": "fixed_compute_variable_data"
            }
        }
        
        filename = f"iso_flop_{suffix}.json"
        filepath = config_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
        
        generated.append({
            'file': filename,
            'params': N,
            'data_fraction': data_fraction,
            'data_percent': data_pct,
            'required_tokens': int(D_required)
        })
        
        print(f"  {filename}: {N:>10,} params, {data_pct:>5.1f}% data ({int(D_required):>12,} tokens)")
    
    return generated


def create_sampling_utility(experiment_dir):
    """Create a Python script for random data sampling directly to FASTA."""
    script_content = '''#!/usr/bin/env python3
"""Data sampling utility for Iso-Param analysis.

Loads all sequences from NCBI FTP data, randomly samples a percentage,
and outputs directly to a FASTA file.
"""

import argparse
import random
from pathlib import Path


def sample_to_fasta(data_dir, sample_percent, output_file, min_len=64, max_len=2048, seed=42):
    """Create a random sample of sequences and save to FASTA.
    
    Args:
        data_dir: Directory containing *_CDS.fasta files
        sample_percent: Percentage of sequences to sample (0-100)
        output_file: Path to save sampled FASTA
        min_len: Minimum sequence length to keep
        max_len: Maximum sequence length
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Collect all sequences from all files
    data_path = Path(data_dir)
    fasta_files = list(data_path.glob("*_CDS.fasta"))
    
    print(f"Loading from {len(fasta_files)} FASTA files...")
    
    all_records = []  # [(header, sequence), ...]
    
    for f in fasta_files:
        count = 0
        with open(f) as fp:
            current_header = None
            current_seq = []
            
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous record
                    if current_header and current_seq:
                        seq = ''.join(current_seq)
                        seq_len = len(seq)
                        if min_len <= seq_len <= max_len:
                            all_records.append((current_header, seq))
                            count += 1
                        elif seq_len > max_len:
                            # Truncate
                            all_records.append((current_header, seq[:max_len]))
                            count += 1
                    # Start new record
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last record
            if current_header and current_seq:
                seq = ''.join(current_seq)
                seq_len = len(seq)
                if min_len <= seq_len <= max_len:
                    all_records.append((current_header, seq))
                    count += 1
                elif seq_len > max_len:
                    all_records.append((current_header, seq[:max_len]))
                    count += 1
        
        print(f"  {f.name}: {count} sequences")
    
    total_seqs = len(all_records)
    print(f"\nTotal sequences loaded: {total_seqs:,}")
    
    # Calculate sample size
    sample_size = int(total_seqs * sample_percent / 100)
    print(f"Sampling {sample_percent}% = {sample_size:,} sequences")
    
    # Random sample
    sampled = random.sample(all_records, sample_size)
    
    # Save to FASTA
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for header, seq in sampled:
            f.write(f"{header}\\n")
            # Write sequence in 80-character lines (standard FASTA)
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\\n")
    
    # Calculate tokens
    total_bases = sum(len(seq) for _, seq in sampled)
    
    print(f"Saved to: {output_path}")
    print(f"Sampled sequences: {sample_size:,}")
    print(f"Estimated tokens: {total_bases:,}")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample sequences from NCBI data to FASTA")
    parser.add_argument('--data_dir', default='ncbi_ftp_output', help='Directory with FASTA files')
    parser.add_argument('--sample_percent', type=float, required=True, help='Percentage to sample (0-100)')
    parser.add_argument('--output', required=True, help='Output FASTA file path')
    parser.add_argument('--min_len', type=int, default=64, help='Minimum sequence length')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    sample_to_fasta(args.data_dir, args.sample_percent, args.output, 
                    args.min_len, args.max_len, args.seed)
'''
    
    script_path = Path(experiment_dir) / "utils" / "sample_to_fasta.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"\nCreated sampling utility: {script_path}")
    return script_path


def create_preprocessing_script(experiment_dir, ncbi_data="ncbi_ftp_output"):
    """Create a script to pre-generate all data samples for Iso-Param."""
    script_content = f'''#!/bin/bash
# Pre-generate data samples for Iso-Param analysis

SAMPLES_DIR="{experiment_dir}/data_samples"
DATA_DIR="{ncbi_data}"

mkdir -p "$SAMPLES_DIR"

echo "Generating data samples for Iso-Param analysis..."
echo ""

# Sampling levels (percentage of full data)
LEVELS=(6.25 12.5 25 50 100)

for pct in "${{LEVELS[@]}}"; do
    echo "Generating ${{pct}}% sample..."
    python3 {experiment_dir}/utils/sample_to_fasta.py \\
        --data_dir "$DATA_DIR" \\
        --sample_percent "$pct" \\
        --output "$SAMPLES_DIR/sample_${{pct}}pct.fasta" \\
        --seed 42
done

echo ""
echo "All samples generated in: $SAMPLES_DIR"
echo ""
ls -lh "$SAMPLES_DIR"
'''
    
    script_path = Path(experiment_dir) / "prepare_data_samples.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Created preprocessing script: {script_path}")
    return script_path


def main():
    print("="*60)
    print("Setting Up Scaling Law Experiment")
    print("="*60)
    
    # Setup folders
    experiment_dir = setup_folder_structure("scaling_experiment")
    
    # Generate Iso-Token configs (varying model sizes)
    print("\n[Phase 1] Generating Iso-Token configs...")
    iso_token_configs = generate_iso_token_configs(
        experiment_dir / "configs" / "iso_token"
    )
    
    # Use 100M config as "best" for Iso-Param (conservative middle-ground)
    # In practice, Phase 1 will determine the actual best
    best_config = iso_token_configs[-2]['config']  # 100m model
    
    # Generate Iso-Param config
    print("\n[Phase 2] Generating Iso-Param config...")
    iso_param_config = generate_iso_param_configs(
        experiment_dir / "configs" / "iso_param",
        best_config
    )
    
    # Generate Iso-FLOP configs
    print("\n[Phase 3] Generating Iso-FLOP configs...")
    # Fixed compute based on 1M model with full data
    iso_flop_configs = generate_iso_flop_configs(
        experiment_dir / "configs" / "iso_flop"
    )
    
    # Create sampling utility (outputs FASTA directly)
    sampling_utility = create_sampling_utility(experiment_dir)
    
    # Create preprocessing script
    preprocessing_script = create_preprocessing_script(experiment_dir)
    
    # Create summary file
    summary = {
        "experiment_structure": {
            "iso_token": {
                "description": "Vary model size, constant full data (~3.95B tokens)",
                "configs": len(iso_token_configs),
                "model_sizes": [c['params'] for c in iso_token_configs]
            },
            "iso_param": {
                "description": "Constant model, vary data via random sampling",
                "base_model": "model_100m.json",
                "sampling_levels": [6.25, 12.5, 25, 50, 100],
                "data_samples_dir": "data_samples"
            },
            "iso_flop": {
                "description": "Fixed compute budget: 1M model with 100% data. Variable data per model (data_fraction = N_ref/N_model)",
                "reference_model": "1M params with 3.95B tokens",
                "target_flops": "~2.37e+16",
                "configs": len(iso_flop_configs),
                "models": [{"name": c["file"], "params": c["params"], "data_pct": c["data_percent"]} for c in iso_flop_configs]
            }
        },
        "total_tokens_available": 3_950_000_000,
        "chinchilla_optimal_params": 197_500_000,
        "utilities": {
            "sampling": str(sampling_utility),
            "preprocessing": str(preprocessing_script)
        }
    }
    
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Generate data samples:")
    print(f"   ./scaling_experiment/prepare_data_samples.sh")
    print(f"\n2. Run the orchestration script:")
    print(f"   ./run_scaling_orchestrator.sh")


if __name__ == '__main__':
    main()
