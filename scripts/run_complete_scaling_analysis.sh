#!/bin/bash
# Complete Scaling Law Analysis - Master Orchestrator
# One command to run everything: setup → samples → all 3 phases

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_DIR="scaling_experiment"
NCBI_DATA="ncbi_ftp_output"
CLEANUP_AFTER=false
SKIP_PHASES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP_AFTER=true
            shift
            ;;
        --skip-setup)
            SKIP_PHASES="setup $SKIP_PHASES"
            shift
            ;;
        --skip-samples)
            SKIP_PHASES="samples $SKIP_PHASES"
            shift
            ;;
        --skip-token)
            SKIP_PHASES="token $SKIP_PHASES"
            shift
            ;;
        --skip-param)
            SKIP_PHASES="param $SKIP_PHASES"
            shift
            ;;
        --skip-flop)
            SKIP_PHASES="flop $SKIP_PHASES"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cleanup       Delete data_samples and checkpoints after completion"
            echo "  --skip-setup    Skip experiment setup (configs already generated)"
            echo "  --skip-samples  Skip data sampling (samples already exist)"
            echo "  --skip-token    Skip Phase 1 (Iso-Token)"
            echo "  --skip-param    Skip Phase 2 (Iso-Param)"
            echo "  --skip-flop     Skip Phase 3 (Iso-FLOP)"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run everything"
            echo "  $0 --cleanup          # Run everything and cleanup after"
            echo "  $0 --skip-setup       # Skip setup, run from existing configs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# PHASE 0: Setup
# ============================================
run_setup() {
    if [[ "$SKIP_PHASES" == *"setup"* ]]; then
        log_info "Skipping setup (--skip-setup specified)"
        return 0
    fi

    log_info "Phase 0: Setting up experiment..."
    
    if [ -d "$EXPERIMENT_DIR" ]; then
        log_warning "Experiment directory exists. Regenerating configs..."
        rm -rf "$EXPERIMENT_DIR"
    fi
    
    python3 scripts/setup_scaling_experiment.py
    
    if [ ! -d "$EXPERIMENT_DIR" ]; then
        log_error "Setup failed - experiment directory not created"
        exit 1
    fi
    
    log_success "Setup complete"
}

# ============================================
# PHASE 0.5: Data Sampling
# ============================================
run_sampling() {
    if [[ "$SKIP_PHASES" == *"samples"* ]]; then
        log_info "Skipping data sampling (--skip-samples specified)"
        return 0
    fi

    log_info "Phase 0.5: Generating data samples..."
    
    # Check if samples already exist
    if [ -f "$EXPERIMENT_DIR/data_samples/sample_100pct.fasta" ]; then
        log_warning "Data samples already exist. Use --skip-samples to skip this phase."
        read -p "Regenerate samples? (y/N): " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Using existing samples"
            return 0
        fi
        rm -rf "$EXPERIMENT_DIR/data_samples/"/*.fasta
    fi
    
    # Run sampling script
    if [ -f "$EXPERIMENT_DIR/prepare_data_samples.sh" ]; then
        cd "$EXPERIMENT_DIR"
        bash prepare_data_samples.sh
        cd ..
    else
        log_error "Sampling script not found. Run setup first."
        exit 1
    fi
    
    log_success "Data sampling complete"
}

# ============================================
# PHASE 1: Iso-Token
# ============================================
run_iso_token() {
    if [[ "$SKIP_PHASES" == *"token"* ]]; then
        log_info "Skipping Phase 1 (Iso-Token)"
        return 0
    fi

    log_info "========================================"
    log_info "PHASE 1: Iso-Token Analysis"
    log_info "Training 11 models on full dataset"
    log_info "========================================"
    
    CONFIG_DIR="$EXPERIMENT_DIR/configs/iso_token"
    PHASE_DIR="$EXPERIMENT_DIR/checkpoints/iso_token"
    LOG_DIR="$EXPERIMENT_DIR/logs"
    
    mkdir -p "$PHASE_DIR" "$LOG_DIR"
    
    # Get all NCBI data
    local all_data=$(find "$NCBI_DATA" -name "*_CDS.fasta" | tr '\n' ' ')
    
    local best_model=""
    local best_loss=999999
    local results_file="$EXPERIMENT_DIR/results/phase1_iso_token.json"
    
    log_info "Found $(echo $all_data | wc -w) data files"
    
    for config_file in "$CONFIG_DIR"/*.json; do
        local model_name=$(basename "$config_file" .json)
        local run_name="iso_token_${model_name}"
        local log_file="$LOG_DIR/training_log_${run_name}.json"
        
        log_info "Training $model_name..."
        
        if python src/train.py \
            --config "$config_file" \
            --data_path $all_data \
            --max_seq_len 2048 \
            --min_seq_len 64 \
            --batch_size 32 \
            --learning_rate 1e-4 \
            --mask_prob 0.15 \
            --num_epochs 1 \
            --eval_every 500 \
            --seed 42 \
            --run_name "$run_name" \
            --save_dir "$PHASE_DIR" \
            --log_dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/${run_name}.stdout"; then
            
            if [ -f "$log_file" ]; then
                local final_loss=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['log'][-1]['eval_loss'] if data['log'] else 999)
")
                local params=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['num_parameters'])
")
                
                log_success "$model_name: Loss=$final_loss, Params=$params"
                
                # Track best
                if (( $(echo "$final_loss < $best_loss" | bc -l 2>/dev/null || echo 0) )); then
                    best_loss=$final_loss
                    best_model=$model_name
                    log_info "  → New best model: $best_model"
                fi
            fi
        else
            log_error "$model_name failed"
        fi
    done
    
    # Save results
    python3 -c "
import json
result = {
    'phase': 'iso_token',
    'best_model': '$best_model',
    'best_loss': $best_loss,
    'note': 'Best model for Phase 2'
}
with open('$results_file', 'w') as f:
    json.dump(result, f, indent=2)
"
    
    log_success "Phase 1 complete. Best model: $best_model"
    
    # Export for next phases
    export ISO_TOKEN_BEST_MODEL="$best_model"
    export ISO_TOKEN_BEST_CONFIG="$EXPERIMENT_DIR/configs/iso_token/$best_model.json"
}

# ============================================
# PHASE 2: Iso-Param
# ============================================
run_iso_param() {
    if [[ "$SKIP_PHASES" == *"param"* ]]; then
        log_info "Skipping Phase 2 (Iso-Param)"
        return 0
    fi

    log_info "========================================"
    log_info "PHASE 2: Iso-Param Analysis"
    log_info "Testing optimal token count"
    log_info "========================================"
    
    # Determine which model to use
    if [ -z "$ISO_TOKEN_BEST_MODEL" ]; then
        # Try to load from Phase 1 results
        local p1_results="$EXPERIMENT_DIR/results/phase1_iso_token.json"
        if [ -f "$p1_results" ]; then
            ISO_TOKEN_BEST_MODEL=$(python3 -c "
import json
with open('$p1_results') as f:
    data = json.load(f)
    print(data.get('best_model', 'model_100m'))
")
            ISO_TOKEN_BEST_CONFIG="$EXPERIMENT_DIR/configs/iso_token/$ISO_TOKEN_BEST_MODEL.json"
            log_info "Using best model from Phase 1: $ISO_TOKEN_BEST_MODEL"
        else
            # Default fallback
            ISO_TOKEN_BEST_MODEL="model_100m"
            ISO_TOKEN_BEST_CONFIG="$EXPERIMENT_DIR/configs/iso_token/model_100m.json"
            log_warning "Phase 1 results not found, using default: $ISO_TOKEN_BEST_MODEL"
        fi
    fi
    
    local phase_dir="$EXPERIMENT_DIR/checkpoints/iso_param"
    local log_dir="$EXPERIMENT_DIR/logs"
    local results_file="$EXPERIMENT_DIR/results/phase2_iso_param.json"
    
    mkdir -p "$phase_dir"
    
    log_info "Testing $ISO_TOKEN_BEST_MODEL on different data sizes..."
    
    for sample_file in "$EXPERIMENT_DIR/data_samples"/sample_*.fasta; do
        local sample_name=$(basename "$sample_file" .fasta)
        local run_name="iso_param_${ISO_TOKEN_BEST_MODEL}_${sample_name}"
        local log_file="$log_dir/training_log_${run_name}.json"
        
        log_info "Training on $sample_name..."
        
        if python src/train.py \
            --config "$ISO_TOKEN_BEST_CONFIG" \
            --data_path "$sample_file" \
            --max_seq_len 2048 \
            --min_seq_len 64 \
            --batch_size 32 \
            --learning_rate 1e-4 \
            --mask_prob 0.15 \
            --num_epochs 1 \
            --eval_every 500 \
            --seed 42 \
            --run_name "$run_name" \
            --save_dir "$phase_dir" \
            --log_dir "$log_dir" 2>&1 | tee "$log_dir/${run_name}.stdout"; then
            
            if [ -f "$log_file" ]; then
                local final_loss=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['log'][-1]['eval_loss'] if data['log'] else 999)
")
                local tokens=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['final_tokens_seen'])
")
                log_success "$sample_name: Loss=$final_loss, Tokens=$tokens"
            fi
        else
            log_error "$sample_name failed"
        fi
    done
    
    # Save results summary
    python3 -c "
import json
import glob

results = []
for log_file in glob.glob('$log_dir/training_log_iso_param_*.json'):
    with open(log_file) as f:
        data = json.load(f)
        if data.get('log'):
            results.append({
                'model': '$ISO_TOKEN_BEST_MODEL',
                'tokens': data['final_tokens_seen'],
                'loss': data['log'][-1]['eval_loss']
            })

result = {'phase': 'iso_param', 'model': '$ISO_TOKEN_BEST_MODEL', 'results': results}
with open('$results_file', 'w') as f:
    json.dump(result, f, indent=2)
" 2>/dev/null || true
    
    log_success "Phase 2 complete"
}

# ============================================
# PHASE 3: Iso-FLOP
# ============================================
run_iso_flop() {
    if [[ "$SKIP_PHASES" == *"flop"* ]]; then
        log_info "Skipping Phase 3 (Iso-FLOP)"
        return 0
    fi

    log_info "========================================"
    log_info "PHASE 3: Iso-FLOP Analysis"
    log_info "Fixed compute: variable data per model"
    log_info "========================================"
    
    local phase_dir="$EXPERIMENT_DIR/checkpoints/iso_flop"
    local log_dir="$EXPERIMENT_DIR/logs"
    local samples_dir="$EXPERIMENT_DIR/data_samples/iso_flop"
    local results_file="$EXPERIMENT_DIR/results/phase3_iso_flop.json"
    
    mkdir -p "$phase_dir" "$samples_dir"
    
    # Get all NCBI data files
    local all_data=$(find "$NCBI_DATA" -name "*_CDS.fasta" | tr '\n' ' ')
    
    log_info "Testing models with data sampled for fixed compute budget..."
    
    for config_file in "$EXPERIMENT_DIR/configs/iso_flop"/*.json; do
        local model_name=$(basename "$config_file" .json)
        
        # Extract config metadata
        local data_fraction=$(python3 -c "
import json
with open('$config_file') as f:
    data = json.load(f)
    print(data.get('_meta', {}).get('data_fraction', 1.0))
")
        local data_pct=$(python3 -c "
import json
with open('$config_file') as f:
    data = json.load(f)
    print(data.get('_meta', {}).get('data_percent', 100))
")
        local params=$(python3 -c "
import json
with open('$config_file') as f:
    data = json.load(f)
    print(data.get('_meta', {}).get('params', 0))
")
        local target_flops=$(python3 -c "
import json
with open('$config_file') as f:
    data = json.load(f)
    print(data.get('_meta', {}).get('target_flops', 0))
")
        
        local run_name="${model_name}"
        local sample_file="$samples_dir/${model_name}_sample.fasta"
        local log_file="$log_dir/training_log_${run_name}.json"
        
        # Create sampled data if it doesn't exist
        if [ ! -f "$sample_file" ]; then
            log_info "Creating ${data_pct}% data sample for $model_name..."
            
            python3 << EOF
import random
from pathlib import Path
import sys

random.seed(42)  # Reproducible sampling

# Get all input files
input_files = "$all_data".split()
data_fraction = float("$data_fraction")
sample_file = "$sample_file"

# Collect all sequences with their headers
all_sequences = []
for filepath in input_files:
    filepath = Path(filepath)
    if not filepath.exists():
        continue
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse FASTA
    entries = content.split('>')[1:]  # Skip empty first split
    for entry in entries:
        if '\n' in entry:
            header, seq = entry.split('\n', 1)
            seq = seq.replace('\n', '').strip()
            if len(seq) >= 64:  # Minimum length filter
                all_sequences.append((header, seq))

# Sample
num_to_sample = max(1, int(len(all_sequences) * data_fraction))
sampled = random.sample(all_sequences, min(num_to_sample, len(all_sequences)))

# Write output
Path(sample_file).parent.mkdir(parents=True, exist_ok=True)
with open(sample_file, 'w') as f:
    for header, seq in sampled:
        f.write(f">{header}\n{seq}\n")

print(f"Sampled {len(sampled)}/{len(all_sequences)} sequences ({data_fraction*100:.1f}%)")
EOF
        fi
        
        log_info "Training $model_name (${params} params) on ${data_pct}% data..."
        log_info "Target FLOPs: $(printf '%.2e' $target_flops)"
        
        if python src/train.py \
            --config "$config_file" \
            --data_path "$sample_file" \
            --max_seq_len 2048 \
            --min_seq_len 64 \
            --batch_size 32 \
            --learning_rate 1e-4 \
            --mask_prob 0.15 \
            --num_epochs 1 \
            --eval_every 500 \
            --seed 42 \
            --run_name "$run_name" \
            --save_dir "$phase_dir" \
            --log_dir "$log_dir" 2>&1 | tee "$log_dir/${run_name}.stdout"; then
            
            if [ -f "$log_file" ]; then
                local final_loss=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['log'][-1]['eval_loss'] if data['log'] else 999)
")
                local actual_flops=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data.get('final_flops', 0))
")
                local actual_tokens=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data.get('final_tokens_seen', 0))
")
                log_success "$model_name: Loss=$final_loss, FLOPs=$actual_flops, Tokens=$actual_tokens"
            fi
        else
            log_error "$model_name failed"
        fi
    done
    
    # Generate summary
    log_info "Generating Iso-FLOP summary..."
    python3 << EOF
import json
import glob
from pathlib import Path

log_dir = "$log_dir"
results_file = "$results_file"

results = []
for log_file in glob.glob(f'{log_dir}/training_log_iso_flop_*.json'):
    with open(log_file) as f:
        data = json.load(f)
    
    if data.get('log'):
        meta = data.get('_meta', {})
        results.append({
            'model': data.get('run_name', ''),
            'params': meta.get('params', 0),
            'data_fraction': meta.get('data_fraction', 1.0),
            'target_flops': meta.get('target_flops', 0),
            'actual_flops': data.get('final_flops', 0),
            'tokens': data.get('final_tokens_seen', 0),
            'loss': data['log'][-1]['eval_loss']
        })

Path(results_file).parent.mkdir(parents=True, exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {results_file}")
print(f"Models tested: {len(results)}")
for r in sorted(results, key=lambda x: x['params']):
    print(f"  {r['model']}: {r['params']/1e6:.1f}M params, {r['data_fraction']*100:.1f}% data, loss={r['loss']:.4f}")
EOF
    
    log_success "Phase 3 complete"
}

# ============================================
# Cleanup
# ============================================
run_cleanup() {
    if [ "$CLEANUP_AFTER" = false ]; then
        return 0
    fi

    log_info "========================================"
    log_info "Cleaning up..."
    log_info "========================================"
    
    # Calculate space to be freed
    local sample_size=$(du -sh "$EXPERIMENT_DIR/data_samples/" 2>/dev/null | cut -f1 || echo "0")
    local checkpoint_size=$(du -sh "$EXPERIMENT_DIR/checkpoints/" 2>/dev/null | cut -f1 || echo "0")
    
    log_info "Will free: ~$sample_size (samples) + ~$checkpoint_size (checkpoints)"
    
    read -p "Proceed with cleanup? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$EXPERIMENT_DIR/data_samples/"/*.fasta
        rm -rf "$EXPERIMENT_DIR/checkpoints/"
        log_success "Cleanup complete. Kept logs/ and results/ for analysis."
    else
        log_info "Cleanup skipped"
    fi
}

# ============================================
# Summary
# ============================================
show_summary() {
    log_info "========================================"
    log_info "SCALING LAW ANALYSIS COMPLETE"
    log_info "========================================"
    echo ""
    echo "Results available in: $EXPERIMENT_DIR/results/"
    ls -lh "$EXPERIMENT_DIR/results/" 2>/dev/null || echo "  (No result files found)"
    echo ""
    echo "Logs available in: $EXPERIMENT_DIR/logs/"
    echo "  Training logs: $(ls -1 "$EXPERIMENT_DIR/logs/"/*.json 2>/dev/null | wc -l) files"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in $EXPERIMENT_DIR/results/"
    echo "  2. Analyze with plotting scripts"
    echo "  3. Optionally cleanup: rm -rf $EXPERIMENT_DIR/data_samples/ $EXPERIMENT_DIR/checkpoints/"
    echo ""
    
    # Show disk usage
    echo "Disk usage:"
    du -sh "$EXPERIMENT_DIR/" 2>/dev/null || true
}

# ============================================
# Main
# ============================================
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║    COMPLETE SCALING LAW ANALYSIS FOR DNA MLM             ║"
    echo "║    Automatic Pipeline - All 3 Phases                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check prerequisites
    if [ ! -d "$NCBI_DATA" ]; then
        log_error "NCBI data directory not found: $NCBI_DATA"
        log_info "Please download data first using download_ncbi_ftp.py"
        exit 1
    fi
    
    if [ ! -f "train.py" ]; then
        log_error "train.py not found in current directory"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
    log_info "NCBI data: $(du -sh $NCBI_DATA | cut -f1)"
    
    # Run all phases
    run_setup
    run_sampling
    run_iso_token
    run_iso_param
    run_iso_flop
    
    # Show final summary
    show_summary
    
    # Optional cleanup
    run_cleanup
    
    log_success "All phases complete!"
}

# Run
main "$@"
