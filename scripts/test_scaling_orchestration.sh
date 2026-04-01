#!/bin/bash
# Small-Scale Test Orchestration for DNA MLM Scaling Laws
# This script runs a quick local test using synthetic data
# Perfect for testing the pipeline before full-scale runs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - SMALL SCALE for local testing
TEST_DIR="test_experiment"
LOG_DIR="$TEST_DIR/logs"
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
CONFIG_DIR="$TEST_DIR/configs"
PLOT_DIR="$TEST_DIR/plots"
RESULTS_DIR="$TEST_DIR/results"

# Small model configs for quick testing (100K to 4M params)
# Quick training parameters
NUM_SYNTHETIC=1000        # Small synthetic dataset
MAX_SEQ_LEN=256           # Shorter sequences for speed
BATCH_SIZE=16             # Smaller batches
EVAL_EVERY=50             # Log frequently for detailed curves
NUM_EPOCHS=2              # Short training
LEARNING_RATE=1e-4
MASK_PROB=0.15
SEED=42

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
# Setup
# ============================================
setup_test() {
    log_info "Setting up test environment..."
    
    rm -rf "$TEST_DIR"
    mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$CONFIG_DIR"/{iso_token,iso_param,iso_flop} "$PLOT_DIR" "$RESULTS_DIR"
    
    # Generate small model configs
    python3 << 'EOF'
import json
import os

config_dir = "test_experiment/configs"

# Small models for quick testing (100K - 4M params)
configs = [
    (64, 2, 2, 256, "model_100k"),      # ~100K
    (128, 4, 2, 512, "model_400k"),    # ~400K
    (192, 3, 4, 768, "model_1m"),      # ~1.8M
    (256, 4, 4, 1024, "model_4m"),      # ~3.2M
]

for d_model, n_heads, n_layers, d_ff, name in configs:
    config = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": 0.1,
        "vocab_size": 7,
        "max_seq_len": 256
    }
    # Save to iso_token
    with open(f"{config_dir}/iso_token/{name}.json", 'w') as f:
        json.dump(config, f, indent=2)
    # Also save to iso_flop for phase 3
    with open(f"{config_dir}/iso_flop/{name}.json", 'w') as f:
        json.dump(config, f, indent=2)

# Iso-param config (use 1M model as base)
best_config = {
    "d_model": 192,
    "n_heads": 3,
    "n_layers": 4,
    "d_ff": 768,
    "dropout": 0.1,
    "vocab_size": 7,
    "max_seq_len": 256
}
with open(f"{config_dir}/iso_param/model_1m.json", 'w') as f:
    json.dump(best_config, f, indent=2)

print("Generated test configs:")
print("  - 4 Iso-Token configs (100K to 4M params)")
print("  - 1 Iso-Param config (1M params)")
print("  - 4 Iso-FLOP configs (100K to 4M params)")
EOF
    
    log_success "Test environment ready"
}

# ============================================
# PHASE 1: Iso-Token (Find Optimal Model Size)
# ============================================
run_iso_token() {
    log_info "========================================"
    log_info "PHASE 1: Iso-Token Analysis (TEST SCALE)"
    log_info "Training 4 small models on synthetic data"
    log_info "========================================"
    
    local best_model=""
    local best_loss=999999
    
    for config_file in "$CONFIG_DIR/iso_token"/*.json; do
        local model_name=$(basename "$config_file" .json)
        local run_name="iso_token_${model_name}"
        
        log_info "Training $model_name..."
        
        if python src/train.py \
            --config "$config_file" \
            --num_synthetic $NUM_SYNTHETIC \
            --max_seq_len $MAX_SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --mask_prob $MASK_PROB \
            --num_epochs $NUM_EPOCHS \
            --eval_every $EVAL_EVERY \
            --seed $SEED \
            --run_name "$run_name" \
            --save_dir "$CHECKPOINT_DIR/iso_token" \
            --log_dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/${run_name}.stdout"; then
            
            local log_file="$LOG_DIR/training_log_${run_name}.json"
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
                
                if (( $(echo "$final_loss < $best_loss" | bc -l 2>/dev/null || echo 0) )); then
                    best_loss=$final_loss
                    best_model=$model_name
                    log_info "  -> New best model: $best_model"
                fi
            fi
        else
            log_error "$model_name failed"
        fi
    done
    
    # Save results
    cat > "$RESULTS_DIR/phase1_iso_token.json" << EOF
{
  "phase": "iso_token",
  "best_model": "$best_model",
  "best_loss": $best_loss,
  "note": "Best model from Phase 1 (TEST SCALE)"
}
EOF
    
    export ISO_TOKEN_BEST_MODEL="$best_model"
    log_success "Phase 1 complete. Best model: $best_model"
}

# ============================================
# PHASE 2: Iso-Param (Find Ideal Token Count)
# ============================================
run_iso_param() {
    log_info "========================================"
    log_info "PHASE 2: Iso-Param Analysis (TEST SCALE)"
    log_info "Testing optimal token count"
    log_info "========================================"
    
    local model_name="${ISO_TOKEN_BEST_MODEL:-model_1m}"
    local config_file="$CONFIG_DIR/iso_param/$model_name.json"
    
    # Test different data sizes by varying num_synthetic (as % of full dataset)
    # 10%, 25%, 50%, 75%, 100% of NUM_SYNTHETIC (1000)
    local sample_sizes=(100 250 500 750 1000)
    local sample_labels=("10pct" "25pct" "50pct" "75pct" "100pct")
    
    for i in "${!sample_sizes[@]}"; do
        local num_seqs="${sample_sizes[$i]}"
        local label="${sample_labels[$i]}"
        local run_name="iso_param_${model_name}_${label}"
        
        log_info "Training on $label of data ($num_seqs synthetic sequences)..."
        
        if python src/train.py \
            --config "$config_file" \
            --num_synthetic $num_seqs \
            --max_seq_len $MAX_SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --mask_prob $MASK_PROB \
            --num_epochs $NUM_EPOCHS \
            --eval_every $EVAL_EVERY \
            --seed $SEED \
            --run_name "$run_name" \
            --save_dir "$CHECKPOINT_DIR/iso_param" \
            --log_dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/${run_name}.stdout"; then
            
            local log_file="$LOG_DIR/training_log_${run_name}.json"
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
                log_success "$label ($num_seqs seqs): Loss=$final_loss, Tokens=$tokens"
            fi
        else
            log_error "$label ($num_seqs seqs) failed"
        fi
    done
    
    log_success "Phase 2 complete"
}

# ============================================
# PHASE 3: Iso-FLOP (Find Optimal N/D Allocation)
# ============================================
run_iso_flop() {
    log_info "========================================"
    log_info "PHASE 3: Iso-FLOP Analysis (TEST SCALE)"
    log_info "Finding optimal N/D allocation"
    log_info "========================================"
    
    # Use different epoch counts to achieve different effective token counts
    local epoch_counts=(1 2 4)
    
    for config_file in "$CONFIG_DIR/iso_flop"/*.json; do
        local model_name=$(basename "$config_file" .json)
        
        for epochs in "${epoch_counts[@]}"; do
            local run_name="iso_flop_${model_name}_e${epochs}"
            
            log_info "Training $model_name for $epochs epochs..."
            
            if python src/train.py \
                --config "$config_file" \
                --num_synthetic $NUM_SYNTHETIC \
                --max_seq_len $MAX_SEQ_LEN \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --mask_prob $MASK_PROB \
                --num_epochs $epochs \
                --eval_every $EVAL_EVERY \
                --seed $SEED \
                --run_name "$run_name" \
                --save_dir "$CHECKPOINT_DIR/iso_flop" \
                --log_dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/${run_name}.stdout"; then
                
                local log_file="$LOG_DIR/training_log_${run_name}.json"
                if [ -f "$log_file" ]; then
                    local final_loss=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data['log'][-1]['eval_loss'] if data['log'] else 999)
")
                    local flops=$(python3 -c "
import json
with open('$log_file') as f:
    data = json.load(f)
    print(data.get('final_flops', 0))
")
                    log_success "$model_name (e$epochs): Loss=$final_loss, FLOPs=$flops"
                fi
            else
                log_error "$model_name (e$epochs) failed"
            fi
        done
    done
    
    log_success "Phase 3 complete"
}

# ============================================
# Generate Plots
# ============================================
generate_plots() {
    log_info "========================================"
    log_info "Generating Plots"
    log_info "========================================"
    
    log_info "Generating scaling law plots..."
    python scripts/plot_scaling.py \
        --log_dir "$LOG_DIR" \
        --output_dir "$PLOT_DIR"
    
    log_info "Generating Chinchilla-style IsoFLOP plots..."
    python scripts/plot_ttp.py \
        --log_dir "$LOG_DIR" \
        --output_dir "$PLOT_DIR" \
        --style chinchilla
    
    log_success "Plots saved to $PLOT_DIR/"
    ls -lh "$PLOT_DIR/"
}

# ============================================
# Summary
# ============================================
show_summary() {
    log_info "========================================"
    log_info "TEST RUN COMPLETE"
    log_info "========================================"
    echo ""
    echo "Results:"
    ls -lh "$RESULTS_DIR/" 2>/dev/null || echo "  (No result files)"
    echo ""
    echo "Logs: $LOG_DIR/"
    echo "  Training logs: $(ls -1 "$LOG_DIR/"/*.json 2>/dev/null | wc -l) files"
    echo ""
    echo "Plots: $PLOT_DIR/"
    ls -1 "$PLOT_DIR/"/*.png 2>/dev/null | while read f; do
        echo "  - $(basename $f)"
    done
    echo ""
    echo "To clean up test files:"
    echo "  rm -rf $TEST_DIR"
    echo ""
    echo "To run full-scale experiment:"
    echo "  ./scripts/run_complete_scaling_analysis.sh"
}

# ============================================
# Main
# ============================================
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║    SMALL-SCALE TEST FOR DNA MLM SCALING LAWS             ║"
    echo "║    Quick Local Test with Synthetic Data                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check prerequisites
    if ! python -c "import torch; import numpy; import matplotlib" 2>/dev/null; then
        log_error "Missing dependencies. Install with:"
        log_error "  pip install torch numpy matplotlib"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
    log_info "Device: $(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
    
    # Run all phases
    setup_test
    run_iso_token
    run_iso_param
    run_iso_flop
    generate_plots
    
    # Show final summary
    show_summary
    
    log_success "All test phases complete!"
}

# Run
main "$@"
