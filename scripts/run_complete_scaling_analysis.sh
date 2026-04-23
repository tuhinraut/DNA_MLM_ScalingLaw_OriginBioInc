#!/bin/bash
# Master orchestrator for the DNA-MLM scaling-law sweep.
#
# Pipeline:
#   0. Generate configs via setup_scaling_experiment.py (CLI-configurable).
#   1. Carve one fixed held-out eval FASTA (used by EVERY run).
#   2. Pre-sample per-fraction FASTAs for Iso-Param, and per-run FASTAs for Iso-FLOP.
#   3. Phase 1 Iso-Token  : full train data, sweep N.
#   4. Phase 2 Iso-Param  : fixed model, sweep D.
#   5. Phase 3 Iso-FLOP   : multiple compute budgets, each an N sweep.
#   6. Plots: scaling (loss vs N/D/C), iso-analysis, TTP.

set -euo pipefail

# -------- defaults (override via env vars) -----------------------------------
EXPERIMENT_DIR="${EXPERIMENT_DIR:-scaling_experiment}"
DATA_DIR="${DATA_DIR:-ncbi_ftp_output}"
EVAL_TOKENS="${EVAL_TOKENS:-50000000}"            # 50M held-out eval tokens
ISO_TOKEN_POINTS="${ISO_TOKEN_POINTS:-8}"
ISO_FLOP_POINTS="${ISO_FLOP_POINTS:-5}"
ISO_FLOP_BUDGETS="${ISO_FLOP_BUDGETS:-1e15,3e15,1e16,3e16}"
ISO_PARAM_MODEL_PARAMS="${ISO_PARAM_MODEL_PARAMS:-30000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

SKIP=""
CLEANUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)   SKIP="$SKIP setup" ;;
    --skip-split)   SKIP="$SKIP split" ;;
    --skip-samples) SKIP="$SKIP samples" ;;
    --skip-token)   SKIP="$SKIP token" ;;
    --skip-param)   SKIP="$SKIP param" ;;
    --skip-flop)    SKIP="$SKIP flop" ;;
    --skip-plots)   SKIP="$SKIP plots" ;;
    --cleanup)      CLEANUP=1 ;;
    --help)
      sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

skip() { [[ " $SKIP " == *" $1 "* ]]; }
info() { echo "[info] $*"; }
ok()   { echo "[ok]   $*"; }
err()  { echo "[err]  $*" >&2; }

# -------- prerequisite checks ------------------------------------------------
if [[ ! -f "src/train.py" ]]; then
  err "src/train.py not found (run this from the repo root)"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  err "data directory not found: $DATA_DIR"
  err "run: python data_downloaders/download_ncbi_ftp.py"
  exit 1
fi
info "data_dir=$DATA_DIR experiment_dir=$EXPERIMENT_DIR"

# -------- 0: setup -----------------------------------------------------------
if skip setup; then
  info "skip setup"
else
  info "phase 0: counting tokens in $DATA_DIR ..."
  RAW_TOKENS=$(python3 -c "
import pathlib
total = 0
for p in pathlib.Path('$DATA_DIR').glob('*_CDS.fasta'):
    with open(p) as f:
        for line in f:
            if not line.startswith('>'):
                total += len(line.strip())
print(total)
")
  info "raw token count: $RAW_TOKENS"
  info "phase 0: generate configs"
  python scripts/setup_scaling_experiment.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --data_total_tokens "$RAW_TOKENS" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --iso_token_points "$ISO_TOKEN_POINTS" \
    --iso_param_model "$ISO_PARAM_MODEL_PARAMS" \
    --iso_flop_budgets "$ISO_FLOP_BUDGETS" \
    --iso_flop_points_per_bucket "$ISO_FLOP_POINTS"
  ok "configs generated"
fi

SPLIT_DIR="$EXPERIMENT_DIR/data_samples/split"
EVAL_FASTA="$SPLIT_DIR/eval.fasta"
TRAIN_FASTA="$SPLIT_DIR/train_full.fasta"

# -------- 1: eval/train split -----------------------------------------------
if skip split; then
  info "skip split"
else
  info "phase 1: carve fixed held-out eval set ($EVAL_TOKENS tokens)"
  python "$EXPERIMENT_DIR/utils/carve_eval_split.py" \
    --data_dir "$DATA_DIR" \
    --eval_tokens "$EVAL_TOKENS" \
    --train_out "$TRAIN_FASTA" \
    --eval_out "$EVAL_FASTA" \
    --max_len "$MAX_SEQ_LEN" --seed "$SEED"
  ok "split carved"
fi

SAMPLE_DIR="$EXPERIMENT_DIR/data_samples/iso_param"
# -------- 2: iso-param data samples (token-budgeted) ------------------------
if skip samples; then
  info "skip samples"
else
  info "phase 2: pre-sample iso-param FASTAs (by token budget)"
  mkdir -p "$SAMPLE_DIR"
  # Fractions of what remains in the train pool AFTER the eval split.
  TRAIN_POOL_TOKENS=$(python3 -c "
import sys
total = 0
with open('$TRAIN_FASTA') as f:
    for line in f:
        if not line.startswith('>'):
            total += len(line.strip())
print(total)
")
  info "train pool has $TRAIN_POOL_TOKENS tokens"
  for FRAC in 0.0625 0.125 0.25 0.5 1.0; do
    TARGET=$(python3 -c "print(int($TRAIN_POOL_TOKENS * $FRAC))")
    LABEL=$(python3 -c "print(f'{$FRAC*100:g}pct'.replace('.','p'))")
    OUT="$SAMPLE_DIR/sample_${LABEL}.fasta"
    if [[ -f "$OUT" ]]; then
      info "  exists: $OUT"
      continue
    fi
    python "$EXPERIMENT_DIR/utils/sample_to_fasta.py" \
      --data_dir "$SPLIT_DIR" \
      --target_tokens "$TARGET" \
      --output "$OUT" \
      --pattern "train_full.fasta" \
      --max_len "$MAX_SEQ_LEN" --seed "$SEED"
  done
  ok "iso-param samples ready"
fi

# -------- helper: run training ----------------------------------------------
CKPT_BASE="$EXPERIMENT_DIR/checkpoints"
LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

train_one() {
  # args: config, run_name, train_fasta, phase_dir
  local CONFIG="$1" NAME="$2" TRAIN="$3" PHASE="$4"
  local CKPT_DIR="$CKPT_BASE/$PHASE"
  mkdir -p "$CKPT_DIR"
  if [[ -f "$LOG_DIR/training_log_${NAME}.json" ]]; then
    info "  [$NAME] already done, skipping"
    return 0
  fi
  python src/train.py \
    --config "$CONFIG" \
    --run_name "$NAME" \
    --data_path "$TRAIN" \
    --eval_data_path "$EVAL_FASTA" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --num_epochs 1 \
    --eval_every "$EVAL_EVERY" \
    --seed "$SEED" \
    --save_dir "$CKPT_DIR" \
    --log_dir "$LOG_DIR"
}

# -------- 3: Iso-Token -------------------------------------------------------
if skip token; then
  info "skip phase 3 (iso-token)"
else
  info "phase 3: Iso-Token (N sweep, fixed D)"
  for CFG in "$EXPERIMENT_DIR"/configs/iso_token/*.json; do
    NAME="iso_token_$(basename "$CFG" .json)"
    train_one "$CFG" "$NAME" "$TRAIN_FASTA" "iso_token"
  done
  ok "iso-token done"
fi

# -------- 4: Iso-Param -------------------------------------------------------
if skip param; then
  info "skip phase 4 (iso-param)"
else
  info "phase 4: Iso-Param (D sweep, fixed N)"
  for CFG in "$EXPERIMENT_DIR"/configs/iso_param/*.json; do
    MODEL_NAME=$(basename "$CFG" .json)
    for SAMPLE in "$SAMPLE_DIR"/sample_*.fasta; do
      LABEL=$(basename "$SAMPLE" .fasta | sed 's/sample_//')
      NAME="iso_param_${MODEL_NAME}_${LABEL}"
      train_one "$CFG" "$NAME" "$SAMPLE" "iso_param"
    done
  done
  ok "iso-param done"
fi

# -------- 5: Iso-FLOP --------------------------------------------------------
if skip flop; then
  info "skip phase 5 (iso-flop)"
else
  info "phase 5: Iso-FLOP (multi-budget, N sweep per budget)"
  ISO_FLOP_DATA_DIR="$EXPERIMENT_DIR/data_samples/iso_flop"
  mkdir -p "$ISO_FLOP_DATA_DIR"
  for BUCKET in "$EXPERIMENT_DIR"/configs/iso_flop/*/; do
    for CFG in "$BUCKET"*.json; do
      [[ -f "$CFG" ]] || continue
      NAME="iso_flop_$(basename "$CFG" .json)"
      # required_tokens is stored in the config's _meta
      TARGET=$(python3 -c "
import json
print(json.load(open('$CFG'))['_meta']['required_tokens'])
")
      SAMPLE="$ISO_FLOP_DATA_DIR/${NAME}.fasta"
      if [[ ! -f "$SAMPLE" ]]; then
        python "$EXPERIMENT_DIR/utils/sample_to_fasta.py" \
          --data_dir "$SPLIT_DIR" \
          --target_tokens "$TARGET" \
          --output "$SAMPLE" \
          --pattern "train_full.fasta" \
          --max_len "$MAX_SEQ_LEN" --seed "$SEED"
      fi
      train_one "$CFG" "$NAME" "$SAMPLE" "iso_flop"
    done
  done
  ok "iso-flop done"
fi

# -------- 6: plots -----------------------------------------------------------
PLOT_DIR="$EXPERIMENT_DIR/plots"
mkdir -p "$PLOT_DIR"
if skip plots; then
  info "skip plots"
else
  info "phase 6: plots"
  python scripts/plot_scaling.py      --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"
  python scripts/plot_iso_analysis.py --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"
  python scripts/plot_ttp.py          --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"
  ok "plots written to $PLOT_DIR"
fi

# -------- cleanup ------------------------------------------------------------
if [[ "$CLEANUP" -eq 1 ]]; then
  info "cleanup: removing data_samples and checkpoints"
  rm -rf "$EXPERIMENT_DIR/data_samples" "$CKPT_BASE"
fi

echo
echo "all done. logs in $LOG_DIR, plots in $PLOT_DIR"
