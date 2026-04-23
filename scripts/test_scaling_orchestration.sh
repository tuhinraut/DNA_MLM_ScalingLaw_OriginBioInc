#!/bin/bash
# Quick local sanity check of the pipeline using synthetic DNA data.
# Trains a handful of tiny models, then produces the plots.
# Always uses --num_epochs 1 (the spec's hard constraint).

set -euo pipefail

TEST_DIR="${TEST_DIR:-test_experiment}"
CONFIG_DIR="$TEST_DIR/configs"
LOG_DIR="$TEST_DIR/logs"
CKPT_DIR="$TEST_DIR/checkpoints"
PLOT_DIR="$TEST_DIR/plots"

NUM_SYNTHETIC="${NUM_SYNTHETIC:-2000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

rm -rf "$TEST_DIR"
mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$CKPT_DIR" "$PLOT_DIR"

# --- configs ----------------------------------------------------------------
python3 - <<PY
import json, os
os.makedirs("$CONFIG_DIR", exist_ok=True)
archs = [
    ("100k", dict(d_model=64,  n_heads=2, n_layers=2, d_ff=256)),
    ("400k", dict(d_model=128, n_heads=4, n_layers=2, d_ff=512)),
    ("1m",   dict(d_model=192, n_heads=3, n_layers=4, d_ff=768)),
    ("4m",   dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024)),
]
for name, cfg in archs:
    cfg.update(dropout=0.1, vocab_size=7, max_seq_len=$MAX_SEQ_LEN)
    with open(f"$CONFIG_DIR/model_{name}.json", "w") as f:
        json.dump(cfg, f, indent=2)
print(f"wrote {len(archs)} tiny configs")
PY

run_train() {
  local CFG="$1" NAME="$2" NSYNTH="$3"
  python src/train.py \
    --config "$CFG" --run_name "$NAME" \
    --num_synthetic "$NSYNTH" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --num_epochs 1 \
    --eval_every "$EVAL_EVERY" \
    --seed "$SEED" \
    --save_dir "$CKPT_DIR" \
    --log_dir "$LOG_DIR"
}

# --- phase 1: iso-token (vary N, fixed synthetic D) -------------------------
echo "[info] test-phase 1: iso-token"
for CFG in "$CONFIG_DIR"/*.json; do
  NAME="iso_token_$(basename "$CFG" .json)"
  run_train "$CFG" "$NAME" "$NUM_SYNTHETIC"
done

# --- phase 2: iso-param (fixed N=1m, vary D) --------------------------------
echo "[info] test-phase 2: iso-param"
for N in 200 500 1000 2000; do
  NAME="iso_param_model_1m_${N}seqs"
  run_train "$CONFIG_DIR/model_1m.json" "$NAME" "$N"
done

# --- phase 3: iso-flop (vary N across 2 budgets by varying synthetic count) -
echo "[info] test-phase 3: iso-flop"
# Budget 1 is ~3x budget 0: same model set, 3x the data.
for CFG in "$CONFIG_DIR"/model_100k.json "$CONFIG_DIR"/model_400k.json "$CONFIG_DIR"/model_1m.json "$CONFIG_DIR"/model_4m.json; do
  NAME_LO="iso_flop_low_$(basename "$CFG" .json)"
  NAME_HI="iso_flop_high_$(basename "$CFG" .json)"
  run_train "$CFG" "$NAME_LO" "$((NUM_SYNTHETIC / 2))"
  run_train "$CFG" "$NAME_HI" "$((NUM_SYNTHETIC * 2))"
done

# --- plots -------------------------------------------------------------------
echo "[info] generating plots"
python scripts/plot_scaling.py      --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"
python scripts/plot_iso_analysis.py --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"
python scripts/plot_ttp.py          --log_dir "$LOG_DIR" --output_dir "$PLOT_DIR"

echo
echo "[ok] done. plots in $PLOT_DIR"
ls -1 "$PLOT_DIR"
