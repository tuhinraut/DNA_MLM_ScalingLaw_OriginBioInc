"""
TTP (Token-to-Parameter ratio) analysis plot.

Reads training log JSON files and produces an IsoFLOP plot:
for each approximate compute budget, plots loss as a function
of the TTP ratio (D / N) to show where the optimum lies.

Usage:
    python plot_ttp.py --log_dir logs --output_dir plots
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_logs(log_dir):
    """Load all training_log_*.json files from a directory."""
    logs = []
    for path in sorted(Path(log_dir).glob('training_log*.json')):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


def plot_ttp(logs, output_dir='plots'):
    """Produce the TTP analysis plot.

    TODO: Implement this function.

    The goal is an "IsoFLOP" style plot:
      - X-axis: TTP ratio  (D / N)  — tokens per parameter.
      - Y-axis: Test loss.
      - Each curve / colour represents a group of runs at roughly
        the same compute budget  (C ≈ 6 N D).

    Steps:
      1. For every completed run, extract N (params), D (tokens),
         final eval loss, and compute C = 6 * N * D.
      2. Bucket runs into iso-FLOP groups.  You decide the bucketing
         strategy (log-spaced bins, nearest order of magnitude, etc.).
      3. Within each bucket, plot loss vs. D/N.
      4. Mark the minimum-loss point on each curve — that is the
         empirically optimal TTP for that compute level.
      5. Optionally overlay a trend line through the optima to show
         how optimal TTP evolves with compute.

    Save the resulting figure to <output_dir>/ttp_analysis.png.
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TTP analysis plot from training logs')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing training_log_*.json files')
    parser.add_argument('--output_dir', type=str, default='plots')
    args = parser.parse_args()

    logs = load_training_logs(args.log_dir)
    if not logs:
        print(f"No training logs found in {args.log_dir}")
    else:
        print(f"Loaded {len(logs)} training logs")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_ttp(logs, args.output_dir)
