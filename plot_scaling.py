"""
Plotting utilities for scaling-law analysis.

Reads training log JSON files produced by train.py and generates
publication-style scaling plots.

Usage:
    python plot_scaling.py --log_dir logs --output_dir plots
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


def fit_power_law(x, y):
    """Fit y = a * x^b in log-log space via least-squares.

    Returns (a, b).
    """
    log_x = np.log(x)
    log_y = np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return np.exp(log_a), b


def _extract_final_metrics(logs):
    """Pull (params, tokens, final_eval_loss) from each log."""
    params, tokens, losses = [], [], []
    for log in logs:
        entries = log.get('log', [])
        if not entries:
            continue
        params.append(log['num_parameters'])
        tokens.append(log['final_tokens_seen'])
        losses.append(entries[-1]['eval_loss'])
    return np.array(params), np.array(tokens), np.array(losses)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def plot_loss_vs_params(params, losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(params, losses, s=60, zorder=5)

    if len(params) >= 2:
        a, b = fit_power_law(params, losses)
        x_fit = np.geomspace(params.min(), params.max(), 100)
        ax.plot(x_fit, a * x_fit ** b, '--',
                label=f'L = {a:.2e} · N^({b:.3f})')
        ax.legend()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (N)')
    ax.set_ylabel('Test Loss (L)')
    ax.set_title('Loss vs. Parameters')
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(output_dir) / 'loss_vs_params.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_vs_tokens(tokens, losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tokens, losses, s=60, zorder=5)

    if len(tokens) >= 2:
        a, b = fit_power_law(tokens, losses)
        x_fit = np.geomspace(tokens.min(), tokens.max(), 100)
        ax.plot(x_fit, a * x_fit ** b, '--',
                label=f'L = {a:.2e} · D^({b:.3f})')
        ax.legend()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Tokens (D)')
    ax.set_ylabel('Test Loss (L)')
    ax.set_title('Loss vs. Training Tokens')
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(output_dir) / 'loss_vs_tokens.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_vs_flops(flops, losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(flops, losses, s=60, zorder=5)

    if len(flops) >= 2:
        a, b = fit_power_law(flops, losses)
        x_fit = np.geomspace(flops.min(), flops.max(), 100)
        ax.plot(x_fit, a * x_fit ** b, '--',
                label=f'L = {a:.2e} · C^({b:.3f})')
        ax.legend()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (FLOPs)')
    ax.set_ylabel('Test Loss (L)')
    ax.set_title('Loss vs. Compute')
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(output_dir) / 'loss_vs_flops.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_scaling_laws(logs, output_dir='plots'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    params, tokens, losses = _extract_final_metrics(logs)
    if len(params) == 0:
        print("No completed runs found in logs.")
        return

    flops = 6 * params * tokens

    plot_loss_vs_params(params, losses, output_dir)
    plot_loss_vs_tokens(tokens, losses, output_dir)
    plot_loss_vs_flops(flops, losses, output_dir)

    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate scaling-law plots from training logs')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing training_log_*.json files')
    parser.add_argument('--output_dir', type=str, default='plots')
    args = parser.parse_args()

    logs = load_training_logs(args.log_dir)
    if not logs:
        print(f"No training logs found in {args.log_dir}")
    else:
        print(f"Loaded {len(logs)} training logs")
        plot_scaling_laws(logs, args.output_dir)
