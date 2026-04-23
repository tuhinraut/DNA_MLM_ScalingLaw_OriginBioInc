"""Core scaling-law plots: Loss vs N, Loss vs D, Loss vs C, with power-law fits.

Uses the *final* eval loss of each run (not per-step intermediate points), so the
fits reflect converged performance at each (N, D) pair.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(log_dir):
    """Return a list of dicts: {run, N, D, C, loss}, one per training log."""
    runs = []
    for p in sorted(Path(log_dir).glob('training_log_*.json')):
        with open(p) as f:
            data = json.load(f)
        N = data.get('num_parameters')
        D = data.get('final_tokens_seen')
        loss = data.get('final_eval_loss')
        if loss is None and data.get('log'):
            loss = data['log'][-1].get('eval_loss')
        if not (N and D and loss and loss > 0):
            continue
        runs.append({'run': data.get('run_name', p.stem),
                     'N': N, 'D': D, 'C': 6 * N * D, 'loss': loss})
    return runs


def _fit_power_law(x, y):
    """Fit log(y) = a + b*log(x); return (a, b) or (None, None) if degenerate."""
    log_x, log_y = np.log(x), np.log(y)
    if log_x.std() < 1e-6 or len(np.unique(log_x)) < 2:
        return None, None
    b, a = np.polyfit(log_x, log_y, 1)
    return a, b


def _plot_axis(ax, xs, ys, xlabel, color, label_prefix):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ax.scatter(xs, ys, s=80, c=color, edgecolors='black', linewidth=1.2, zorder=3)
    a, b = _fit_power_law(xs, ys) if len(xs) >= 2 else (None, None)
    if a is not None:
        grid = np.geomspace(xs.min(), xs.max(), 100)
        ax.plot(grid, np.exp(a) * grid ** b, 'r--', linewidth=2, alpha=0.8,
                label=f'L = {np.exp(a):.3f} · {label_prefix}^({b:.3f})')
        ax.legend(fontsize=10)
    return b


def plot_scaling(runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    Ns = [r['N'] for r in runs]
    Ds = [r['D'] for r in runs]
    Cs = [r['C'] for r in runs]
    Ls = [r['loss'] for r in runs]

    exponents = {}
    for (xs, label, color, key, fname) in [
        (Ns, 'Parameters (N)', 'steelblue', 'N', 'loss_vs_params.png'),
        (Ds, 'Training Tokens (D)', 'forestgreen', 'D', 'loss_vs_tokens.png'),
        (Cs, 'Compute FLOPs (C = 6ND)', 'darkorange', 'C', 'loss_vs_flops.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        b = _plot_axis(ax, xs, Ls, label, color, key)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(label); ax.set_ylabel('Eval Loss')
        ax.set_title(f'Loss vs {label} (final-eval, n={len(runs)})')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / fname, dpi=150)
        plt.close(fig)
        exponents[key] = b

    summary = {'runs': len(runs), 'exponents': exponents}
    with open(Path(output_dir) / 'scaling_exponents.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[plot_scaling] {len(runs)} runs -> {output_dir}/')
    for k, v in exponents.items():
        if v is not None:
            print(f'  L ~ {k}^{v:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--output_dir', default='plots')
    args = ap.parse_args()

    runs = load_runs(args.log_dir)
    if not runs:
        print(f'[plot_scaling] no usable training logs in {args.log_dir}')
        return
    plot_scaling(runs, args.output_dir)


if __name__ == '__main__':
    main()
