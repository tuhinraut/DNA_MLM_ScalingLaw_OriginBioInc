"""Supplementary iso-axis plots (one panel per spec axis).

Each run contributes exactly one final-eval point -- intermediate checkpoints
are ignored. Run names starting with `iso_token`, `iso_param`, or `iso_flop`
are routed to their respective panel.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs_by_phase(log_dir):
    by_phase = defaultdict(list)
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
        name = data.get('run_name', p.stem)
        key = 'iso_token' if name.startswith('iso_token') else \
              'iso_param' if name.startswith('iso_param') else \
              'iso_flop' if name.startswith('iso_flop') else 'other'
        by_phase[key].append({'run': name, 'N': N, 'D': D, 'C': 6 * N * D, 'loss': loss})
    return by_phase


def _fit_line(ax, xs, ys, xlabel, var, color):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ax.scatter(xs, ys, s=100, c=color, edgecolors='black', linewidth=1, zorder=3)
    if len(xs) >= 2 and len(np.unique(xs)) >= 2:
        b, a = np.polyfit(np.log(xs), np.log(ys), 1)
        grid = np.geomspace(xs.min(), xs.max(), 100)
        ax.plot(grid, np.exp(a) * grid ** b, 'r--', linewidth=2, alpha=0.8,
                label=f'L ∝ {var}^{b:.3f}')
        ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel('Eval Loss')
    ax.grid(True, alpha=0.3, which='both')


def plot_iso(by_phase, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if by_phase.get('iso_token'):
        runs = by_phase['iso_token']
        fig, ax = plt.subplots(figsize=(8, 5))
        _fit_line(ax, [r['N'] for r in runs], [r['loss'] for r in runs],
                  'Parameters (N)', 'N', 'steelblue')
        ax.set_title(f'Iso-Token: loss vs N (n={len(runs)})')
        fig.tight_layout(); fig.savefig(out_dir / 'iso_token_analysis.png', dpi=150); plt.close(fig)

    if by_phase.get('iso_param'):
        runs = by_phase['iso_param']
        fig, ax = plt.subplots(figsize=(8, 5))
        _fit_line(ax, [r['D'] for r in runs], [r['loss'] for r in runs],
                  'Training Tokens (D)', 'D', 'forestgreen')
        ax.set_title(f'Iso-Param: loss vs D (n={len(runs)})')
        fig.tight_layout(); fig.savefig(out_dir / 'iso_param_analysis.png', dpi=150); plt.close(fig)

    if by_phase.get('iso_flop'):
        runs = by_phase['iso_flop']
        fig, ax = plt.subplots(figsize=(8, 5))
        _fit_line(ax, [r['C'] for r in runs], [r['loss'] for r in runs],
                  'FLOPs (C)', 'C', 'darkorange')
        ax.set_title(f'Iso-FLOP: loss vs C (n={len(runs)})')
        fig.tight_layout(); fig.savefig(out_dir / 'iso_flop_analysis.png', dpi=150); plt.close(fig)

    # 2D landscape across every run
    all_runs = [r for rs in by_phase.values() for r in rs]
    if len(all_runs) >= 4:
        Ns = np.array([r['N'] for r in all_runs], dtype=float)
        Ds = np.array([r['D'] for r in all_runs], dtype=float)
        Ls = np.array([r['loss'] for r in all_runs], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(Ns, Ds, c=Ls, cmap='viridis_r', s=120,
                        edgecolors='black', linewidth=0.6)
        plt.colorbar(sc, ax=ax, label='Eval Loss')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Parameters (N)'); ax.set_ylabel('Training Tokens (D)')
        ax.set_title(f'Loss landscape in (N, D)  (n={len(all_runs)})')
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout(); fig.savefig(out_dir / 'loss_landscape_2d.png', dpi=150); plt.close(fig)

    print(f'[plot_iso] {sum(len(v) for v in by_phase.values())} runs -> {out_dir}/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--output_dir', default='plots')
    args = ap.parse_args()
    by_phase = load_runs_by_phase(args.log_dir)
    if not any(by_phase.values()):
        print(f'[plot_iso] no usable training logs in {args.log_dir}')
        return
    plot_iso(by_phase, args.output_dir)


if __name__ == '__main__':
    main()
