"""Chinchilla-style IsoFLOP / TTP analysis.

Consumes the *final* (N, D, loss) of each training run -- one point per run --
and groups them into iso-FLOP buckets by log10(C). Produces a 4-panel figure:

    left   : loss vs N for each iso-FLOP bucket, parabolic fit, vertex marked
    center : optimal N vs C (power-law fit; Chinchilla ~ 0.50)
    right  : optimal D vs C (power-law fit; Chinchilla ~ 0.50)
    bottom : loss vs TTP = D/N for each iso-FLOP bucket (the spec's TTP view)

Requires at least one bucket with >=3 runs for parabolic fits.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_runs(log_dir):
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
        runs.append({'N': N, 'D': D, 'C': 6 * N * D, 'loss': loss,
                     'run': data.get('run_name', p.stem)})
    return runs


def bucket_by_flops(runs, round_digits=1):
    """Group runs by log10(C), rounded to `round_digits` decimal places."""
    buckets = defaultdict(list)
    for r in runs:
        buckets[round(np.log10(r['C']), round_digits)].append(r)
    return buckets


def _parabola_vertex(log_x, y):
    """Fit y = a*(log_x)^2 + b*log_x + c; return (coeffs, x_vertex, y_vertex)."""
    coeffs = np.polyfit(log_x, y, 2)
    a, b, c = coeffs
    if a <= 0:  # not a valley
        return coeffs, None, None
    lv = -b / (2 * a)
    return coeffs, np.exp(lv), a * lv * lv + b * lv + c


def plot_isoflop(runs, output_dir):
    buckets = bucket_by_flops(runs)
    sorted_keys = sorted(buckets)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(sorted_keys))))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.3)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_C = fig.add_subplot(gs[0, 1])
    ax_R = fig.add_subplot(gs[0, 2])
    ax_B = fig.add_subplot(gs[1, :])

    optima = []  # (C, N_opt, D_opt, loss_opt)

    for i, key in enumerate(sorted_keys):
        group = sorted(buckets[key], key=lambda r: r['N'])
        color = cmap[i]
        Ns = np.array([r['N'] for r in group], dtype=float)
        Ds = np.array([r['D'] for r in group], dtype=float)
        Ls = np.array([r['loss'] for r in group], dtype=float)

        ax_L.scatter(Ns, Ls, c=[color], s=80, edgecolors='black', linewidth=0.7, zorder=3)
        ax_B.scatter(Ds / Ns, Ls, c=[color], s=80, edgecolors='black', linewidth=0.7, zorder=3)

        if len(Ns) >= 3:
            coeffs, N_opt, L_opt = _parabola_vertex(np.log(Ns), Ls)
            grid = np.linspace(np.log(Ns.min()), np.log(Ns.max()), 200)
            ax_L.plot(np.exp(grid), np.polyval(coeffs, grid),
                      color=color, alpha=0.6, linewidth=1.8, zorder=2)
            if N_opt is not None:
                C_med = float(np.median([r['C'] for r in group]))
                D_opt = C_med / (6 * N_opt)
                optima.append((C_med, N_opt, D_opt, L_opt))
                ax_L.scatter([N_opt], [L_opt], marker='*', s=200,
                             c=[color], edgecolors='black', linewidth=1.2, zorder=4)
        elif len(Ns) >= 2:
            ax_L.plot(Ns, Ls, color=color, linestyle='--', alpha=0.5, linewidth=1.2)

    ax_L.set_xscale('log')
    ax_L.set_xlabel('Parameters (N)')
    ax_L.set_ylabel('Eval Loss')
    ax_L.set_title('IsoFLOP curves (loss vs N)')
    ax_L.grid(True, alpha=0.3, which='both')
    legend_handles = [
        Line2D([0], [0], color=cmap[i], marker='*', linewidth=2,
               label=f'6ND ≈ 1e{k:.1f}')
        for i, k in enumerate(sorted_keys)
    ]
    ax_L.legend(handles=legend_handles, fontsize=8, title='IsoFLOP', loc='best')

    # Center: optimal N vs C ------------------------------------------------
    if len(optima) >= 2:
        Cs = np.array([o[0] for o in optima])
        Ns = np.array([o[1] for o in optima])
        ax_C.scatter(Cs, Ns, c=cmap[:len(optima)], s=150, marker='*',
                     edgecolors='black', linewidth=1.2, zorder=3)
        b, a = np.polyfit(np.log(Cs), np.log(Ns), 1)
        grid = np.geomspace(Cs.min(), Cs.max(), 100)
        ax_C.plot(grid, np.exp(a) * grid ** b, 'k--', alpha=0.8,
                  label=f'N_opt ∝ C^{b:.3f}')
        ax_C.legend(fontsize=10)
    ax_C.set_xscale('log'); ax_C.set_yscale('log')
    ax_C.set_xlabel('FLOPs C'); ax_C.set_ylabel('Optimal N')
    ax_C.set_title('Optimal N vs compute')
    ax_C.grid(True, alpha=0.3, which='both')

    # Right: optimal D vs C -------------------------------------------------
    if len(optima) >= 2:
        Cs = np.array([o[0] for o in optima])
        Ds = np.array([o[2] for o in optima])
        ax_R.scatter(Cs, Ds, c=cmap[:len(optima)], s=150, marker='*',
                     edgecolors='black', linewidth=1.2, zorder=3)
        b, a = np.polyfit(np.log(Cs), np.log(Ds), 1)
        grid = np.geomspace(Cs.min(), Cs.max(), 100)
        ax_R.plot(grid, np.exp(a) * grid ** b, 'k--', alpha=0.8,
                  label=f'D_opt ∝ C^{b:.3f}')
        ax_R.legend(fontsize=10)
    ax_R.set_xscale('log'); ax_R.set_yscale('log')
    ax_R.set_xlabel('FLOPs C'); ax_R.set_ylabel('Optimal D')
    ax_R.set_title('Optimal D vs compute')
    ax_R.grid(True, alpha=0.3, which='both')

    # Bottom: loss vs TTP (the spec's TTP plot) -----------------------------
    ax_B.set_xscale('log')
    ax_B.set_xlabel('TTP = D / N')
    ax_B.set_ylabel('Eval Loss')
    ax_B.set_title('Loss vs tokens-per-parameter, one curve per iso-FLOP bucket')
    ax_B.grid(True, alpha=0.3, which='both')

    # Reconnect points per bucket on the TTP axis, parabolic fit over log(D/N).
    for i, key in enumerate(sorted_keys):
        group = sorted(buckets[key], key=lambda r: r['D'] / r['N'])
        if len(group) < 2:
            continue
        xs = np.array([r['D'] / r['N'] for r in group], dtype=float)
        ys = np.array([r['loss'] for r in group], dtype=float)
        color = cmap[i]
        if len(xs) >= 3:
            coeffs, x_opt, y_opt = _parabola_vertex(np.log(xs), ys)
            grid = np.linspace(np.log(xs.min()), np.log(xs.max()), 200)
            ax_B.plot(np.exp(grid), np.polyval(coeffs, grid),
                      color=color, alpha=0.6, linewidth=1.8, zorder=2)
            if x_opt is not None:
                ax_B.scatter([x_opt], [y_opt], marker='*', s=200, c=[color],
                             edgecolors='black', linewidth=1.2, zorder=4)
        else:
            ax_B.plot(xs, ys, color=color, linestyle='--', alpha=0.5, linewidth=1.2)

    fig.suptitle('IsoFLOP / TTP analysis  (final-eval points only)',
                 fontsize=13, fontweight='bold', y=0.995)
    out = Path(output_dir) / 'ttp_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'num_runs': len(runs),
        'num_buckets': len(sorted_keys),
        'bucket_sizes': {f'1e{k:.1f}': len(buckets[k]) for k in sorted_keys},
        'optima': [{'C': c, 'N_opt': n, 'D_opt': d, 'loss': L} for (c, n, d, L) in optima],
    }
    with open(Path(output_dir) / 'ttp_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[plot_ttp] {len(runs)} runs in {len(sorted_keys)} buckets -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--output_dir', default='plots')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.log_dir)
    if not runs:
        print(f'[plot_ttp] no usable training logs in {args.log_dir}')
        return
    plot_isoflop(runs, args.output_dir)


if __name__ == '__main__':
    main()
