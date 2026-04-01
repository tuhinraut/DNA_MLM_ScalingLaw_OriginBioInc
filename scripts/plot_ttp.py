"""
TTP / IsoFLOP analysis plot - Chinchilla-style (Hoffmann et al. 2022).

Produces Figure 3-style IsoFLOP curves with parabolic fits:
- Left panel: IsoFLOP curves (loss vs N) with parabolic fits showing valleys
- Center & Right panels: Optimal N and D projections vs FLOPs

Based on "Training Compute-Optimal Large Language Models" (Chinchilla paper).

Usage:
    python scripts/plot_ttp.py --log_dir experiments/logs --output_dir plots
"""

import json
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def load_training_logs(log_dir):
    """Load all training_log_*.json files from a directory."""
    logs = []
    for path in sorted(Path(log_dir).glob('training_log*.json')):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


def _loss_from_entry(entry):
    """Prefer training loss; fall back to eval loss."""
    for key in ('train_loss', 'eval_loss', 'loss'):
        if key in entry and entry[key] is not None:
            return float(entry[key])
    return None


def fit_parabola(x, y):
    """
    Fit a parabola y = a*(x-h)^2 + k to find the minimum (valley).
    Returns (a, h, k) where (h, k) is the vertex (minimum point).
    """
    try:
        # Fit quadratic: y = ax^2 + bx + c
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs

        # Vertex is at x = -b/(2a)
        h = -b / (2 * a)
        k = a * h**2 + b * h + c

        return a, h, k, coeffs
    except:
        return None, None, None, None


def plot_ttp_chinchilla(logs, output_dir='plots'):
    """
    Chinchilla-style IsoFLOP plot with parabolic fits (Figure 3).

    Key features:
    - Parabolic fits to each iso-FLOP bucket showing clear valleys
    - Marked minimum points (optimal N for each compute budget)
    - Three panels: IsoFLOP curves, optimal N projection, optimal D projection
    """
    entries = []
    for log in logs:
        log_entries = log.get('log', [])
        for entry in log_entries:
            N = entry['num_parameters']
            D = entry['tokens_seen']
            loss = _loss_from_entry(entry)
            if loss is None:
                continue
            C = 6 * N * D
            entries.append({'N': N, 'D': D, 'C': C, 'loss': loss})

    if not entries:
        print('No entries with loss values to plot.')
        return

    # Bucket by approximate compute (log10 FLOPs, one decimal).
    buckets = defaultdict(list)
    for e in entries:
        c_val = float(e['C'])
        buckets[round(np.log10(c_val), 1)].append(e)

    # Create three-panel figure (like Chinchilla Figure 3)
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    ax_left = fig.add_subplot(gs[0, 0])      # IsoFLOP curves
    ax_center = fig.add_subplot(gs[0, 1])     # Optimal N vs FLOPs
    ax_right = fig.add_subplot(gs[0, 2])     # Optimal D vs FLOPs

    # Color map for different FLOP budgets
    sorted_buckets = sorted(buckets.keys())
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(sorted_buckets)))

    # Store optimal points for center and right panels
    optimal_points = []  # (C, N_opt, D_opt, loss_opt)

    # ========== LEFT PANEL: IsoFLOP curves with parabolic fits ==========
    for idx, logc_key in enumerate(sorted_buckets):
        group = buckets[logc_key]
        color = cmap[idx]

        # Sort by N for plotting
        sorted_g = sorted(group, key=lambda x: x['N'])
        x_vals = np.array([e['N'] for e in sorted_g])
        y_vals = np.array([e['loss'] for e in sorted_g])
        d_vals = [e['D'] for e in sorted_g]

        # Plot data points
        for i, (x, y, d) in enumerate(zip(x_vals, y_vals, d_vals)):
            ax_left.scatter(
                x, y,
                c=[color],
                marker='o',
                s=80,
                edgecolors='0.2',
                linewidths=0.8,
                zorder=3,
                alpha=0.9,
            )

        # Fit and plot parabola (the key Chinchilla feature!)
        if len(sorted_g) >= 3:
            a, h, k, coeffs = fit_parabola(np.log(x_vals), y_vals)

            if a is not None and a > 0:  # a > 0 means it's a valley (convex)
                # Generate smooth curve
                x_smooth = np.linspace(np.log(x_vals.min()), np.log(x_vals.max()), 200)
                y_smooth = np.polyval(coeffs, x_smooth)

                # Plot parabolic fit
                ax_left.plot(
                    np.exp(x_smooth), y_smooth,
                    color=color,
                    linewidth=2,
                    alpha=0.6,
                    zorder=2,
                    linestyle='-',
                )

                # Mark the minimum point (optimal N for this C)
                n_opt = np.exp(h)
                loss_opt = k

                ax_left.scatter(
                    [n_opt], [loss_opt],
                    c=[color],
                    marker='*',
                    s=200,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=4,
                )

                # Store for center panel
                c_median = np.median([e['C'] for e in sorted_g])
                d_opt = c_median / (6 * n_opt)  # D = C / (6N)
                optimal_points.append((c_median, n_opt, d_opt, loss_opt))

                # Add annotation for the minimum
                ax_left.annotate(
                    f'{n_opt/1e6:.1f}M',
                    (n_opt, loss_opt),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    color=color,
                    fontweight='bold',
                )
        else:
            # Just connect points with line if not enough for parabola
            ax_left.plot(
                x_vals, y_vals,
                color=color,
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
                linestyle='--',
            )

    ax_left.set_xlabel('Parameters (N)', fontsize=11, fontweight='bold')
    ax_left.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax_left.set_title('IsoFLOP curves (parabolic fits)\nMarker = minimum (optimal N)',
                      fontsize=12, fontweight='bold')
    ax_left.set_xscale('log')
    ax_left.grid(True, alpha=0.3, which='both')

    # Legend for FLOP buckets
    flop_handles = [
        Line2D(
            [0], [0],
            linestyle='-',
            color=cmap[i],
            linewidth=2.5,
            marker='*',
            markersize=10,
            label=f'6ND = 10^{k:.1f}',
        )
        for i, k in enumerate(sorted_buckets)
    ]
    ax_left.legend(
        handles=flop_handles,
        title='IsoFLOP Budget',
        loc='upper right',
        fontsize=8,
        title_fontsize=9,
    )

    # ========== CENTER PANEL: Optimal N vs FLOPs ==========
    if optimal_points:
        c_vals = np.array([p[0] for p in optimal_points])
        n_vals = np.array([p[1] for p in optimal_points])

        # Plot optimal points
        ax_center.scatter(
            c_vals, n_vals,
            c=cmap[:len(optimal_points)],
            s=150,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            zorder=3,
        )

        # Fit power law: N_opt ~ C^a
        log_c = np.log(c_vals)
        log_n = np.log(n_vals)

        if len(c_vals) >= 2:
            # Linear fit in log-log space
            fit = np.polyfit(log_c, log_n, 1)
            a, b = fit  # N = exp(b) * C^a

            # Generate smooth curve
            c_smooth = np.linspace(c_vals.min(), c_vals.max() * 10, 200)
            n_pred = np.exp(b) * c_smooth ** a

            ax_center.plot(
                c_smooth, n_pred,
                'k--',
                linewidth=2,
                alpha=0.7,
                zorder=2,
                label=f'N ~ C^{a:.2f}',
            )

            print(f"\nPower law fit: N_opt ~ C^{a:.3f}")
            print(f"  (Chinchilla paper reports a ~ 0.49-0.50)")

        ax_center.set_xlabel('FLOPs (C = 6ND)', fontsize=11, fontweight='bold')
        ax_center.set_ylabel('Optimal Parameters (N)', fontsize=11, fontweight='bold')
        ax_center.set_title('Optimal model size vs compute\n(star = minimum of each parabola)',
                          fontsize=12, fontweight='bold')
        ax_center.set_xscale('log')
        ax_center.set_yscale('log')
        ax_center.grid(True, alpha=0.3, which='both')
        ax_center.legend(loc='upper left', fontsize=9)

        # Add Chinchilla-style annotation
        if len(c_vals) > 0:
            max_c = c_vals.max()
            max_n = n_vals[c_vals.argmax()]
            ax_center.annotate(
                f'C = {max_c:.1e}\nN = {max_n/1e6:.0f}M',
                (max_c, max_n),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            )

    # ========== RIGHT PANEL: Optimal D vs FLOPs ==========
    if optimal_points:
        c_vals = np.array([p[0] for p in optimal_points])
        d_vals = np.array([p[2] for p in optimal_points])

        # Plot optimal points
        ax_right.scatter(
            c_vals, d_vals,
            c=cmap[:len(optimal_points)],
            s=150,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            zorder=3,
        )

        # Fit power law: D_opt ~ C^b
        log_c = np.log(c_vals)
        log_d = np.log(d_vals)

        if len(c_vals) >= 2:
            fit = np.polyfit(log_c, log_d, 1)
            b, c = fit  # D = exp(c) * C^b

            # Generate smooth curve
            c_smooth = np.linspace(c_vals.min(), c_vals.max() * 10, 200)
            d_pred = np.exp(c) * c_smooth ** b

            ax_right.plot(
                c_smooth, d_pred,
                'k--',
                linewidth=2,
                alpha=0.7,
                zorder=2,
                label=f'D ~ C^{b:.2f}',
            )

            print(f"Power law fit: D_opt ~ C^{b:.3f}")
            print(f"  (Chinchilla paper reports b ~ 0.50-0.51)")
            print(f"\nScaling recommendation: For every doubling of compute,")
            print(f"  increase model size by {2**a:.1f}x and tokens by {2**b:.1f}x")

        ax_right.set_xlabel('FLOPs (C = 6ND)', fontsize=11, fontweight='bold')
        ax_right.set_ylabel('Optimal Training Tokens (D)', fontsize=11, fontweight='bold')
        ax_right.set_title('Optimal tokens vs compute\n(star = from parabola minima)',
                         fontsize=12, fontweight='bold')
        ax_right.set_xscale('log')
        ax_right.set_yscale('log')
        ax_right.grid(True, alpha=0.3, which='both')
        ax_right.legend(loc='upper left', fontsize=9)

        # Add Chinchilla-style annotation
        if len(c_vals) > 0:
            max_c = c_vals.max()
            max_d = d_vals[c_vals.argmax()]
            ax_right.annotate(
                f'C = {max_c:.1e}\nD = {max_d/1e9:.1f}B',
                (max_c, max_d),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3),
            )

    # Overall title
    fig.suptitle(
        'Chinchilla-style IsoFLOP Analysis\n' +
        '(Following Hoffmann et al. 2022: "Training Compute-Optimal Large Language Models")',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttp_analysis_chinchilla.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nChinchilla-style plot saved to: {output_dir}/ttp_analysis_chinchilla.png")
    print(f"  Data points: {len(entries)}")
    print(f"  IsoFLOP buckets: {len(buckets)}")
    print(f"  Parabolic fits: {len(optimal_points)}")


def plot_ttp_simple(logs, output_dir='plots'):
    """
    Simple IsoFLOP plot (original style, for comparison).
    """
    entries = []
    for log in logs:
        log_entries = log.get('log', [])
        for entry in log_entries:
            N = entry['num_parameters']
            D = entry['tokens_seen']
            loss = _loss_from_entry(entry)
            if loss is None:
                continue
            C = 6 * N * D
            entries.append({'N': N, 'D': D, 'C': C, 'loss': loss})

    if not entries:
        print('No entries with loss values to plot.')
        return

    buckets = defaultdict(list)
    for e in entries:
        buckets[round(np.log10(e['C']), 1)].append(e)

    all_D = sorted({e['D'] for e in entries})
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']
    D_to_marker = {d: marker_cycle[i % len(marker_cycle)] for i, d in enumerate(all_D)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(buckets), 1)))

    for idx, logc_key in enumerate(sorted(buckets.keys())):
        group = buckets[logc_key]
        color = cmap[idx % len(cmap)]
        for e in sorted(group, key=lambda x: x['N']):
            ax.scatter(
                e['N'], e['loss'],
                c=[color], marker=D_to_marker[e['D']], s=72,
                edgecolors='0.2', linewidths=0.6, zorder=3,
            )
        sorted_g = sorted(group, key=lambda x: x['N'])
        if len(sorted_g) > 1:
            ax.plot(
                [x['N'] for x in sorted_g], [x['loss'] for x in sorted_g],
                color=color, alpha=0.35, linewidth=1.2, zorder=1,
            )

    ax.set_xlabel('Parameters (N)')
    ax.set_ylabel('Loss')
    ax.set_title('IsoFLOP curves: loss vs parameters (marker = tokens seen D)')

    n_span = max(e['N'] for e in entries) / max(min(e['N'] for e in entries), 1)
    if n_span > 10:
        ax.set_xscale('log')

    flop_handles = [
        Line2D([0], [0], linestyle='-', color=cmap[i % len(cmap)], linewidth=2.5,
               label=f'6ND ~ {float(np.median([e["C"] for e in buckets[k]])):.2e}')
        for i, k in enumerate(sorted(buckets.keys()))
    ]
    token_handles = [
        Line2D([0], [0], linestyle='', marker=D_to_marker[d], color='0.35', markersize=8,
               label=f'D = {d:,}')
        for d in all_D
    ]

    leg1 = ax.legend(handles=flop_handles, title='IsoFLOP (approx.)',
                     loc='upper right', fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=token_handles, title='Tokens seen (D)',
              loc='lower left', fontsize=8)

    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttp_analysis.png'), dpi=150)
    plt.close()
    print(f"Simple plot saved to: {output_dir}/ttp_analysis.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TTP / IsoFLOP analysis plots (Chinchilla-style with parabolic fits)')
    parser.add_argument('--log_dir', type=str, default='experiments/logs',
                        help='Directory containing training_log_*.json files')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--style', type=str, default='chinchilla',
                        choices=['chinchilla', 'simple'],
                        help='Plot style: chinchilla (3-panel with parabolas) or simple (1-panel)')
    args = parser.parse_args()

    logs = load_training_logs(args.log_dir)
    if not logs:
        print(f"No training logs found in {args.log_dir}")
    else:
        print(f"Loaded {len(logs)} training logs")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        if args.style == 'chinchilla':
            plot_ttp_chinchilla(logs, args.output_dir)
        else:
            plot_ttp_simple(logs, args.output_dir)
