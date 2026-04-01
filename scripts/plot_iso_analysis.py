#!/usr/bin/env python3
"""
Iso-Analysis plotting - Visualizing scaling laws with 2D hyperparameter relationships.

This script creates plots that properly show the Iso-Token, Iso-Param, and Iso-FLOP
analyses by tracking how loss varies with TWO changing hyperparameters.

Key visualizations:
1. Iso-Token: Training curves (loss vs tokens) colored by model size
2. Iso-Param: Training curves (loss vs tokens) colored by data size  
3. Iso-FLOP: Loss vs parameters for fixed compute (parabolic fits)
4. 2D Contour: Loss landscape in (N, D) space with efficient frontier

Usage:
    python scripts/plot_iso_analysis.py --log_dir experiments/logs --output_dir plots
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
import matplotlib.colors as mcolors


def load_training_logs(log_dir):
    """Load all training_log_*.json files and categorize by run type."""
    logs = {'iso_token': [], 'iso_param': [], 'iso_flop': []}
    
    for path in sorted(Path(log_dir).glob('training_log*.json')):
        with open(path) as f:
            data = json.load(f)
            run_name = data.get('run_name', path.stem)
            
            if 'iso_token' in run_name:
                logs['iso_token'].append(data)
            elif 'iso_param' in run_name:
                logs['iso_param'].append(data)
            elif 'iso_flop' in run_name:
                logs['iso_flop'].append(data)
            else:
                # Default to iso_token if unclear
                logs['iso_token'].append(data)
    
    return logs


def plot_iso_token_curves(logs, output_dir):
    """
    Iso-Token Analysis: Fixed data, varying model size.
    
    Shows scaling relationship between model size and loss.
    Single plot: Loss vs Model Parameters with power law fit.
    """
    if not logs:
        print("No Iso-Token logs found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by model size (parameters)
    model_groups = defaultdict(list)
    data_tokens = None
    for log in logs:
        params = log.get('num_parameters', 0)
        model_groups[params].append(log)
        # Extract data size from final_tokens_seen
        tokens = log.get('final_tokens_seen', 0)
        if tokens > 0:
            data_tokens = tokens
    
    # Sort by parameter count
    sorted_params = sorted(model_groups.keys())
    
    # Collect data points for each model size
    data_points = []
    
    for params in sorted_params:
        for log in model_groups[params]:
            entries = log.get('log', [])
            if not entries:
                continue
            
            final_loss = entries[-1].get('eval_loss', entries[-1].get('train_loss', 0))
            data_points.append((params, final_loss))
    
    if not data_points:
        print("No data points for Iso-Token plot")
        return
    
    params_arr = np.array([p[0] for p in data_points])
    losses_arr = np.array([p[1] for p in data_points])
    
    # Scatter plot
    ax.scatter(params_arr, losses_arr, s=150, alpha=0.8, c='steelblue', 
               edgecolors='black', linewidth=1.5, zorder=5)
    
    # Power law fit
    if len(params_arr) >= 2:
        log_p = np.log(params_arr)
        log_l = np.log(losses_arr)
        coeffs = np.polyfit(log_p, log_l, 1)
        b, a = coeffs  # log(L) = a + b*log(N)
        
        # Plot fit
        p_fit = np.geomspace(params_arr.min(), params_arr.max(), 100)
        l_fit = np.exp(a) * p_fit ** b
        ax.plot(p_fit, l_fit, 'r--', linewidth=2.5, 
                label=f'L = {np.exp(a):.3f} × N^({b:.3f})', zorder=3)
        ax.legend(fontsize=12, loc='upper right')
        
        print(f"\nIso-Token Scaling Law: L ∝ N^{b:.3f}")
        print(f"  (Exponents around -0.5 to -0.7 are typical)")
    
    # Format data size for title
    data_str = f"{data_tokens/1e6:.1f}M" if data_tokens and data_tokens >= 1e6 else f"{data_tokens/1e3:.1f}K" if data_tokens else "unknown"
    
    ax.set_xlabel('Model Parameters (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'Iso-Token Analysis: Loss vs Model Size\n(Fixed data D={data_str}, varying N)', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'iso_token_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: iso_token_analysis.png")


def plot_iso_param_curves(logs, output_dir):
    """
    Iso-Param Analysis: Fixed model, varying data size.
    
    Shows scaling relationship between data size and loss for a fixed model.
    """
    if not logs:
        print("No Iso-Param logs found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by data size (inferred from total tokens in final step)
    data_groups = defaultdict(list)
    model_params = None
    for log in logs:
        total_tokens = log.get('final_tokens_seen', 0)
        params = log.get('num_parameters', 0)
        if params > 0:
            model_params = params
        if total_tokens > 0:
            data_groups[total_tokens].append(log)
    
    sorted_tokens = sorted(data_groups.keys())
    
    # Collect loss vs data size points
    data_points = []
    for total_tok in sorted_tokens:
        for log in data_groups[total_tok]:
            entries = log.get('log', [])
            if entries:
                final_loss = entries[-1].get('eval_loss', entries[-1].get('train_loss', 0))
                data_points.append((total_tok, final_loss))
    
    if data_points:
        tokens_arr = np.array([p[0] for p in data_points])
        losses_arr = np.array([p[1] for p in data_points])
        
        ax.scatter(tokens_arr, losses_arr, s=150, alpha=0.8, c='forestgreen', 
                   edgecolors='black', linewidth=1.5, zorder=5)
        
        # Power law fit
        if len(tokens_arr) >= 2:
            log_t = np.log(tokens_arr)
            log_l = np.log(losses_arr)
            coeffs = np.polyfit(log_t, log_l, 1)
            b, a = coeffs
            
            t_fit = np.geomspace(tokens_arr.min(), tokens_arr.max(), 100)
            l_fit = np.exp(a) * t_fit ** b
            ax.plot(t_fit, l_fit, 'r--', linewidth=2.5, zorder=3,
                    label=f'L = {np.exp(a):.3f} × D^({b:.3f})')
            ax.legend(fontsize=12, loc='upper right')
            
            print(f"\nIso-Param Scaling Law: L ∝ D^{b:.3f}")
            print(f"  (Exponents around -0.5 are typical)")
        
        # Add model size info to title
        param_str = f"{model_params/1e6:.1f}M" if model_params and model_params >= 1e6 else f"{model_params/1e3:.1f}K" if model_params else "unknown"
        
        ax.set_xlabel('Training Tokens (D)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax.set_title(f'Iso-Param Analysis: Loss vs Data Size\n(Fixed model N={param_str}, varying D)', 
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'iso_param_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: iso_param_analysis.png")


def plot_2d_loss_landscape(all_logs, output_dir):
    """
    Create a 2D contour plot of loss in (N, D) space.
    Shows the efficient frontier and iso-loss contours.
    """
    # Collect all final (N, D, Loss) points
    points = []
    for log in all_logs:
        entries = log.get('log', [])
        if not entries:
            continue
        
        N = log.get('num_parameters', 0)
        for entry in entries:
            D = entry.get('tokens_seen', 0)
            loss = entry.get('eval_loss', entry.get('train_loss', None))
            if loss is not None and N > 0 and D > 0:
                points.append((N, D, loss))
    
    if len(points) < 10:
        print("Not enough data points for 2D loss landscape")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract arrays
    N_vals = np.array([p[0] for p in points])
    D_vals = np.array([p[1] for p in points])
    L_vals = np.array([p[2] for p in points])
    
    # Create scatter plot with color-coded loss
    scatter = ax.scatter(N_vals, D_vals, c=L_vals, s=80, alpha=0.6, 
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Loss', fontsize=12, fontweight='bold')
    
    # Try to fit and plot efficient frontier
    # For each N, find minimum loss
    unique_N = np.unique(N_vals)
    frontier_points = []
    
    for N in unique_N:
        mask = N_vals == N
        if mask.sum() > 0:
            min_loss_idx = np.argmin(L_vals[mask])
            D_at_min = D_vals[mask][min_loss_idx]
            frontier_points.append((N, D_at_min))
    
    if len(frontier_points) > 1:
        frontier_N = np.array([p[0] for p in frontier_points])
        frontier_D = np.array([p[1] for p in frontier_points])
        
        # Sort by N
        sort_idx = np.argsort(frontier_N)
        frontier_N = frontier_N[sort_idx]
        frontier_D = frontier_D[sort_idx]
        
        # Plot frontier
        ax.plot(frontier_N, frontier_D, 'r-', linewidth=3, 
                label='Efficient Frontier', marker='*', markersize=15)
        ax.legend(fontsize=11, loc='upper left')
    
    ax.set_xlabel('Model Parameters (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Tokens (D)', fontsize=13, fontweight='bold')
    ax.set_title('Loss Landscape in (N, D) Space\nLower loss = better (darker colors)', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'loss_landscape_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: loss_landscape_2d.png")


def plot_combined_scaling_summary(all_logs, output_dir):
    """
    Create a comprehensive summary plot combining all analyses.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Collect all final points
    all_points = []
    for log in all_logs:
        entries = log.get('log', [])
        if entries:
            N = log.get('num_parameters', 0)
            final = entries[-1]
            D = final.get('tokens_seen', 0)
            L = final.get('eval_loss', final.get('train_loss', 0))
            C = 6 * N * D
            all_points.append((N, D, L, C))
    
    if not all_points:
        print("No data for summary plot")
        return
    
    N_arr = np.array([p[0] for p in all_points])
    D_arr = np.array([p[1] for p in all_points])
    L_arr = np.array([p[2] for p in all_points])
    C_arr = np.array([p[3] for p in all_points])
    
    # Plot 1: Loss vs N
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(N_arr, L_arr, s=80, alpha=0.6, c='steelblue', edgecolors='black')
    ax1.set_xlabel('Parameters (N)', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss vs Model Size', fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs D
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(D_arr, L_arr, s=80, alpha=0.6, c='forestgreen', edgecolors='black')
    ax2.set_xlabel('Tokens (D)', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Loss vs Data Size', fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss vs C
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(C_arr, L_arr, s=80, alpha=0.6, c=L_arr, 
                          cmap='viridis_r', edgecolors='black')
    ax3.set_xlabel('FLOPs (C=6ND)', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Loss vs Compute', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Loss')
    
    # Plot 4: N vs D (showing compute budget lines)
    ax4 = fig.add_subplot(gs[1, :2])
    scatter = ax4.scatter(N_arr, D_arr, c=L_arr, s=100, alpha=0.7, 
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add iso-FLOP lines
    C_values = np.geomspace(C_arr.min() if C_arr.min() > 0 else 1e15, 
                            C_arr.max(), 5)
    N_range = np.geomspace(N_arr.min() if N_arr.min() > 0 else 1e5, 
                           N_arr.max(), 100)
    
    for C in C_values:
        D_iso = C / (6 * N_range)
        ax4.plot(N_range, D_iso, 'w--', alpha=0.5, linewidth=1)
        # Label at middle point
        idx = len(N_range) // 2
        ax4.annotate(f'C={C:.1e}', (N_range[idx], D_iso[idx]), 
                fontsize=7, color='white', alpha=0.7)
    
    ax4.set_xlabel('Parameters (N)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Tokens (D)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Size vs Data Size (colored by loss)\nWhite dashed = Iso-FLOP lines', 
                 fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Loss')
    
    # Plot 5: Scaling summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Fit power laws
    if len(N_arr) >= 3:
        log_n = np.log(N_arr)
        log_d = np.log(D_arr)
        log_l = np.log(L_arr)
        log_c = np.log(C_arr)
        
        # L vs N
        coeffs_ln = np.polyfit(log_n, log_l, 1)
        alpha_n = coeffs_ln[0]
        
        # L vs D
        coeffs_ld = np.polyfit(log_d, log_l, 1)
        alpha_d = coeffs_ld[0]
        
        # L vs C
        coeffs_lc = np.polyfit(log_c, log_l, 1)
        alpha_c = coeffs_lc[0]
        
        summary_text = f"""
Scaling Law Summary:

L ∝ N^{alpha_n:.3f}
L ∝ D^{alpha_d:.3f}  
L ∝ C^{alpha_c:.3f}

Data Points: {len(all_points)}
Model Sizes: {len(np.unique(N_arr))}
Data Sizes: {len(np.unique(D_arr))}

Chinchilla predicts:
L ∝ N^-0.50
L ∝ D^-0.50
L ∝ C^-0.25

See individual analysis
plots for detailed views.
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Scaling Law Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(Path(output_dir) / 'scaling_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scaling_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive Iso-analysis plots')
    parser.add_argument('--log_dir', type=str, default='experiments/logs',
                        help='Directory containing training_log_*.json files')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    print(f"Loading logs from: {args.log_dir}")
    logs_by_type = load_training_logs(args.log_dir)
    
    all_logs = []
    for log_type, logs in logs_by_type.items():
        print(f"  Found {len(logs)} {log_type} logs")
        all_logs.extend(logs)
    
    if not all_logs:
        print("No training logs found!")
        return
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots in: {args.output_dir}")
    
    # Generate individual analysis plots
    plot_iso_token_curves(logs_by_type['iso_token'], args.output_dir)
    plot_iso_param_curves(logs_by_type['iso_param'], args.output_dir)
    plot_2d_loss_landscape(all_logs, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")
    print("Generated:")
    print("  - iso_token_analysis.png: Training curves by model size")
    print("  - iso_param_analysis.png: Training curves by data size")
    print("  - loss_landscape_2d.png: 2D loss landscape with frontier")


if __name__ == '__main__':
    main()
