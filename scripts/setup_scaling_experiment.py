"""Generate model/data configs for the three-phase scaling-law sweep.

Everything here is CLI-configurable -- no hardcoded scale counts.

Produces, under <experiment_dir>:
    configs/iso_token/*.json      -- N sweep at ~constant D (the full dataset)
    configs/iso_param/*.json      -- one architecture, many D values
    configs/iso_flop/<C>/*.json   -- one subdirectory per compute budget,
                                     each holding an N sweep at that C
    prepare_data_samples.sh       -- creates token-budgeted FASTA samples
                                     (for iso-param and per-iso-flop-run)
    experiment_summary.json       -- run manifest
"""

import argparse
import json
import math
import os
from pathlib import Path


# --- Architecture helpers ----------------------------------------------------

def _round_heads(d_model):
    """Return n_heads such that head_dim = 64. Never goes below head_dim = 64."""
    return max(1, d_model // 64)


def _architecture_for(target_params, max_seq_len, vocab_size=7):
    """Pick (d_model, n_heads, n_layers, d_ff) whose estimated param count is
    close to `target_params`. Uses the standard 4x FFN ratio and a width/depth
    schedule that roughly matches GPT-like models.

    param estimate:
        E  = 2*V*d              (embed + tied-shape output head)
        A  = 4*d*d              (Q,K,V,O per layer)
        F  = 2*d*ff             (2 linears per layer)
        LN = 4*d                (2 layernorms per layer)
        total = E + n_layers*(A + F + LN)

    Constraints:
        - d_model must be a multiple of 64 (ensures head_dim = 64 exactly)
        - n_layers <= 4 * (d_model // 64)  (prevents extreme narrow+deep configs)
    """
    best = None
    for d_model in (64, 128, 192, 256, 320, 384, 512, 640, 768, 896,
                    1024, 1152, 1280, 1408, 1536, 1792, 2048):
        d_ff = 4 * d_model
        max_layers = 4 * (d_model // 64)
        for n_layers in range(2, max_layers + 1):
            per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model
            total = 2 * vocab_size * d_model + n_layers * per_layer
            err = abs(total - target_params) / target_params
            if best is None or err < best[0]:
                best = (err, d_model, n_layers, d_ff, total)
    _, d_model, n_layers, d_ff, total = best
    n_heads = _round_heads(d_model)
    return {
        'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
        'd_ff': d_ff, 'dropout': 0.1, 'vocab_size': vocab_size,
        'max_seq_len': max_seq_len, '_params_est': total,
    }


# --- Folder scaffolding ------------------------------------------------------

def _mkdirs(base):
    dirs = [
        'configs/iso_token', 'configs/iso_param', 'configs/iso_flop',
        'logs', 'checkpoints', 'results', 'data_samples', 'utils',
    ]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)


# --- Phase builders ----------------------------------------------------------

def build_iso_token(cfg_dir, n_points, min_params, max_params, max_seq_len):
    """N sweep: n_points model sizes evenly in log space."""
    out = []
    grid = [int(round(x)) for x in _logspace(min_params, max_params, n_points)]
    for N in grid:
        arch = _architecture_for(N, max_seq_len)
        name = f'model_{_human(arch["_params_est"])}'
        path = cfg_dir / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(arch, f, indent=2)
        out.append({'file': str(path), 'params': arch['_params_est'], 'name': name})
    return out


def build_iso_param(cfg_dir, params_target, max_seq_len):
    """One architecture; the D sweep is done by selecting different FASTAs."""
    arch = _architecture_for(params_target, max_seq_len)
    name = f'model_{_human(arch["_params_est"])}'
    path = cfg_dir / f'{name}.json'
    with open(path, 'w') as f:
        json.dump(arch, f, indent=2)
    return {'file': str(path), 'params': arch['_params_est'], 'name': name}


def build_iso_flop(cfg_dir, compute_budgets, n_points_per_bucket, max_seq_len,
                   d_total_tokens):
    """For each compute budget C, sweep N and compute D = C/(6N).

    A run is only emitted if its required D is <= `d_total_tokens` (otherwise
    we cannot feed enough data).
    """
    all_configs = []
    for C in compute_budgets:
        bucket_dir = cfg_dir / f'C_{C:.2e}'
        bucket_dir.mkdir(parents=True, exist_ok=True)
        # Pick N grid centred on the Chinchilla-optimal for this C (~sqrt(C/6/20)
        # roughly). We just sweep over a decade on each side.
        N_centre = max(int(math.sqrt(C / (6 * 20))), 10_000)
        N_lo = max(10_000, N_centre // 10)
        N_hi = min(N_centre * 10, 500_000_000)
        if N_hi <= N_lo:
            continue
        Ns = _logspace(N_lo, N_hi, n_points_per_bucket)
        for N_target in Ns:
            arch = _architecture_for(int(round(N_target)), max_seq_len)
            N = arch['_params_est']
            D = int(C / (6 * N))
            if D < 1_000_000 or D > d_total_tokens:
                continue
            meta = {
                '_meta': {
                    'target_flops': C,
                    'params': N,
                    'required_tokens': D,
                    'data_fraction': D / d_total_tokens,
                }
            }
            full = {**arch, **meta}
            name = f'iso_flop_{_sci(C)}_{_human(N)}'
            path = bucket_dir / f'{name}.json'
            with open(path, 'w') as f:
                json.dump(full, f, indent=2)
            all_configs.append({
                'file': str(path), 'params': N, 'tokens': D,
                'flops': C, 'name': name, 'bucket': f'{C:.2e}',
            })
    return all_configs


# --- Helpers -----------------------------------------------------------------

def _logspace(lo, hi, n):
    if n <= 1:
        return [hi]
    log_lo, log_hi = math.log(lo), math.log(hi)
    return [math.exp(log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]


def _human(n):
    for thr, suf in [(1_000_000_000, 'b'), (1_000_000, 'm'), (1_000, 'k')]:
        if n >= thr:
            return f'{n / thr:.1f}{suf}'.replace('.0', '')
    return str(n)


def _sci(x):
    return f'{x:.1e}'.replace('+', '').replace('.0e', 'e')


# --- Sampling utility generator ---------------------------------------------

def write_sampling_utility(base):
    """Writes a helper that samples sequences *by target token count*."""
    code = '''\
"""Sample sequences from FASTA files until a token budget is hit."""
import argparse, random
from pathlib import Path


def sample_tokens(input_dir, target_tokens, output_file,
                  min_len=64, max_len=2048, seed=42, pattern='*_CDS.fasta'):
    random.seed(seed)
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'sample_to_fasta: no files matching "{pattern}" in {input_dir}')
    records = []
    for fp in files:
        current = []
        header = None
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header and current:
                        seq = ''.join(current)
                        if len(seq) >= min_len:
                            records.append((header, seq[:max_len]))
                    header = line
                    current = []
                else:
                    current.append(line)
        if header and current:
            seq = ''.join(current)
            if len(seq) >= min_len:
                records.append((header, seq[:max_len]))
    random.shuffle(records)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    used = 0
    written = 0
    with open(out, 'w') as f:
        for header, seq in records:
            if used >= target_tokens:
                break
            f.write(header + '\\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\\n')
            used += len(seq)
            written += 1
    print(f'  {out}: wrote {written:,} seqs / {used:,} tokens (target {target_tokens:,})')
    return used


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--target_tokens', type=int, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min_len', type=int, default=64)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pattern', default='*_CDS.fasta',
                    help='Glob pattern for input FASTA files.')
    args = ap.parse_args()
    sample_tokens(args.data_dir, args.target_tokens, args.output,
                  args.min_len, args.max_len, args.seed, args.pattern)
'''
    path = base / 'utils' / 'sample_to_fasta.py'
    path.write_text(code)
    os.chmod(path, 0o755)
    return path


# --- Eval-set carving --------------------------------------------------------

def write_eval_split_utility(base):
    """Writes a helper that carves a fixed held-out FASTA from raw data."""
    code = '''\
"""Carve a fixed held-out eval FASTA from raw data (run once, reuse everywhere).

Everything not selected for eval is written to the train FASTA.
"""
import argparse, random
from pathlib import Path


def split(input_dir, eval_tokens, train_out, eval_out,
          min_len=64, max_len=2048, seed=42, pattern='*_CDS.fasta'):
    random.seed(seed)
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    records = []
    for fp in files:
        current = []
        header = None
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header and current:
                        seq = ''.join(current)
                        if len(seq) >= min_len:
                            records.append((header, seq[:max_len]))
                    header = line
                    current = []
                else:
                    current.append(line)
        if header and current:
            seq = ''.join(current)
            if len(seq) >= min_len:
                records.append((header, seq[:max_len]))
    random.shuffle(records)

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(eval_out).parent.mkdir(parents=True, exist_ok=True)
    e_tokens = 0
    e_count = 0
    with open(eval_out, 'w') as fe, open(train_out, 'w') as ft:
        for header, seq in records:
            if e_tokens < eval_tokens:
                fe.write(header + '\\n')
                for i in range(0, len(seq), 80):
                    fe.write(seq[i:i+80] + '\\n')
                e_tokens += len(seq)
                e_count += 1
            else:
                ft.write(header + '\\n')
                for i in range(0, len(seq), 80):
                    ft.write(seq[i:i+80] + '\\n')
    total = len(records)
    print(f'  eval : {e_count:,} seqs / {e_tokens:,} tokens -> {eval_out}')
    print(f'  train: {total - e_count:,} seqs                -> {train_out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--eval_tokens', type=int, required=True)
    ap.add_argument('--train_out', required=True)
    ap.add_argument('--eval_out', required=True)
    ap.add_argument('--min_len', type=int, default=64)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    split(args.data_dir, args.eval_tokens, args.train_out, args.eval_out,
          args.min_len, args.max_len, args.seed)
'''
    path = base / 'utils' / 'carve_eval_split.py'
    path.write_text(code)
    os.chmod(path, 0o755)
    return path


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Generate scaling-sweep configs')
    ap.add_argument('--experiment_dir', default='scaling_experiment')
    ap.add_argument('--data_total_tokens', type=int, default=3_950_000_000,
                    help='Approximate total tokens available in the dataset.')
    ap.add_argument('--max_seq_len', type=int, default=2048)

    ap.add_argument('--iso_token_points', type=int, default=8,
                    help='Number of N values in the Iso-Token sweep.')
    ap.add_argument('--iso_token_min_params', type=int, default=100_000)
    ap.add_argument('--iso_token_max_params', type=int, default=200_000_000)

    ap.add_argument('--iso_param_model', type=int, default=30_000_000,
                    help='Target param count of the fixed model used in Iso-Param.')
    ap.add_argument('--iso_param_fractions', type=float, nargs='+',
                    default=[0.0625, 0.125, 0.25, 0.5, 1.0])

    ap.add_argument('--iso_flop_budgets', type=str,
                    default='1e15,3e15,1e16,3e16',
                    help='Comma-separated compute budgets in FLOPs.')
    ap.add_argument('--iso_flop_points_per_bucket', type=int, default=5)
    args = ap.parse_args()

    base = Path(args.experiment_dir)
    _mkdirs(base)
    sampling_util = write_sampling_utility(base)
    eval_split_util = write_eval_split_utility(base)

    iso_token_dir = base / 'configs' / 'iso_token'
    iso_param_dir = base / 'configs' / 'iso_param'
    iso_flop_dir = base / 'configs' / 'iso_flop'

    print('=' * 60)
    print('Setting up scaling experiment')
    print('=' * 60)

    print('\n[iso_token] configs')
    iso_token_cfgs = build_iso_token(
        iso_token_dir, args.iso_token_points,
        args.iso_token_min_params, args.iso_token_max_params, args.max_seq_len,
    )
    for c in iso_token_cfgs:
        print(f'  {c["name"]}: {c["params"]:,} params -> {c["file"]}')

    print('\n[iso_param] config')
    iso_param_cfg = build_iso_param(iso_param_dir, args.iso_param_model, args.max_seq_len)
    print(f'  {iso_param_cfg["name"]}: {iso_param_cfg["params"]:,} params '
          f'-> {iso_param_cfg["file"]}')

    print('\n[iso_flop] configs')
    budgets = [float(x) for x in args.iso_flop_budgets.split(',') if x.strip()]
    iso_flop_cfgs = build_iso_flop(
        iso_flop_dir, budgets, args.iso_flop_points_per_bucket,
        args.max_seq_len, args.data_total_tokens,
    )
    for c in iso_flop_cfgs:
        print(f'  [C={c["bucket"]}] {c["name"]}: '
              f'N={c["params"]:,} D={c["tokens"]:,}')

    summary = {
        'args': vars(args),
        'iso_token': iso_token_cfgs,
        'iso_param': iso_param_cfg,
        'iso_flop': iso_flop_cfgs,
        'utilities': {
            'sampling': str(sampling_util),
            'eval_split': str(eval_split_util),
        },
    }
    with open(base / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print('\nSetup complete. Manifest:', base / 'experiment_summary.json')


if __name__ == '__main__':
    main()
