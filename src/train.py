"""Training driver for DNA MLM scaling-law experiments.

Supports:
- gradient accumulation & clipping at accumulation boundaries
- cosine LR with linear warm-up (warmup fraction scales with total steps)
- a *fixed* held-out eval set (so loss is comparable across data sizes)
- periodic eval + a final eval after the last optimiser step
- JSON logging of step, losses, perplexity, tokens-seen, params, FLOPs, LR
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import DNATransformerMLM, VOCAB_SIZE
from dataset import (
    DNASequenceDataset,
    DNATokenizer,
    load_sequences,
    generate_synthetic_sequences,
)
from loss import MLMCrossEntropyLoss


# --- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DNA MLM trainer for scaling-law sweeps')
    p.add_argument('--config', type=str, required=True, help='Model architecture JSON')
    p.add_argument('--run_name', type=str, default=None)

    # Data
    p.add_argument('--data_path', type=str, nargs='+', default=None,
                   help='Training FASTA(s). If omitted, synthetic data is used.')
    p.add_argument('--eval_data_path', type=str, nargs='+', default=None,
                   help='Optional FASTA(s) for a fixed held-out eval set. '
                        'If omitted, a 10% split of --data_path is used.')
    p.add_argument('--num_synthetic', type=int, default=10000)
    p.add_argument('--min_seq_len', type=int, default=64)
    p.add_argument('--max_seq_len', type=int, default=2048)

    # Optimisation
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--warmup_frac', type=float, default=0.05,
                   help='Warm-up fraction of total optimiser steps.')
    p.add_argument('--max_steps', type=int, default=None,
                   help='If set, overrides num_epochs (counted in optimiser steps).')
    p.add_argument('--mask_prob', type=float, default=0.15)

    # Bookkeeping
    p.add_argument('--eval_every', type=int, default=500,
                   help='Evaluate every N optimiser steps (0 = only final eval).')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no_progress', action='store_true',
                   help='Disable tqdm progress bars (useful for clean CI logs).')
    return p.parse_args()


def load_model_config(path):
    with open(path) as f:
        return json.load(f)


# --- Schedule & FLOPs --------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def estimate_flops(num_params, num_tokens):
    return 6 * num_params * num_tokens


# --- Data --------------------------------------------------------------------

def _build_datasets(args, tokenizer):
    """Returns (train_dataset, eval_dataset).

    If --eval_data_path is given, use it as a fixed eval set.
    Otherwise carve a 10% random slice from the training data (seeded)."""
    if args.data_path is not None:
        train_seqs = load_sequences(
            args.data_path, min_len=args.min_seq_len, max_seq_len=args.max_seq_len)
        if not train_seqs:
            raise RuntimeError(f'No sequences loaded from {args.data_path}')
    else:
        train_seqs = generate_synthetic_sequences(
            args.num_synthetic, max_seq_len=args.max_seq_len, seed=args.seed)

    if args.eval_data_path is not None:
        eval_seqs = load_sequences(
            args.eval_data_path, min_len=args.min_seq_len, max_seq_len=args.max_seq_len)
        if not eval_seqs:
            raise RuntimeError(f'No sequences loaded from {args.eval_data_path}')
        train_ds = DNASequenceDataset(train_seqs, tokenizer, args.max_seq_len, args.mask_prob)
        eval_ds = DNASequenceDataset(eval_seqs, tokenizer, args.max_seq_len, args.mask_prob)
        return train_ds, eval_ds

    full = DNASequenceDataset(train_seqs, tokenizer, args.max_seq_len, args.mask_prob)
    n = len(full)
    eval_n = max(1, int(0.1 * n))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    eval_idx, train_idx = perm[:eval_n], perm[eval_n:]
    return Subset(full, train_idx), Subset(full, eval_idx)


# --- Evaluation --------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_loader, loss_fn, device, show_progress=False):
    model.eval()
    total_loss = 0.0
    total_masked = 0
    it = tqdm(eval_loader, desc='eval', leave=False, disable=not show_progress)
    for batch in it:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        logits = model(input_ids, attention_mask)
        loss, num_masked = loss_fn(logits, labels)
        if num_masked > 0:
            total_loss += loss.item() * num_masked
            total_masked += num_masked
    model.train()
    avg = total_loss / total_masked if total_masked > 0 else 0.0
    ppl = math.exp(avg) if avg > 0 else float('inf')
    return {'loss': avg, 'perplexity': ppl, 'num_masked': total_masked}


# --- Training ----------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == 'cuda'
    show = not args.no_progress

    run_name = args.run_name or Path(args.config).stem
    config = load_model_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    tokenizer = DNATokenizer()
    train_ds, eval_ds = _build_datasets(args, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_cuda,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda,
    )

    model = DNATransformerMLM(
        vocab_size=VOCAB_SIZE,
        d_model=config['d_model'], n_heads=config['n_heads'],
        d_ff=config['d_ff'], n_layers=config['n_layers'],
        max_seq_len=args.max_seq_len, dropout=config.get('dropout', 0.1),
    ).to(device)
    num_params = model.count_parameters()

    loss_fn = MLMCrossEntropyLoss(vocab_size=VOCAB_SIZE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    micro_per_opt = max(1, args.gradient_accumulation_steps)
    opt_steps_per_epoch = max(1, len(train_loader) // micro_per_opt)
    total_opt_steps = args.max_steps or (opt_steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(args.warmup_frac * total_opt_steps))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    print(f'[{run_name}] device={device} params={num_params:,} '
          f'train={len(train_ds):,} eval={len(eval_ds):,} '
          f'opt_steps={total_opt_steps:,} warmup={warmup_steps:,} '
          f'effective_bs={args.batch_size * micro_per_opt}')

    log_entries = []
    tokens_seen = 0
    micro_step = 0
    global_step = 0
    best_eval_loss = float('inf')
    best_ckpt_path = os.path.join(args.save_dir, f'best_model_{run_name}.pth')
    t0 = time.time()

    pbar = tqdm(total=total_opt_steps, desc=f'train[{run_name}]',
                unit='step', disable=not show)

    def do_eval(label):
        res = evaluate(model, eval_loader, loss_fn, device, show_progress=show)
        entry = {
            'step': global_step,
            'tokens_seen': tokens_seen,
            'train_loss': entry_train_loss,
            'eval_loss': res['loss'],
            'eval_perplexity': res['perplexity'],
            'num_parameters': num_params,
            'flops': estimate_flops(num_params, tokens_seen),
            'learning_rate': scheduler.get_last_lr()[0],
            'phase': label,
        }
        log_entries.append(entry)
        nonlocal best_eval_loss
        if res['loss'] < best_eval_loss:
            best_eval_loss = res['loss']
            torch.save(model.state_dict(), best_ckpt_path)
            tqdm.write(f'[{run_name}] step={global_step} tokens={tokens_seen:,} '
                       f'eval_loss={res["loss"]:.4f} ppl={res["perplexity"]:.2f}  '
                       f'[checkpoint saved]')
        else:
            tqdm.write(f'[{run_name}] step={global_step} tokens={tokens_seen:,} '
                       f'eval_loss={res["loss"]:.4f} ppl={res["perplexity"]:.2f}')

    model.train()
    optimizer.zero_grad()
    entry_train_loss = float('nan')
    stopped = False

    for epoch in range(args.num_epochs):
        if stopped:
            break
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss, _ = loss_fn(logits, labels)
            (loss / micro_per_opt).backward()

            tokens_seen += int(attention_mask.sum().item())
            micro_step += 1
            entry_train_loss = loss.item()

            if micro_step % micro_per_opt == 0:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f'{entry_train_loss:.3f}',
                                 lr=f'{scheduler.get_last_lr()[0]:.2e}',
                                 tok=f'{tokens_seen/1e6:.1f}M')

                if args.eval_every and global_step % args.eval_every == 0:
                    do_eval('periodic')

                if args.max_steps and global_step >= args.max_steps:
                    stopped = True
                    break

    pbar.close()
    # Final eval after the loop (catches the tail if total_steps % eval_every != 0).
    do_eval('final')
    wall = time.time() - t0

    log_path = os.path.join(args.log_dir, f'training_log_{run_name}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'run_name': run_name,
            'config': config,
            'num_parameters': num_params,
            'final_tokens_seen': tokens_seen,
            'final_flops': estimate_flops(num_params, tokens_seen),
            'final_eval_loss': log_entries[-1]['eval_loss'] if log_entries else None,
            'best_eval_loss': best_eval_loss if best_eval_loss != float('inf') else None,
            'total_opt_steps': global_step,
            'wall_seconds': wall,
            'log': log_entries,
        }, f, indent=2)

    print(f'[{run_name}] done in {wall:.1f}s '
          f'final_eval_loss={log_entries[-1]["eval_loss"]:.4f} '
          f'best={best_eval_loss:.4f} -> {log_path}')


if __name__ == '__main__':
    train(parse_args())
