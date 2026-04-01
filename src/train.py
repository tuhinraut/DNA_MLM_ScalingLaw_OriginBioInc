import argparse
import torch
from torch.utils.data import DataLoader, random_split
import math
import json
import os
from pathlib import Path

from model import DNATransformerMLM, VOCAB_SIZE
from dataset import DNASequenceDataset, DNATokenizer, load_sequences, generate_synthetic_sequences
from loss import MLMCrossEntropyLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='DNA MLM Training for Scaling Laws Investigation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config JSON')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (default: config filename stem)')
    parser.add_argument('--data_path', type=str, nargs='+', default=None,
                        help='Path(s) to FASTA file(s). Can specify multiple files.')
    parser.add_argument('--min_seq_len', type=int, default=64,
                        help='Minimum sequence length to keep (for FASTA loading)')
    parser.add_argument('--num_synthetic', type=int, default=10000,
                        help='Number of synthetic sequences when --data_path is not set')
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max optimiser steps (overrides num_epochs)')
    parser.add_argument('--eval_every', type=int, default=500,
                        help='Evaluate every N optimiser steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model_config(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warm-up."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def estimate_flops(num_params, num_tokens):
    """Rough FLOPs estimate: C ≈ 6 * N * D."""
    return 6 * num_params * num_tokens


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, eval_loader, loss_fn, device, args):
    """Evaluate model on a dataset.

    TODO: Implement this function.
    - Set model to eval mode
    - Iterate over eval_loader with torch.no_grad()
    - Compute loss on each batch using loss_fn(logits, labels)
    - Track total loss (sum of per-token losses) and total masked tokens
    - Compute average loss = total_weighted_loss / total_masked_tokens
    - Compute perplexity = exp(average_loss)
    - Set model back to train mode
    - Return a dict: {'loss': ..., 'perplexity': ...}

    Hint: loss_fn returns (loss, num_masked).  To accumulate properly,
    multiply the returned loss by num_masked before summing.
    """

    total_loss = 0
    total_masked_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            logits = model(input_ids, attention_mask)
            loss, num_masked = loss_fn(logits, labels)
            total_loss += loss.item() * num_masked
            total_masked_tokens += num_masked
    model.train()
    average_loss = total_loss / total_masked_tokens if total_masked_tokens > 0 else 0
    perplexity = math.exp(average_loss) if average_loss > 0 else float('inf')
    return {'loss': average_loss, 'perplexity': perplexity}

# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)

    run_name = args.run_name or Path(args.config).stem
    config = load_model_config(args.config)

    tokenizer = DNATokenizer()

    if args.data_path is not None:
        print(f"Loading data from {len(args.data_path)} file(s):")
        for path in args.data_path:
            print(f"  - {path}")
        sequences = load_sequences(args.data_path, min_len=args.min_seq_len, max_seq_len=args.max_seq_len)
    else:
        sequences = generate_synthetic_sequences(args.num_synthetic)
        print(f"Using {len(sequences)} synthetic sequences")

    dataset = DNASequenceDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mask_prob=args.mask_prob,
    )

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---- Model ----
    model = DNATransformerMLM(
        vocab_size=VOCAB_SIZE,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        max_seq_len=args.max_seq_len,
        dropout=config.get('dropout', 0.1),
    ).to(args.device)

    num_params = model.count_parameters()
    print(f"Run        : {run_name}")
    print(f"Parameters : {num_params:,}")

    # ---- Loss / Optimizer / Scheduler ----
    loss_fn = MLMCrossEntropyLoss(vocab_size=VOCAB_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = args.max_steps or (steps_per_epoch * args.num_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_entries = []
    tokens_seen = 0
    # ---- Training Loop ----
    # TODO: Implement the training loop.
    #
    # The loop should:
    #   - Iterate over train_loader for args.num_epochs epochs
    #   - Forward pass: logits = model(input_ids, attention_mask)
    #   - Compute loss via loss_fn(logits, labels)
    #   - Backward pass with gradient accumulation over
    #     args.gradient_accumulation_steps micro-steps
    #   - Clip gradients, step optimizer and scheduler at accumulation
    #     boundaries
    #   - Track tokens_seen (real tokens only, not padding)
    #   - Every args.eval_every optimiser steps, call evaluate() and
    #     append a log entry with: step, train_loss, eval_loss,
    #     eval_perplexity, tokens_seen, num_parameters, flops,
    #     learning_rate
    #   - Save a checkpoint when eval loss improves
    #   - Stop early if global_step reaches total_steps
    best_eval_loss = float('inf')
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            logits = model(input_ids, attention_mask)
            loss, num_masked = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            real_tokens = attention_mask.sum().item()
            tokens_seen += real_tokens
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break

            if global_step % args.eval_every == 0:
                eval_results = evaluate(model, eval_loader, loss_fn, args.device, args)
                log_entries.append({
                    'step': tokens_seen,
                    'train_loss': loss.item(),
                    'eval_loss': eval_results['loss'],
                    'eval_perplexity': eval_results['perplexity'],
                    'tokens_seen': tokens_seen,
                    'num_parameters': num_params,
                    'flops': estimate_flops(num_params, tokens_seen),
                    'learning_rate': scheduler.get_last_lr()[0],
                })
                if eval_results['loss'] < best_eval_loss:
                    best_eval_loss = eval_results['loss']
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_{run_name}.pth'))   
                    print(f"Model saved to {os.path.join(args.save_dir, f'best_model_{run_name}.pth')}")    
                    print(f"New best eval loss: {best_eval_loss}")

    # ---- Save final log ----
    log_path = os.path.join(args.log_dir, f'training_log_{run_name}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'run_name': run_name,
            'config': config,
            'num_parameters': num_params,
            'final_tokens_seen': tokens_seen,
            'final_flops': estimate_flops(num_params, tokens_seen),
            'log': log_entries,
        }, f, indent=2)

    print(f"Training complete. Log saved to {log_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
