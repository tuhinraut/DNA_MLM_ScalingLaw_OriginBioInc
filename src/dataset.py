"""Dataset, tokenizer, and FASTA loader for DNA MLM."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DNATokenizer:
    def __init__(self):
        self.vocab = {
            '[PAD]': 0, '[MASK]': 1,
            'A': 2, 'C': 3, 'G': 4, 'T': 5, 'N': 6,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['[PAD]']
        self.mask_token_id = self.vocab['[MASK]']

    def encode(self, sequence):
        out = []
        for char in sequence.upper():
            out.append(self.vocab.get(char, self.vocab['N']))
        return out

    def decode(self, token_ids):
        return ''.join(self.id_to_token.get(tid, '?') for tid in token_ids)


class DNASequenceDataset(Dataset):
    """Tokenises DNA strings, pads to max_seq_len, applies MLM masking on-the-fly."""

    def __init__(self, sequences, tokenizer, max_seq_len=512, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.encode(self.sequences[idx])
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        seq_len = len(token_ids)
        labels = token_ids.copy()
        masked_ids = token_ids.copy()

        for i in range(seq_len):
            if masked_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100
                continue
            if random.random() < self.mask_prob:
                masked_ids[i] = self.tokenizer.mask_token_id
            else:
                labels[i] = -100

        pad_len = self.max_seq_len - seq_len
        masked_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        return {
            'input_ids': torch.tensor(masked_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


def _parse_fasta(filepath):
    """Yield sequences from a FASTA file, joining multi-line records."""
    current = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current:
                    yield ''.join(current)
                    current = []
            else:
                current.append(line)
        if current:
            yield ''.join(current)


def load_sequences(filepaths, min_len=64, max_seq_len=2048):
    """Load sequences from one or more FASTA files, filtered by length.

    Sequences shorter than min_len are dropped; longer than max_seq_len are truncated.
    """
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]

    out = []
    for fp in filepaths:
        fp = Path(fp)
        if not fp.exists():
            continue
        for seq in _parse_fasta(fp):
            if len(seq) < min_len:
                continue
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            out.append(seq)
    return out


def generate_synthetic_sequences(num_sequences, min_len=64, max_seq_len=1000, seed=42):
    """Random DNA strings for CPU debugging."""
    rng = random.Random(seed)
    nucleotides = 'ACGT'
    lo = min(min_len, max_seq_len)
    hi = max(lo, max_seq_len)
    return [
        ''.join(rng.choice(nucleotides) for _ in range(rng.randint(lo, hi)))
        for _ in range(num_sequences)
    ]
