import torch
from torch.utils.data import Dataset
import random


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
        """Map a raw DNA string to a list of token ids."""
        tokens = []
        for char in sequence.upper():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['N'])
        return tokens

    def decode(self, token_ids):
        return ''.join(self.id_to_token.get(tid, '?') for tid in token_ids)


class DNASequenceDataset(Dataset):
    """Dataset for DNA sequence MLM training.

    Takes a list of raw DNA strings, tokenizes them, pads to a
    fixed length, and applies MLM masking on the fly.
    """

    def __init__(self, sequences, tokenizer, max_seq_len=512, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        token_ids = self.tokenizer.encode(sequence)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        seq_len = len(token_ids)
        labels = token_ids.copy()
        masked_ids = token_ids.copy()

        
        for i in range(len(masked_ids)):
            if masked_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100
                continue

            if random.random() < self.mask_prob: ## error here. the 15 percent masking is made to be 85% because of the incorrect if statements. changed the > to <
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequences(filepaths, min_len, max_seq_len):
    """Load sequences from one or more FASTA files.
    
    Args:
        filepaths: Single path (str/Path) or list of paths to FASTA files
        min_len: Minimum sequence length to keep
        max_seq_len: Maximum sequence length (truncate if longer)
    
    Returns:
        List of filtered sequences
    """
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    
    all_sequences = []
    total_skipped = 0
    total_truncated = 0
    
    for filepath in filepaths:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        sequences = []
        current_seq = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append(''.join(current_seq))
        
        skipped_count = 0
        trunc_count = 0
        filtered_sequences = []
        
        for seq in sequences:
            seq_len = len(seq)
            if seq_len < min_len:
                skipped_count += 1
                continue
            if seq_len > max_seq_len:
                trunc_count += 1
                filtered_sequences.append(seq[:max_seq_len])
                continue
            filtered_sequences.append(seq)
        
        all_sequences.extend(filtered_sequences)
        total_skipped += skipped_count
        total_truncated += trunc_count
        print(f"  Loaded {filepath.name}: {len(filtered_sequences):,} sequences "
              f"(skipped {skipped_count}, truncated {trunc_count})")
    
    print(f"Total sequences loaded: {len(all_sequences):,}")
    return all_sequences
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


def generate_synthetic_sequences(num_sequences, min_len=100, max_seq_len=1000, seed=42):
    """Generate random DNA sequences for testing / debugging."""
    rng = random.Random(seed)
    nucleotides = 'ACGT'
    sequences = []
    for _ in range(num_sequences):
        length = rng.randint(min_len, max_seq_len)
        seq = ''.join(rng.choice(nucleotides) for _ in range(length))
        sequences.append(seq)
    return sequences