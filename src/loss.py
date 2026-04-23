"""MLM cross-entropy: only masked positions (labels != -100) contribute."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMCrossEntropyLoss(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        """Returns (scalar_loss, num_masked_tokens)."""
        mask = labels != -100
        num_masked = int(mask.sum().item())
        if num_masked == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0
        return F.cross_entropy(logits[mask], labels[mask]), num_masked
