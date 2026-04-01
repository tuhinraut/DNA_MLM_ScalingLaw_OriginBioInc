import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for Masked Language Modeling.

    Computes loss only over masked token positions
    (positions where labels != -100).
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] — contains true token ids at
                    masked positions and -100 everywhere else.

        Returns:
            loss:       scalar loss value
            num_masked: number of masked tokens in this batch (for logging)
        """
        mask = labels != -100

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device,
                                requires_grad=True), 0

        masked_logits = logits[mask]
        masked_labels = labels[mask]

        log_probs = F.log_softmax(masked_logits, dim=-1)
        loss = -log_probs.mean()

        return loss, mask.sum().item()
