import torch
import torch.nn as nn
import math


VOCAB = {
    '[PAD]': 0,
    '[MASK]': 1,
    'A': 2,
    'C': 3,
    'G': 4,
    'T': 5,
    'N': 6,
}

VOCAB_SIZE = len(VOCAB)
PAD_TOKEN_ID = VOCAB['[PAD]']
MASK_TOKEN_ID = VOCAB['[MASK]']


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Split x into two halves along the last dim and rotate."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(self.head_dim) ## divided by the head_dim to scale the attention weights

        if mask is not None:
            attn_weights = attn_weights.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf')
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Pre-LN Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        x = self.attn_norm(x + self.attn(x, mask))
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full MLM Transformer
# ---------------------------------------------------------------------------

class DNATransformerMLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model,
                                            padding_idx=PAD_TOKEN_ID)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        TODO: Implement proper weight initialization.
        - Embedding: normal distribution with std = 0.02
        - Linear layers: Xavier uniform
        - LayerNorm: weight = 1.0, bias = 0.0
        - MLM head bias: zero
        """
        for name, param in self.named_parameters():
            if 'token_embedding' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'LayerNorm' in name or 'norm' in name:
                nn.init.ones_(param) if 'weight' in name else nn.init.zeros_(param)
            elif 'mlm_head' in name and 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)  ##  MISSING
        logits = self.mlm_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
