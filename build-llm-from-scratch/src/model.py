"""
model.py — TinyGPT transformer architecture
============================================
The complete neural network — every layer written from scratch.

Concepts covered:
  - Token and positional embeddings
  - Scaled dot-product attention (the key innovation of transformers)
  - Causal masking (why the model can only look backwards)
  - Multi-head attention
  - Feed-forward network with GELU activation
  - LayerNorm and residual connections
  - Weight tying (embedding ↔ output projection)

Architecture diagram:
  Token IDs → Embedding → [Attention + FFN] × N layers → LM Head → logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    "Causal" means token at position t can only attend to positions 0..t.
    This forces the model to predict the NEXT token using only PAST tokens —
    which is exactly what we want for generation.

    The attention formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) · V

    Where:
        Q = query (what am I looking for?)
        K = key   (what do I contain?)
        V = value (what do I return if you pick me?)
        d_head = dimension per head (normalises dot products)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads  # dimension per head
        self.d_model = d_model

        # Single matrix projects input to Q, K, V simultaneously (efficient)
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, d_model

        # Project to Q, K, V and split
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        # Reshape for multi-head: (B, T, C) → (B, n_heads, T, d_head)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        att   = (q @ k.transpose(-2, -1)) / scale    # (B, H, T, T)

        # Causal mask: upper triangle = -∞ → softmax gives 0
        # This prevents attending to future positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values
        out = att @ v                                  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to each position after attention.
    Expands to 4× width then projects back.
    GELU activation (smoother than ReLU, works better in practice).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = self-attention + feed-forward.

    Both use:
    - Pre-norm (LayerNorm before the operation, more stable than post-norm)
    - Residual connection (x = x + operation(x)) — prevents vanishing gradients
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ff(self.ln2(x))     # feed-forward with residual
        return x


class TinyGPT(nn.Module):
    """
    TinyGPT — Minimal GPT-style language model.

    Same architecture as GPT-2 but much smaller (~1.8M params vs 117M).
    Small enough to train on a laptop; large enough to learn Python patterns.

    Args:
        vocab_size  : Number of unique tokens (from tokenizer)
        d_model     : Embedding dimension — width of the model
        n_heads     : Attention heads (must divide d_model evenly)
        n_layers    : Transformer blocks — depth of the model
        ctx_len     : Max tokens the model sees at once (context window)
        dropout     : Regularisation rate
    """

    def __init__(
        self,
        vocab_size : int,
        d_model    : int   = 128,
        n_heads    : int   = 4,
        n_layers   : int   = 4,
        ctx_len    : int   = 256,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.ctx_len = ctx_len

        # Two learned embedding tables
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # token meanings
        self.pos_emb = nn.Embedding(ctx_len, d_model)       # position info

        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f    = nn.LayerNorm(d_model)

        # Output projection: d_model → vocab_size (predicts next token)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection
        # Halves parameters, often improves quality
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Small initialisation for stable training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : Token IDs, shape (batch_size, seq_len)

        Returns:
            logits : Unnormalised scores, shape (batch_size, seq_len, vocab_size)
                     logits[b, t, :] = score for each token being at position t+1
        """
        B, T = x.shape
        assert T <= self.ctx_len, f"Seq len {T} > context window {self.ctx_len}"

        positions = torch.arange(T, device=x.device)              # [0, 1, ..., T-1]
        x = self.drop(self.tok_emb(x) + self.pos_emb(positions))  # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)                                     # (B, T, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
#  Run directly to verify architecture                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    vocab_size = 90
    model = TinyGPT(vocab_size=vocab_size)
    print(f"TinyGPT — {model.count_params():,} parameters")

    # Test forward pass
    x      = torch.randint(0, vocab_size, (2, 64))   # batch=2, seq=64
    logits = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {logits.shape}")
    print("Architecture OK ✓")
