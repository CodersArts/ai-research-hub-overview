"""
word2vec.py — Word2Vec (skip-gram) from scratch
================================================
Word2Vec learns dense vector representations of words such that
semantically similar words have similar vectors.

Skip-gram: given a word, predict its surrounding context words.
Example: given "Python", predict ["learn", "language", "is", "great"]

Key idea — Negative Sampling:
  Instead of computing softmax over entire vocabulary (slow),
  we train a binary classifier:
    - Positive: (center_word, context_word) → label 1
    - Negative: (center_word, random_word)  → label 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np


class Word2Vec(nn.Module):
    """
    Skip-gram Word2Vec with negative sampling.

    Two embedding tables:
      center_emb  : embeddings for center (input) words
      context_emb : embeddings for context (output) words

    After training, use center_emb as your word vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        # Two separate embedding tables (standard Word2Vec design)
        self.center_emb  = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)

        # Initialise with small uniform values
        nn.init.uniform_(self.center_emb.weight,  -0.1, 0.1)
        nn.init.uniform_(self.context_emb.weight, -0.1, 0.1)

    def forward(self,
                center: torch.Tensor,    # (batch,)
                context: torch.Tensor,   # (batch,)
                negative: torch.Tensor,  # (batch, n_neg)
                ) -> torch.Tensor:
        """
        Compute negative sampling loss.

        Positive pairs: center + true context → high dot product
        Negative pairs: center + random words → low dot product
        """
        c_emb = self.center_emb(center)            # (B, D)
        o_emb = self.context_emb(context)           # (B, D)
        n_emb = self.context_emb(negative)          # (B, n_neg, D)

        # Positive score: how much does center predict context?
        pos_score = torch.sum(c_emb * o_emb, dim=1)         # (B,)
        pos_loss  = F.logsigmoid(pos_score)

        # Negative score: center should NOT predict random words
        neg_score = torch.bmm(n_emb, c_emb.unsqueeze(2)).squeeze(2)  # (B, n_neg)
        neg_loss  = F.logsigmoid(-neg_score).sum(dim=1)              # (B,)

        return -(pos_loss + neg_loss).mean()

    def get_vector(self, word_id: int) -> np.ndarray:
        """Get the word vector for a given word ID."""
        with torch.no_grad():
            return self.center_emb(torch.tensor(word_id)).numpy()

    def most_similar(self,
                     word_id: int,
                     vocab: dict[int, str],
                     top_k: int = 5) -> list[tuple[str, float]]:
        """Find the most similar words by cosine similarity."""
        with torch.no_grad():
            all_emb = F.normalize(self.center_emb.weight, dim=1)
            query   = all_emb[word_id]
            sims    = (all_emb @ query).cpu().numpy()

        # Get top-k (excluding the query word itself)
        top_ids = np.argsort(sims)[::-1][1:top_k+1]
        return [(vocab.get(int(i), "?"), float(sims[i])) for i in top_ids]


if __name__ == "__main__":
    print("Word2Vec demo — run src/train.py to train on real data")
    model = Word2Vec(vocab_size=1000, embed_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
