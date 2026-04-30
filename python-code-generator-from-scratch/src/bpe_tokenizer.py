"""
bpe_tokenizer.py — Byte-Pair Encoding tokenizer from scratch
==============================================================
BPE is the tokenizer used by GPT-2, GPT-4, LLaMA, and most modern LLMs.

How it works:
  1. Start with individual characters as the vocabulary
  2. Count all adjacent character pairs in the corpus
  3. Merge the most frequent pair into a new token
  4. Repeat until vocabulary reaches target size

Example:
  Corpus: "for for for if if"
  Step 1: most common pair = ('f','o') → merge to 'fo'
  Step 2: most common pair = ('fo','r') → merge to 'for'
  Now 'for' is a single token instead of 3 characters.

This makes sequences shorter → faster training, better quality.
"""

import re
from collections import Counter, defaultdict
from typing import Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Attributes:
        vocab      : set of all known tokens (characters + merged pairs)
        merges     : ordered list of merge rules learned during training
        char_to_id : token string → integer ID
        id_to_char : integer ID → token string
    """

    # Special tokens
    PAD = "<PAD>"
    UNK = "<UNK>"
    SEP = "<SEP>"
    EOW = "</w>"   # end-of-word marker

    def __init__(self) -> None:
        self.merges:     list[tuple[str, str]] = []
        self.vocab:      set[str]              = set()
        self.char_to_id: dict[str, int]        = {}
        self.id_to_char: dict[int, str]        = {}

    # ────────────────────────────────────────────────────────────────── #
    #  Training: learn merge rules from corpus                           #
    # ────────────────────────────────────────────────────────────────── #

    def train(self, corpus: str, vocab_size: int = 1000) -> None:
        """
        Learn BPE merge rules from a text corpus.

        Args:
            corpus     : All training text concatenated
            vocab_size : Target vocabulary size (merges until we reach this)
        """
        print(f"Training BPE tokenizer (target vocab: {vocab_size})...")

        # Step 1: Split corpus into words, represent each as tuple of chars
        # Each word ends with </w> to mark word boundaries
        word_freqs = Counter(
            word + self.EOW
            for word in re.findall(r'\S+|\n', corpus)
        )
        # Convert each word to a tuple of characters
        word_splits: dict[str, list[str]] = {
            word: list(word) for word in word_freqs
        }

        # Initial vocabulary = all unique characters
        self.vocab = set(
            ch for word in word_splits for ch in word
        ) | {self.PAD, self.UNK, self.SEP}

        # Step 2: Merge until target vocab size
        while len(self.vocab) < vocab_size:
            # Count all adjacent pairs across all words (weighted by frequency)
            pair_counts: Counter = Counter()
            for word, chars in word_splits.items():
                freq = word_freqs[word]
                for i in range(len(chars) - 1):
                    pair_counts[(chars[i], chars[i+1])] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Merge this pair everywhere in the corpus
            merged = best_pair[0] + best_pair[1]
            self.vocab.add(merged)

            for word in word_splits:
                word_splits[word] = self._apply_merge(
                    word_splits[word], best_pair
                )

            if len(self.vocab) % 100 == 0:
                print(f"  Vocab size: {len(self.vocab)}")

        self._build_lookup_tables()
        print(f"BPE training complete: {len(self.vocab)} tokens, "
              f"{len(self.merges)} merge rules")

    @staticmethod
    def _apply_merge(chars: list[str], pair: tuple[str, str]) -> list[str]:
        """Merge all occurrences of pair in a character list."""
        merged = pair[0] + pair[1]
        result = []
        i = 0
        while i < len(chars):
            if i < len(chars)-1 and chars[i] == pair[0] and chars[i+1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(chars[i])
                i += 1
        return result

    def _build_lookup_tables(self) -> None:
        """Build char_to_id and id_to_char from current vocab."""
        specials = [self.PAD, self.UNK, self.SEP]
        regular  = sorted(self.vocab - set(specials))
        all_toks = specials + regular
        self.char_to_id = {t: i for i, t in enumerate(all_toks)}
        self.id_to_char = {i: t for i, t in enumerate(all_toks)}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    # ────────────────────────────────────────────────────────────────── #
    #  Encoding and decoding                                             #
    # ────────────────────────────────────────────────────────────────── #

    def tokenize(self, text: str) -> list[str]:
        """Text → list of BPE token strings (before converting to IDs)."""
        tokens = []
        for word in re.findall(r'\S+|\n', text):
            chars = list(word + self.EOW)
            for merge in self.merges:
                chars = self._apply_merge(chars, merge)
            tokens.extend(chars)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Text → list of integer token IDs."""
        unk = self.char_to_id[self.UNK]
        return [self.char_to_id.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        """Integer token IDs → text."""
        tokens = [self.id_to_char.get(i, self.UNK) for i in ids]
        text   = "".join(tokens)
        text   = text.replace(self.EOW, " ").strip()
        return text


if __name__ == "__main__":
    corpus = open("data/train.jsonl").read()
    tok    = BPETokenizer()
    tok.train(corpus, vocab_size=500)

    sample  = "for item in items:"
    encoded = tok.encode(sample)
    print(f"\nSample : {sample!r}")
    print(f"Tokens : {tok.tokenize(sample)}")
    print(f"IDs    : {encoded}")
