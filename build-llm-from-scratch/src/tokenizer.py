"""
tokenizer.py — Character-level tokenizer for TinyGPT
=====================================================
Converts text ↔ integer token IDs.

Concepts covered:
  - What a vocabulary is and how to build one from data
  - Why we need to encode text as numbers for neural networks
  - How encoding and decoding work as lookup tables

Run this file directly to build and save the tokenizer:
  python src/tokenizer.py
"""

import json
from pathlib import Path


class CharTokenizer:
    """
    Character-level tokenizer.

    Every unique character in your training data gets a unique integer ID.
    Example:
        'a' → 3
        'b' → 4
        '\n' → 1
        ' ' → 0

    Why character-level?
        Simple, no external library needed, works for any language.
        Downside: sequences are longer than word-level tokenizers.
        Module 2 upgrades this to BPE (byte-pair encoding).
    """

    # Special tokens — reserved IDs with fixed meaning
    PAD_TOKEN = "<PAD>"   # ID 0 — used to pad short sequences
    UNK_TOKEN = "<UNK>"   # ID 1 — unknown character not seen during training
    SEP_TOKEN = "<SEP>"   # ID 2 — separates prompt from code in training data

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------ #
    #  Building the vocabulary                                            #
    # ------------------------------------------------------------------ #

    def build_from_texts(self, texts: list[str]) -> None:
        """
        Scan all texts, collect unique characters, assign integer IDs.

        Args:
            texts: list of strings (all prompts + all code from dataset)
        """
        # Special tokens always get the first IDs (0, 1, 2)
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.SEP_TOKEN]

        # Collect every unique character from all texts
        all_chars = sorted(set("".join(texts)))

        # Full vocabulary = specials + regular characters
        vocab = specials + [ch for ch in all_chars if ch not in specials]

        # Build bidirectional lookup tables
        self.char_to_id = {ch: idx for idx, ch in enumerate(vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"Vocabulary built: {self.vocab_size} tokens")
        print(f"  Special tokens : {specials}")
        print(f"  Regular chars  : {len(all_chars)}")

    # ------------------------------------------------------------------ #
    #  Encoding and decoding                                              #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        """
        Text → list of integer token IDs.

        Unknown characters (not seen during training) map to UNK_TOKEN.
        """
        unk_id = self.char_to_id[self.UNK_TOKEN]
        return [self.char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """
        List of integer token IDs → text.
        """
        return "".join(self.id_to_char.get(i, self.UNK_TOKEN) for i in ids)

    @property
    def sep_id(self) -> int:
        return self.char_to_id[self.SEP_TOKEN]

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]

    # ------------------------------------------------------------------ #
    #  Save and load                                                       #
    # ------------------------------------------------------------------ #

    def save(self, path: str = "tokenizer.json") -> None:
        payload = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Tokenizer saved → {path}")

    def load(self, path: str = "tokenizer.json") -> None:
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.vocab_size = len(self.char_to_id)
        print(f"Tokenizer loaded: {self.vocab_size} tokens")


# ------------------------------------------------------------------ #
#  Helper: load dataset and build tokenizer                           #
# ------------------------------------------------------------------ #

def build_from_dataset(jsonl_path: str = "data/train.jsonl") -> CharTokenizer:
    """Read all prompts and code from the JSONL dataset, build tokenizer."""
    texts = []
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {jsonl_path}\n"
            "Add your training examples to data/train.jsonl first.\n"
            "Format: {\"prompt\": \"...\", \"code\": \"...\"}"
        )

    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            texts.append(obj["prompt"])
            texts.append(obj["code"])

    tok = CharTokenizer()
    tok.build_from_texts(texts)
    return tok


# ------------------------------------------------------------------ #
#  Run directly: python src/tokenizer.py                             #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Building tokenizer from data/train.jsonl ...")
    tok = build_from_dataset("data/train.jsonl")
    tok.save("tokenizer.json")

    # Sanity check — encode then decode should return original
    sample = "for i in range(10):\n    print(i)"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)

    print(f"\nSample text : {sample!r}")
    print(f"Encoded IDs : {encoded[:10]} ...")
    print(f"Decoded back: {decoded!r}")

    assert sample == decoded, "Encode/decode roundtrip FAILED"
    print("\nRoundtrip OK ✓")
