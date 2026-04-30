"""
dataset.py — Dataset utilities
================================
Helpers for loading, validating, and expanding your training data.

Run:
  python src/dataset.py        # validates data/train.jsonl and shows stats
"""
import json
from pathlib import Path


def validate(path="data/train.jsonl"):
    """Check dataset format and print statistics."""
    examples, errors = [], []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                assert "prompt" in obj and "code" in obj, "Missing 'prompt' or 'code' key"
                examples.append(obj)
            except Exception as e:
                errors.append(f"  Line {i}: {e}")

    print(f"Dataset: {path}")
    print(f"  Total examples : {len(examples)}")
    print(f"  Errors         : {len(errors)}")
    if errors:
        print("\n".join(errors))

    avg_prompt = sum(len(e["prompt"]) for e in examples) / max(len(examples), 1)
    avg_code   = sum(len(e["code"])   for e in examples) / max(len(examples), 1)
    print(f"  Avg prompt len : {avg_prompt:.0f} chars")
    print(f"  Avg code len   : {avg_code:.0f} chars")
    return examples


if __name__ == "__main__":
    validate()
