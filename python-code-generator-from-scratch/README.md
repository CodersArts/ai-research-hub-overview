<div align="center">

# 🧠 Python Code Generator — From Scratch

### BPE Tokenizer · Grammar-Aware Generation · AST Validation

Train a model that writes syntactically correct Python — **no Copilot, no APIs.**

[![Stars](https://img.shields.io/github/stars/CodersArts/python-code-generator-from-scratch?style=flat-square&color=gold)](https://github.com/CodersArts/python-code-generator-from-scratch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=google-colab)](https://colab.research.google.com/github/CodersArts/python-code-generator-from-scratch/blob/main/notebooks/train_colab.ipynb)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 2 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What makes this different from Module 1

Module 1 built a character-level LLM. This module upgrades every component:

| Component | Module 1 | Module 2 (this) |
|-----------|---------|----------------|
| Tokenizer | Character-level | **BPE** (byte-pair encoding) |
| Vocabulary | ~90 chars | **1,000–5,000 subwords** |
| Validation | None | **AST syntax check** |
| Decoding | Temperature sampling | **Beam search** |
| Output | Sometimes invalid Python | **Always valid Python** |

---

## What you will build

```
Prompt:  "sort a dictionary by value"

Output:  sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1]))

AST check: ✓ Valid Python syntax
```

---

## Key concepts you implement from scratch

### 1. Byte-Pair Encoding (BPE)
The same algorithm used by GPT-2, GPT-4, and LLaMA.
Starts with characters, merges the most frequent pairs repeatedly:

```
Step 0:  'f' 'o' 'r' ' ' 'i' 't' 'e' 'm' ...   (characters)
Step 1:  'fo' 'r' ' ' 'it' 'em' ...              (merge 'f'+'o')
Step 2:  'for' ' ' 'item' ...                     (merge 'fo'+'r')
Step 100: 'for' 'item' 'in' 'range' '(' ...       (full words)
```

### 2. AST Validation
Every generated snippet is parsed with Python's built-in `ast` module.
Invalid syntax → retry with different sampling temperature.

### 3. Beam Search
Instead of sampling one token at a time, beam search tracks the **k best**
partial sequences simultaneously and returns the highest-probability complete one.

---

## Project structure

```
python-code-generator-from-scratch/
├── src/
│   ├── bpe_tokenizer.py    ← BPE from scratch (merge rules, encode, decode)
│   ├── model.py            ← Seq2seq transformer (encoder + decoder)
│   ├── train.py            ← Training with BPE vocab
│   ├── beam_search.py      ← Beam search decoder
│   ├── ast_validator.py    ← Syntax checker using Python ast module
│   └── generate.py         ← Full pipeline: prompt → BPE → model → validate
├── data/
│   ├── train.jsonl         ← Training examples
│   └── build_vocab.py      ← Learn BPE merge rules from data
├── notebooks/
│   └── train_colab.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/python-code-generator-from-scratch
cd python-code-generator-from-scratch
pip install -r requirements.txt

python data/build_vocab.py          # learn BPE merge rules
python src/train.py                 # train the model
python src/generate.py --prompt "sort a list of dicts by key"
```

---

## Part of the AI Research Hub

| # | Module | Repo |
|---|--------|------|
| 1 | Foundation LLM | [build-llm-from-scratch](https://github.com/CodersArts/build-llm-from-scratch) |
| **2** | **Code Generator** ← you are here | [python-code-generator-from-scratch](https://github.com/CodersArts/python-code-generator-from-scratch) |
| 3 | NLU Pipeline | [nlp-from-scratch-pytorch](https://github.com/CodersArts/nlp-from-scratch-pytorch) |
| 4 | Speech Recognition | [speech-recognition-from-scratch](https://github.com/CodersArts/speech-recognition-from-scratch) |
| 5 | OCR | [ocr-from-scratch-pytorch](https://github.com/CodersArts/ocr-from-scratch-pytorch) |
| 6 | Computer Vision | [cnn-vision-from-scratch](https://github.com/CodersArts/cnn-vision-from-scratch) |

---

## Learn to build this

| What | Link |
|------|------|
| 🎓 Full course (BPE + beam search + AST) | [labs.codersarts.com/courses/code-generator](https://labs.codersarts.com/courses/code-generator) |
| 💬 1-on-1 mentorship | [codersarts.com/mentorship](https://codersarts.com/mentorship) |

---

<div align="center">

**⭐ Star this repo if it helped you**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub)

</div>
