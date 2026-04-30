<div align="center">

# 🔬 Build a Language Model from Scratch

### No OpenAI. No HuggingFace. No pretrained weights. Pure PyTorch.

Train a tiny GPT-style transformer that generates Python code —  
**on your laptop, in 2 hours, for free.**

[![Stars](https://img.shields.io/github/stars/CodersArts/build-llm-from-scratch?style=flat-square&color=gold)](https://github.com/CodersArts/build-llm-from-scratch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com/github/CodersArts/build-llm-from-scratch/blob/main/notebooks/train_colab.ipynb)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 1 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What you will build

A complete language model pipeline — every component written by hand:

```
Prompt:  "read a csv file line by line"

Output:  with open('data.csv', 'r') as f:
             reader = csv.reader(f)
             for row in reader:
                 print(row)
```

No API call. No downloaded model. Your weights. Your code. Your understanding.

---

## Architecture at a glance

```
Input text
    │
    ▼
┌─────────────────────────────────┐
│  Character Tokenizer            │  text → integer IDs
│  vocab_size ≈ 90                │
└─────────────┬───────────────────┘
              │
    ▼
┌─────────────────────────────────┐
│  Token Embedding  (vocab × 128) │  IDs → dense vectors
│  + Positional Embedding         │  adds position info
└─────────────┬───────────────────┘
              │
    ▼
┌─────────────────────────────────┐  ×4 layers
│  Causal Self-Attention          │  each token sees
│  4 heads × 32 dims              │  only past tokens
│  + Feed-Forward (128→512→128)   │
│  + LayerNorm + Residual         │
└─────────────┬───────────────────┘
              │
    ▼
┌─────────────────────────────────┐
│  LM Head  (128 → vocab_size)    │  → next token probs
└─────────────────────────────────┘

Total parameters: ~1.8 million
Training time:    ~30–60 min (free Colab GPU)
Model file size:  ~7 MB
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/build-llm-from-scratch
cd build-llm-from-scratch
pip install -r requirements.txt

# Step 1 — Build tokenizer from dataset
python src/tokenizer.py

# Step 2 — Train the model
python src/train.py

# Step 3 — Generate Python code
python src/generate.py --prompt "for loop over a list"
```

**Expected output after training:**
```
Step    0 | Loss: 4.498  ← random guessing
Step  500 | Loss: 2.103  ← learning structure
Step 1500 | Loss: 1.482  ← learning keywords
Step 3000 | Loss: 0.921  ← learning syntax
Step 5000 | Loss: 0.423  ← ready to use ✓
```

---

## Project structure

```
build-llm-from-scratch/
│
├── src/
│   ├── tokenizer.py        ← Character-level tokenizer (build vocab from data)
│   ├── model.py            ← TinyGPT transformer (attention + feed-forward)
│   ├── train.py            ← Training loop (loss, backprop, optimizer, save)
│   ├── generate.py         ← Inference with temperature + top-k sampling
│   └── dataset.py          ← Data loading, batching, train/val split
│
├── data/
│   └── train.jsonl         ← Your training examples (prompt + code pairs)
│
├── notebooks/
│   └── train_colab.ipynb   ← Run everything on free Google Colab GPU
│
├── assets/
│   └── architecture.png    ← Model architecture diagram
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## What you learn, step by step

| Step | File | Concept |
|------|------|---------|
| 1 | `tokenizer.py` | How text becomes numbers — vocabulary, encoding, decoding |
| 2 | `model.py` | Embeddings, scaled dot-product attention math, residual connections |
| 3 | `train.py` | Cross-entropy loss, backpropagation, AdamW, gradient clipping |
| 4 | `generate.py` | Autoregressive generation, temperature, top-k sampling |
| 5 | `dataset.py` | How to format and batch training data for language models |

---

## Add your own data

The model learns from `data/train.jsonl`. Each line is one example:

```json
{"prompt": "for loop over a list", "code": "for item in items:\n    print(item)"}
{"prompt": "read a json file",      "code": "import json\nwith open('f.json') as f:\n    data = json.load(f)"}
```

Add more examples → better model. Target: **500–2000 examples** for a solid niche model.

**Want a pre-built dataset?** → [labs.codersarts.com/datasets](https://labs.codersarts.com/datasets)

---

## Extend this project

Once the base model trains, here is what to try next:

- **Bigger vocab** — swap character tokenizer for BPE (`Module 2`)
- **More topics** — add Django, NumPy, Pandas examples to your dataset
- **Fine-tune style** — add "beginner", "advanced" tags to control output style
- **Deploy as API** — wrap `generate.py` in a FastAPI endpoint
- **Sell it** — build a teacher tool around this model ([see business guide](https://labs.codersarts.com/monetise))

---

## Requirements

| Package | Version | Why |
|---------|---------|-----|
| Python | 3.10+ | Walrus operator, match statement |
| PyTorch | 2.x | Neural network training |
| NumPy | 1.24+ | Array operations |
| tqdm | 4.x | Training progress bar |

No GPU required. Free Google Colab works perfectly.

---

## Part of the AI Research Hub

This is **Module 1 of 6**. Each module builds a complete AI system from scratch.

| # | Module | Repo |
|---|--------|------|
| **1** | **Foundation LLM** ← you are here | [build-llm-from-scratch](https://github.com/CodersArts/build-llm-from-scratch) |
| 2 | Code Generation Model | [python-code-generator-from-scratch](https://github.com/CodersArts/python-code-generator-from-scratch) |
| 3 | Natural Language Understanding | [nlp-from-scratch-pytorch](https://github.com/CodersArts/nlp-from-scratch-pytorch) |
| 4 | Speech Recognition | [speech-recognition-from-scratch](https://github.com/CodersArts/speech-recognition-from-scratch) |
| 5 | OCR & Document Reader | [ocr-from-scratch-pytorch](https://github.com/CodersArts/ocr-from-scratch-pytorch) |
| 6 | Computer Vision | [cnn-vision-from-scratch](https://github.com/CodersArts/cnn-vision-from-scratch) |

→ **[Full AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

---

## Learn to build this yourself

This repo gives you the code. The course gives you the **deep understanding**.

| What | Details | Link |
|------|---------|------|
| 🎓 Full video course | 10 videos · 8 hours · step-by-step | [View course →](https://labs.codersarts.com/courses/build-llm) |
| 💬 1-on-1 mentorship | Build your own niche model with guidance | [Book session →](https://codersarts.com/mentorship) |
| 📦 Custom dataset | 2,000 Python examples across 20 topics | [Get dataset →](https://labs.codersarts.com/datasets) |

---

## Contributing

Found a bug? Have a cleaner implementation of a component?  
PRs are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<div align="center">

**⭐ Star this repo if it helped you — it helps other developers find it**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub) · [Courses](https://labs.codersarts.com/courses)

</div>
