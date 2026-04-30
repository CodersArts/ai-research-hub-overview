<div align="center">

# 🗣️ NLP Pipeline — From Scratch

### Word2Vec · LSTM · Sentiment · POS Tagging · NER

Build a complete Natural Language Understanding pipeline —  
**no spaCy, no NLTK, no HuggingFace. Pure PyTorch.**

[![Stars](https://img.shields.io/github/stars/CodersArts/nlp-from-scratch-pytorch?style=flat-square&color=gold)](https://github.com/CodersArts/nlp-from-scratch-pytorch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 3 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What you will build

A 5-stage NLU pipeline — each stage builds on the previous one:

```
Stage 1: Word Embeddings (Word2Vec skip-gram from scratch)
          "king" - "man" + "woman" ≈ "queen"  ✓

Stage 2: LSTM classifier
          "This code is terrible" → NEGATIVE  ✓
          "Amazing tutorial!"     → POSITIVE  ✓

Stage 3: Sentiment Analysis (binary + multi-class)
          Review text → [positive / negative / neutral]

Stage 4: POS Tagger
          "Python is powerful" → [NOUN, VERB, ADJ]

Stage 5: Named Entity Recogniser (NER)
          "Google was founded in California" →
          [ORG: Google] [LOC: California]
```

---

## Architecture overview

```
Text → Tokenise → Embed (Word2Vec)
                      │
              ┌───────┴───────────────────┐
              │                           │
         LSTM Encoder               CNN Encoder
              │                           │
        Sentiment                    POS / NER
       Classifier                     Tagger
    (binary/multi)              (sequence labelling)
```

---

## Project structure

```
nlp-from-scratch-pytorch/
├── src/
│   ├── tokenizer.py        ← Simple word tokenizer + vocabulary
│   ├── word2vec.py         ← Skip-gram Word2Vec with negative sampling
│   ├── lstm.py             ← LSTM cell from scratch (gates explained)
│   ├── sentiment.py        ← Sentiment classifier (LSTM + linear)
│   ├── pos_tagger.py       ← POS tagger (BiLSTM + CRF)
│   ├── ner.py              ← NER with BIO tagging scheme
│   └── train.py            ← Train all models
├── data/
│   ├── sentiment.jsonl     ← Sentiment training examples
│   └── ner.jsonl           ← NER training examples
├── notebooks/
│   └── pipeline_demo.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/nlp-from-scratch-pytorch
cd nlp-from-scratch-pytorch
pip install -r requirements.txt

python src/word2vec.py        # train word embeddings
python src/sentiment.py       # train sentiment classifier
python src/pos_tagger.py      # train POS tagger
```

---

## Part of the AI Research Hub

| # | Module | Repo |
|---|--------|------|
| 1 | Foundation LLM | [build-llm-from-scratch](https://github.com/CodersArts/build-llm-from-scratch) |
| 2 | Code Generator | [python-code-generator-from-scratch](https://github.com/CodersArts/python-code-generator-from-scratch) |
| **3** | **NLU Pipeline** ← you are here | [nlp-from-scratch-pytorch](https://github.com/CodersArts/nlp-from-scratch-pytorch) |
| 4 | Speech Recognition | [speech-recognition-from-scratch](https://github.com/CodersArts/speech-recognition-from-scratch) |
| 5 | OCR | [ocr-from-scratch-pytorch](https://github.com/CodersArts/ocr-from-scratch-pytorch) |
| 6 | Computer Vision | [cnn-vision-from-scratch](https://github.com/CodersArts/cnn-vision-from-scratch) |

---

## Learn to build this

| What | Link |
|------|------|
| 🎓 Full course | [labs.codersarts.com/courses/nlp](https://labs.codersarts.com/courses/nlp) |
| 💬 Mentorship | [codersarts.com/mentorship](https://codersarts.com/mentorship) |

---

<div align="center">

**⭐ Star this repo if it helped you**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub)

</div>
