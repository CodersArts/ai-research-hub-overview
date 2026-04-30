<div align="center">

# 🔬 AI Research Hub — Build from Scratch

### 6 complete AI systems · No APIs · No pretrained models · Pure code

[![Stars](https://img.shields.io/github/stars/CodersArts?style=flat-square&color=gold)](https://github.com/CodersArts)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**[labs.codersarts.com/ai-research-hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## The philosophy

> Most AI tutorials teach you to *use* existing systems.  
> These modules teach you to *build* them.

When you call `model = GPT2.from_pretrained("gpt2")`, you get output but learn nothing.  
When you implement causal self-attention line by line, you understand everything.

**This hub is for developers who want to understand the internals** — not just the APIs.

---

## 6 Research Modules

| # | Module | What you build | Difficulty | Stars |
|---|--------|----------------|------------|-------|
| 1 | [build-llm-from-scratch](https://github.com/CodersArts/build-llm-from-scratch) | GPT-style language model · 1.8M params | ⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/build-llm-from-scratch?style=flat-square) |
| 2 | [python-code-generator-from-scratch](https://github.com/CodersArts/python-code-generator-from-scratch) | BPE tokenizer · AST validator · Beam search | ⭐⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/python-code-generator-from-scratch?style=flat-square) |
| 3 | [nlp-from-scratch-pytorch](https://github.com/CodersArts/nlp-from-scratch-pytorch) | Word2Vec · LSTM · Sentiment · NER | ⭐⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/nlp-from-scratch-pytorch?style=flat-square) |
| 4 | [speech-recognition-from-scratch](https://github.com/CodersArts/speech-recognition-from-scratch) | FFT · MFCC · CNN · CTC decoder | ⭐⭐⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/speech-recognition-from-scratch?style=flat-square) |
| 5 | [ocr-from-scratch-pytorch](https://github.com/CodersArts/ocr-from-scratch-pytorch) | Otsu · ConnComp · CNN · Word reconstruction | ⭐⭐⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/ocr-from-scratch-pytorch?style=flat-square) |
| 6 | [cnn-vision-from-scratch](https://github.com/CodersArts/cnn-vision-from-scratch) | Conv2D · BatchNorm · ResNet · CIFAR-10 | ⭐⭐⭐⭐ | ![](https://img.shields.io/github/stars/CodersArts/cnn-vision-from-scratch?style=flat-square) |

---

## Recommended learning order

```
Module 1 (Foundation LLM)
    │
    ├── Module 2 (Code Generator)  — upgrades tokenizer + decoding
    │
    └── Module 3 (NLU)             — applies sequence modelling to text tasks
            │
            └── Module 4 (Speech)  — sequence modelling on audio
                    │
                    └── Module 6 (Vision) — CNN fundamentals
                            │
                            └── Module 5 (OCR) — combines vision + text
```

Start at Module 1. Each module builds on concepts from the previous one.

---

## What you need

- Python 3.10+
- PyTorch 2.x (CPU is fine for Modules 1–3, GPU helps for 4–6)
- NumPy, tqdm
- Free Google Colab works for everything

**Total cost: ₹0 / $0**

---

## Who this is for

✅ CS final year students looking for a research-grade project  
✅ Developers tired of black-box APIs who want real understanding  
✅ Researchers building niche / domain-specific AI tools  
✅ Educators who want to teach AI internals, not just tool usage  
✅ Anyone who wants to sell a tiny domain-specific model  

---

## Learn, build, and sell your own tiny models

Each module is also a business opportunity:

| Module | Who buys | What they pay for |
|--------|----------|-------------------|
| Python Code Generator | EdTech platforms, teachers | Personalised Python tutor model |
| NLU Pipeline | HR tools, EdTech | Feedback classifier, quiz generator |
| Speech Recognition | Language apps, IoT | Offline keyword spotter |
| OCR | Fintech, HR | Document digitisation |
| Computer Vision | Retail, healthcare | Custom image classifier |

→ **[Business guide: how to monetise tiny models](https://labs.codersarts.com/monetise)**

---

## Courses and mentorship

| What | Details | Link |
|------|---------|------|
| 🎓 Module 1 course | Build a LLM from scratch · 10 videos | [View →](https://labs.codersarts.com/courses/build-llm) |
| 🎓 Module 2 course | BPE + Beam search + AST | [View →](https://labs.codersarts.com/courses/code-generator) |
| 🎓 Module 3 course | NLP pipeline from scratch | [View →](https://labs.codersarts.com/courses/nlp) |
| 💬 1-on-1 mentorship | Build any module as your final year project | [Book →](https://codersarts.com/mentorship) |
| 📦 Custom datasets | Pre-curated datasets for each module | [Get →](https://labs.codersarts.com/datasets) |

---

## Contributing

Found a better implementation of a component?  
Have a new module idea?  
PRs and issues are very welcome.

---

<div align="center">

**⭐ Star the repos you use — it helps other developers find them**

Built by [Codersarts](https://codersarts.com)  
[Website](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub) · [Courses](https://labs.codersarts.com/courses) · [Mentorship](https://codersarts.com/mentorship)

</div>
