<div align="center">

# 📄 OCR — From Scratch

### Image Processing · Otsu Binarization · CNN · Character Recognition

Build a complete Optical Character Recognition pipeline — **no Tesseract, no OpenCV ML, no cloud vision APIs.**

[![Stars](https://img.shields.io/github/stars/CodersArts/ocr-from-scratch-pytorch?style=flat-square&color=gold)](https://github.com/CodersArts/ocr-from-scratch-pytorch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 5 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What you will build

A complete pipeline from raw input to intelligent output — every layer hand-written.

---

## Key concepts you implement from scratch

- Reading PNG/BMP images as raw pixel arrays
- Otsu's thresholding for binarization
- Connected component labelling (BFS)
- Character segmentation and bounding boxes
- CNN character recogniser (A-Z + 0-9)
- Word and line reconstruction

---

## Project structure

```
ocr-from-scratch-pytorch/
├── src/
│   ├── image_utils.py    ← Load images as NumPy arrays, binarize
│   ├── otsu.py           ← Otsu thresholding from scratch
│   ├── connected.py      ← Connected component labelling (BFS/DFS)
│   ├── segmenter.py      ← Character bounding box detection
│   ├── cnn_recogniser.py ← CNN for A-Z + 0-9 recognition
│   ├── line_builder.py   ← Reconstruct words and lines from characters
│   ├── ocr.py            ← Full pipeline: image → text
├── data/
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/ocr-from-scratch-pytorch
cd ocr-from-scratch-pytorch
pip install -r requirements.txt

python src/image_utils.py     # load and binarize an image\npython src/segmenter.py      # detect character regions\npython src/train_cnn.py      # train character recogniser\npython src/ocr.py --image sample.png
```

---

## Part of the AI Research Hub

| # | Module | Repo |
|---|--------|------|
| 1 | Foundation LLM | [build-llm-from-scratch](https://github.com/CodersArts/build-llm-from-scratch) |
| 2 | Code Generator | [python-code-generator-from-scratch](https://github.com/CodersArts/python-code-generator-from-scratch) |
| 3 | NLU Pipeline | [nlp-from-scratch-pytorch](https://github.com/CodersArts/nlp-from-scratch-pytorch) |
| 4 | Speech Recognition | [speech-recognition-from-scratch](https://github.com/CodersArts/speech-recognition-from-scratch) |
| 5 | OCR | [ocr-from-scratch-pytorch](https://github.com/CodersArts/ocr-from-scratch-pytorch) |
| 6 | Computer Vision | [cnn-vision-from-scratch](https://github.com/CodersArts/cnn-vision-from-scratch) |

**Module 5 — OCR & Document Reader** ← you are here

---

## Learn to build this

| What | Link |
|------|------|
| 🎓 Full course | [labs.codersarts.com/courses/ocr](https://labs.codersarts.com/courses/ocr) |
| 💬 Mentorship | [codersarts.com/mentorship](https://codersarts.com/mentorship) |

---

<div align="center">

**⭐ Star this repo if it helped you**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub)

</div>
