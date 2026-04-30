<div align="center">

# 👁️ Computer Vision — From Scratch

### Conv2D · MaxPool · BatchNorm · ResNet · CIFAR-10

Build a convolutional neural network from the ground up — **train on CIFAR-10 with no torchvision models.**

[![Stars](https://img.shields.io/github/stars/CodersArts/cnn-vision-from-scratch?style=flat-square&color=gold)](https://github.com/CodersArts/cnn-vision-from-scratch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 6 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What you will build

A complete pipeline from raw input to intelligent output — every layer hand-written.

---

## Key concepts you implement from scratch

- Conv2D as sliding window matrix multiplication
- Max pooling and stride mechanics
- Batch normalisation (mean/variance over mini-batch)
- Residual skip connections (ResNet-style)
- Data augmentation from scratch
- Grad-CAM visualisation

---

## Project structure

```
cnn-vision-from-scratch/
├── src/
│   ├── conv2d.py         ← 2D convolution as im2col matmul
│   ├── pooling.py        ← Max pooling and average pooling
│   ├── batchnorm.py      ← Batch normalisation from scratch
│   ├── resnet.py         ← ResNet with skip connections
│   ├── augment.py        ← Data augmentation (flip, crop, colour jitter)
│   ├── train.py          ← Full CIFAR-10 training loop
│   ├── visualize.py      ← Grad-CAM attention visualisation
├── data/
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/cnn-vision-from-scratch
cd cnn-vision-from-scratch
pip install -r requirements.txt

python src/conv2d.py          # verify convolution from scratch\npython src/download_cifar.py  # get CIFAR-10 dataset\npython src/train.py           # train on CIFAR-10\npython src/visualise.py       # Grad-CAM heatmaps
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

**Module 6 — Computer Vision** ← you are here

---

## Learn to build this

| What | Link |
|------|------|
| 🎓 Full course | [labs.codersarts.com/courses/computer-vision](https://labs.codersarts.com/courses/computer-vision) |
| 💬 Mentorship | [codersarts.com/mentorship](https://codersarts.com/mentorship) |

---

<div align="center">

**⭐ Star this repo if it helped you**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub)

</div>
