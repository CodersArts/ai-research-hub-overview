<div align="center">

# 🎙️ Speech Recognition — From Scratch

### FFT · Mel Spectrograms · MFCC · CNN · CTC Decoder

Build a basic speech recognition system — **no Whisper, no cloud APIs, runs fully offline.**

[![Stars](https://img.shields.io/github/stars/CodersArts/speech-recognition-from-scratch?style=flat-square&color=gold)](https://github.com/CodersArts/speech-recognition-from-scratch/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Part of](https://img.shields.io/badge/Part%20of-AI%20Research%20Hub-purple?style=flat-square)](https://labs.codersarts.com/ai-research-hub)

**Module 4 of 6 — [AI Research Hub](https://labs.codersarts.com/ai-research-hub)**

</div>

---

## What you will build

A complete pipeline from raw input to intelligent output — every layer hand-written.

---

## Key concepts you implement from scratch

- FFT and spectrograms from raw audio bytes
- Mel filter banks and MFCC feature extraction
- 1D CNN keyword spotter (yes/no/silence)
- CTC loss and greedy/beam decoder
- Building a basic speech-to-text pipeline

---

## Project structure

```
speech-recognition-from-scratch/
├── src/
│   ├── audio_utils.py    ← Read WAV files from raw bytes (no librosa)
│   ├── fft.py            ← Discrete Fourier Transform from scratch
│   ├── mfcc.py           ← Mel filter banks + MFCC extraction
│   ├── cnn_classifier.py ← 1D CNN for keyword spotting
│   ├── ctc_model.py      ← RNN + CTC for basic speech-to-text
│   ├── train_keyword.py  ← Training loop for keyword spotter
│   ├── transcribe.py     ← Run inference on a WAV file
├── data/
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/CodersArts/speech-recognition-from-scratch
cd speech-recognition-from-scratch
pip install -r requirements.txt

python src/audio_utils.py    # test loading a WAV file\npython src/mfcc.py            # extract features from audio\npython src/train_keyword.py   # train keyword spotter\npython src/transcribe.py --audio sample.wav
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

**Module 4 — Speech Recognition** ← you are here

---

## Learn to build this

| What | Link |
|------|------|
| 🎓 Full course | [labs.codersarts.com/courses/speech-recognition](https://labs.codersarts.com/courses/speech-recognition) |
| 💬 Mentorship | [codersarts.com/mentorship](https://codersarts.com/mentorship) |

---

<div align="center">

**⭐ Star this repo if it helped you**

Built by [Codersarts](https://codersarts.com) · [AI Research Hub](https://labs.codersarts.com/ai-research-hub)

</div>
