"""
fft.py — Discrete Fourier Transform from scratch
=================================================
The DFT converts a time-domain audio signal into frequency components.
This is the foundation of all audio feature extraction (MFCC, spectrograms).

Math:
  X[k] = sum(n=0..N-1) of x[n] * e^(-j*2*pi*k*n/N)

Where:
  x[n] = signal sample at time n
  X[k] = complex amplitude of frequency component k
  N    = total number of samples

We implement the Cooley-Tukey FFT algorithm (Fast Fourier Transform)
which reduces O(N²) DFT to O(N log N) using divide-and-conquer.
"""

import numpy as np


def dft_naive(x: np.ndarray) -> np.ndarray:
    """
    Naive Discrete Fourier Transform — O(N²).
    Educational: shows the exact formula but too slow for real audio.
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def fft_recursive(x: np.ndarray) -> np.ndarray:
    """
    Cooley-Tukey FFT — O(N log N).
    Divide: split into even-indexed and odd-indexed samples.
    Conquer: compute FFT of each half recursively.
    Combine: butterfly operation.
    """
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        # Fall back to naive DFT for non-power-of-2 lengths
        return dft_naive(x)

    # Divide: split into even and odd indices
    even = fft_recursive(x[0::2])
    odd  = fft_recursive(x[1::2])

    # Twiddle factors: e^(-2πi k/N)
    T = np.exp(-2j * np.pi * np.arange(N//2) / N)

    # Combine (butterfly)
    return np.concatenate([
        even + T * odd,
        even - T * odd,
    ])


def compute_spectrogram(
    signal:    np.ndarray,
    sr:        int   = 16000,
    n_fft:     int   = 512,
    hop_len:   int   = 256,
) -> np.ndarray:
    """
    Short-Time Fourier Transform (STFT) → spectrogram.

    Slides a window over the signal, computes FFT at each position.
    Result is a 2D array: (frequency_bins, time_frames)
    """
    frames = []
    window = np.hanning(n_fft)   # Hanning window reduces spectral leakage

    for start in range(0, len(signal) - n_fft, hop_len):
        frame = signal[start:start + n_fft] * window
        spectrum = np.abs(fft_recursive(frame))[:n_fft // 2]
        frames.append(spectrum)

    return np.array(frames).T   # (freq_bins, time_frames)


if __name__ == "__main__":
    # Generate a test signal: 440 Hz sine wave (concert A)
    sr     = 16000
    t      = np.linspace(0, 1, sr)
    signal = np.sin(2 * np.pi * 440 * t)

    spec = compute_spectrogram(signal, sr)
    print(f"Signal shape     : {signal.shape}")
    print(f"Spectrogram shape: {spec.shape}")

    # Find peak frequency
    peak_bin = np.argmax(spec.mean(axis=1))
    freq_res = sr / 512
    print(f"Detected peak    : ~{peak_bin * freq_res:.0f} Hz  (should be ≈440)")
