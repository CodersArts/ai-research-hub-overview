"""
otsu.py — Otsu's thresholding from scratch
===========================================
Converts a grayscale image to binary (black/white) by finding
the optimal threshold that maximises inter-class variance.

Why binarization?
  OCR needs clear black text on white background.
  Otsu automatically finds the best threshold — works on
  images with varying lighting, no manual tuning needed.

Algorithm:
  For each possible threshold t (0..255):
    - Split pixels into background (<t) and foreground (≥t)
    - Compute weighted variance between the two classes
  Choose t that maximises inter-class variance.
"""

import numpy as np


def otsu_threshold(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Find optimal threshold using Otsu's method.

    Args:
        image: Grayscale image as 2D NumPy array (values 0–255)

    Returns:
        threshold: The optimal threshold value
        binary:    Binarized image (0=background, 255=foreground)
    """
    assert image.ndim == 2, "Input must be grayscale (2D array)"

    # Compute normalised histogram (256 bins, one per intensity level)
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist    = hist.astype(float) / hist.sum()   # probabilities

    best_t    = 0
    best_var  = 0.0

    # Total mean intensity
    total_mean = np.sum(np.arange(256) * hist)

    cumsum_prob = 0.0   # cumulative probability of background class
    cumsum_mean = 0.0   # cumulative mean of background class

    for t in range(256):
        cumsum_prob += hist[t]
        cumsum_mean += t * hist[t]

        if cumsum_prob == 0 or cumsum_prob == 1:
            continue

        # Weight of each class
        w_bg = cumsum_prob
        w_fg = 1.0 - cumsum_prob

        # Mean of each class
        mean_bg = cumsum_mean / w_bg
        mean_fg = (total_mean - cumsum_mean) / w_fg

        # Inter-class variance (we want to MAXIMISE this)
        inter_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2

        if inter_var > best_var:
            best_var = inter_var
            best_t   = t

    # Apply threshold
    binary = np.where(image >= best_t, 255, 0).astype(np.uint8)
    return best_t, binary


if __name__ == "__main__":
    # Create test image: white background with dark text region
    img = np.ones((100, 200), dtype=np.uint8) * 220
    img[30:70, 50:150] = 40    # simulated dark text region

    t, binary = otsu_threshold(img)
    print(f"Otsu threshold: {t}")
    print(f"Binary image unique values: {np.unique(binary)}")
