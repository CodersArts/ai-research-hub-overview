"""
conv2d.py — 2D Convolution from scratch
=========================================
Implements Conv2D as a sliding window matrix multiplication (im2col).

What convolution does:
  Slides a small filter (e.g. 3×3) over the input image.
  At each position, computes the dot product between filter and image patch.
  Output = feature map showing where the filter pattern was detected.

im2col trick:
  Instead of nested loops (slow), reshape the image so that
  each patch becomes a column vector → single matmul for all positions.

This is exactly how PyTorch's nn.Conv2d works internally.
"""

import numpy as np


def im2col(
    x:        np.ndarray,   # (C_in, H, W)
    kernel_h: int,
    kernel_w: int,
    stride:   int = 1,
    padding:  int = 0,
) -> np.ndarray:
    """
    Rearrange image patches into columns for efficient convolution.

    Each column = one flattened kernel_h×kernel_w patch from the input.
    Output shape: (C_in*kernel_h*kernel_w, out_h*out_w)
    """
    C, H, W = x.shape

    # Pad the input
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))

    out_h = (H + 2*padding - kernel_h) // stride + 1
    out_w = (W + 2*padding - kernel_w) // stride + 1

    col = np.zeros((C * kernel_h * kernel_w, out_h * out_w))

    col_idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            col[:, col_idx] = patch.flatten()
            col_idx += 1

    return col


def conv2d(
    x:       np.ndarray,   # (C_in, H, W)
    weight:  np.ndarray,   # (C_out, C_in, kH, kW)
    bias:    np.ndarray,   # (C_out,)
    stride:  int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    2D convolution using im2col + matmul.

    Returns: (C_out, out_H, out_W)
    """
    C_in, H, W       = x.shape
    C_out, _, kH, kW = weight.shape

    out_h = (H + 2*padding - kH) // stride + 1
    out_w = (W + 2*padding - kW) // stride + 1

    # Reshape input patches: (C_in*kH*kW, out_h*out_w)
    col = im2col(x, kH, kW, stride, padding)

    # Reshape weights: (C_out, C_in*kH*kW)
    w_flat = weight.reshape(C_out, -1)

    # Single matmul: (C_out, out_h*out_w)
    out = w_flat @ col + bias[:, np.newaxis]

    return out.reshape(C_out, out_h, out_w)


if __name__ == "__main__":
    # Test: apply an edge-detection filter
    import numpy as np

    # 1-channel 5×5 image
    x = np.random.randn(1, 5, 5)

    # Sobel-like edge filter (1 output channel)
    weight = np.array([[[[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]]])

    bias = np.array([0.0])

    out = conv2d(x, weight, bias, stride=1, padding=1)
    print(f"Input  : {x.shape}")
    print(f"Weight : {weight.shape}")
    print(f"Output : {out.shape}")   # should be (1, 5, 5) with padding=1
    print("Conv2D OK ✓")
