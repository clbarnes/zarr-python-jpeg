import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert interleaved uint8 RGB into interleaved uint8 full-range (JFIF) YCbCr."""
    ycbcr_from_rgb = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        "float64",
    )
    ycbcr = rgb.astype("float64") @ ycbcr_from_rgb.T
    ycbcr[..., 1:] += 128
    return np.rint(ycbcr).clip(0, 255).astype(np.uint8)
