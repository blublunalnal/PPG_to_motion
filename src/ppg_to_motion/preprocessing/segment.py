"""Sliding-window segmentation utilities."""
from __future__ import annotations
from typing import Iterator
import numpy as np


def segment_signal(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start_index, window) pairs from a 1-D signal."""
    n = len(signal)
    for start in range(0, n - window_samples + 1, step_samples):
        yield start, signal[start : start + window_samples]
