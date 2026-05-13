"""IO + segmentation for the VSM Free-Living dataset.

Files are named VSM*processed.parquet (one per subject).
Subject ID is extracted from the filename after 'sub_'.
Native sampling rate is inferred from the Datetime column (typically 32 Hz).

Signals are resampled to 100 Hz. ACC channels are bandpass filtered (0.5–20 Hz).
Segmentation: 30-second windows without overlap (step = 3000 samples at 100 Hz).
diff_image and stft_image are computed from Green1_Raw (channel 0) at build time.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from ppg_to_motion.preprocessing.signal_image_2d import (
    generate_differential_signal_image,
    generate_stft_features,
    normalize_multichannel_image,
)

logger = logging.getLogger(__name__)

_OUT_FS: float = 100.0
_WINDOW_SEC: int = 30
_WINDOW_SAMPLES: int = int(_OUT_FS * _WINDOW_SEC)  # 3000

_PPG_COLUMNS = ["Green1_Raw", "Green2_Raw", "Red1_Raw", "Red2_Raw", "IR1_Raw", "IR2_Raw"]
_ACC_COLUMNS = ["X", "Y", "Z"]

_STFT_N_FFT = 128
_STFT_HOP = 16

_MAX_BAD_FRACTION = 0.05


def _filter_acc(acc_3xN: np.ndarray, fs: float) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass (0.5–20 Hz) applied per axis."""
    sos = butter(4, [0.5, 20.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, acc_3xN, axis=1).astype(np.float32)


def _repair(arr: np.ndarray, name: str, filename: str) -> np.ndarray | None:
    """Replace non-finite values with channel median; return None if > _MAX_BAD_FRACTION bad."""
    bad = ~np.isfinite(arr)
    if bad.mean() > _MAX_BAD_FRACTION:
        logger.warning("%.1f%% non-finite in %s of %s — skipping file", 100.0 * bad.mean(), name, filename)
        return None
    if bad.any():
        med = float(np.nanmedian(arr))
        arr = np.where(bad, med if np.isfinite(med) else 0.0, arr).astype(np.float32)
    return arr


def _infer_fs(df: pd.DataFrame) -> float:
    """Infer sampling rate from Datetime column."""
    dt = df["Datetime"].values
    n = min(1000, len(dt) - 1)
    diffs_ns = np.diff(dt[:n + 1].astype(np.float64))
    diffs_ns = diffs_ns[diffs_ns > 0]
    if len(diffs_ns) == 0:
        logger.warning("Could not infer fs from Datetime; assuming 32 Hz")
        return 32.0
    return round(1e9 / float(np.median(diffs_ns)), 4)


def _resample(arr: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Resample 1-D or (C, N) array from src_fs to dst_fs via polyphase filter."""
    if abs(src_fs - dst_fs) < 0.01:
        return arr
    g = math.gcd(round(dst_fs), round(src_fs))
    up = round(dst_fs) // g
    down = round(src_fs) // g
    if arr.ndim == 1:
        return resample_poly(arr, up, down).astype(np.float32)
    return np.stack(
        [resample_poly(arr[i], up, down).astype(np.float32) for i in range(arr.shape[0])],
        axis=0,
    )


def vsm_freeliving_generator(root: Path | str) -> Iterator[dict]:
    """Yield 30-second non-overlapping PPG segments from VSM free-living parquet files.

    Yields
    ------
    source        : "vsm-free-living"
    subject_id    : str — extracted from filename after 'sub_'
    sampling_rate : 100.0
    source_file   : str — parquet filename
    segment_index : int
    signal        : np.ndarray float32 (6, 3000) — all 6 PPG channels at 100 Hz
    acc           : np.ndarray float32 (3000,) — bandpassed ACC magnitude at 100 Hz
    acc_xyz       : np.ndarray float32 (3, 3000) — bandpassed ACC [X, Y, Z] at 100 Hz
    ppg_channels  : np.ndarray float32 (6, 3000) — same as signal
    diff_image    : np.ndarray float32 (3, 3000) — z-scored [x, dx, d²x] from Green1_Raw
    stft_image    : np.ndarray float32 (3, 65, 188) — z-scored log-mag/cos-phase/sin-phase from Green1_Raw
    """
    root = Path(root)
    parquet_files = sorted(root.glob("VSM*processed.parquet"))
    if not parquet_files:
        logger.warning("No VSM*processed.parquet files found in %s", root)
        return

    for pq_path in parquet_files:
        m = re.search(r"sub_(\w+?)_processed", pq_path.name)
        subject_id = m.group(1) if m else pq_path.stem

        logger.info("Processing %s (subject %s)", pq_path.name, subject_id)
        try:
            df = pd.read_parquet(pq_path, columns=_PPG_COLUMNS + _ACC_COLUMNS + ["Datetime"])
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pq_path.name, exc)
            continue

        if df.empty:
            logger.warning("Empty parquet: %s", pq_path.name)
            continue

        src_fs = _infer_fs(df)
        logger.info("%s: fs=%.2f Hz, %d rows (%.1f min)",
                    pq_path.name, src_fs, len(df), len(df) / src_fs / 60)

        # --- PPG (6, N) ---
        ppg_raw = np.stack(
            [pd.to_numeric(df[c], errors="coerce").values.astype(np.float32)
             for c in _PPG_COLUMNS],
            axis=0,
        )
        skip = False
        for i, col in enumerate(_PPG_COLUMNS):
            repaired = _repair(ppg_raw[i], col, pq_path.name)
            if repaired is None:
                if i == 0:
                    skip = True
                    break
                ppg_raw[i] = np.zeros_like(ppg_raw[i])
            else:
                ppg_raw[i] = repaired
        if skip:
            continue

        # --- ACC (3, N) ---
        acc_raw = np.stack(
            [pd.to_numeric(df[c], errors="coerce").values.astype(np.float32)
             for c in _ACC_COLUMNS],
            axis=0,
        )
        for i, col in enumerate(_ACC_COLUMNS):
            repaired = _repair(acc_raw[i], col, pq_path.name)
            acc_raw[i] = repaired if repaired is not None else np.zeros_like(acc_raw[i])

        # --- Resample to 100 Hz ---
        ppg = _resample(ppg_raw, src_fs, _OUT_FS)   # (6, N')
        acc = _resample(acc_raw, src_fs, _OUT_FS)   # (3, N')

        # --- Bandpass ACC, compute magnitude ---
        acc = _filter_acc(acc, _OUT_FS)
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=0)).astype(np.float32)  # (N',)

        n = ppg.shape[1]
        if n < _WINDOW_SAMPLES:
            logger.warning("%s: too short after resampling (%d < %d samples)",
                           pq_path.name, n, _WINDOW_SAMPLES)
            continue

        logger.info("%s: %d samples at %.0f Hz → %d non-overlapping windows",
                    pq_path.name, n, _OUT_FS, n // _WINDOW_SAMPLES)

        for seg_idx, start in enumerate(range(0, n - _WINDOW_SAMPLES + 1, _WINDOW_SAMPLES)):
            ppg_win     = ppg[:, start: start + _WINDOW_SAMPLES]   # (6, 3000)
            acc_win     = acc_mag[start: start + _WINDOW_SAMPLES]  # (3000,)
            acc_xyz_win = acc[:, start: start + _WINDOW_SAMPLES]   # (3, 3000)
            ch0 = ppg_win[0]

            diff_image = normalize_multichannel_image(
                generate_differential_signal_image(ch0), method="zscore"
            )
            stft_image = generate_stft_features(ch0, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP)

            yield {
                "source":        "vsm-free-living",
                "subject_id":    subject_id,
                "sampling_rate": _OUT_FS,
                "source_file":   pq_path.name,
                "segment_index": seg_idx,
                "signal":        ppg_win,
                "acc":           acc_win,
                "acc_xyz":       acc_xyz_win,
                "ppg_channels":  ppg_win,
                "diff_image":    diff_image,
                "stft_image":    stft_image,
            }
