"""IO + segmentation for the PPG-DaLiA dataset.

Data lives in:
  <root>/data/PPG_FieldStudy/S{1..15}/S{1..15}.pkl

Each pickle contains a dict with:
  signal['wrist']['BVP']  — PPG at 64 Hz, shape (N,) or (N,1)
  signal['wrist']['ACC']  — 3-axis accelerometer at 32 Hz, shape (N_acc,3) or (3,N_acc)
  label                   — HR in BPM, one value per 8-second ECG window (2-second shift)
  subject                 — subject ID string, e.g. "S1"

Wrist ACC (32 Hz) is resampled to 64 Hz to match the PPG sampling rate.

Segmentation: 30-second windows (1920 samples at 64 Hz) with 75% overlap
  → step = 480 samples (7.5 seconds).

HR label per segment: mean of all ECG HR values whose 8-second coverage
overlaps the 30-second PPG window.
"""
from __future__ import annotations

import logging
import pickle
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

_PPG_FS: float = 64.0
_ACC_FS: float = 32.0
_WINDOW_SEC: int = 30
_OVERLAP: float = 0.75
_WINDOW_SAMPLES: int = int(_PPG_FS * _WINDOW_SEC)   # 1920
_STEP_SAMPLES: int = int(_WINDOW_SAMPLES * (1.0 - _OVERLAP))  # 480


def _load_pkl(pkl_file: Path) -> dict | None:
    try:
        with pkl_file.open("rb") as fh:
            return pickle.load(fh, encoding="latin1")
    except Exception as exc:
        logger.warning("Failed to load %s: %s", pkl_file, exc)
        return None


def _to_1d_float32(arr: np.ndarray) -> np.ndarray:
    """Squeeze to 1-D float32."""
    return arr.squeeze().astype(np.float32)


def _to_3xN_float32(arr: np.ndarray) -> np.ndarray:
    """Normalise ACC to shape (3, N) float32."""
    a = arr.astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.T   # (N,3) → (3,N)
    if a.ndim == 2 and a.shape[0] == 3:
        return a     # already (3,N)
    raise ValueError(f"Unexpected ACC shape: {a.shape}")


def _resample_acc(acc_3xN: np.ndarray, n_ppg: int) -> np.ndarray:
    """Upsample ACC from 32 Hz to 64 Hz and trim/pad to match PPG length."""
    up, down = int(_PPG_FS), int(_ACC_FS)
    g = gcd(up, down)
    acc_up = resample_poly(acc_3xN, up // g, down // g, axis=1).astype(np.float32)
    # Trim or edge-pad to exactly match PPG length
    if acc_up.shape[1] > n_ppg:
        return acc_up[:, :n_ppg]
    if acc_up.shape[1] < n_ppg:
        pad = n_ppg - acc_up.shape[1]
        return np.pad(acc_up, ((0, 0), (0, pad)), mode="edge")
    return acc_up


def _mean_hr_for_window(labels: np.ndarray, start_sample: int) -> float | None:
    """Return mean HR (BPM) from DaLiA labels overlapping a 30-second window.

    DaLiA label[i] covers ECG time [2i, 2i+8] seconds (8-second window, 2-second shift).
    """
    t_start = start_sample / _PPG_FS          # seconds
    t_end = t_start + _WINDOW_SEC
    # Label indices whose 8-second window [2i, 2i+8] overlaps [t_start, t_end]
    i_lo = max(0, int((t_start - 8.0) / 2.0) + 1)
    i_hi = min(len(labels), int(t_end / 2.0) + 1)
    if i_lo >= i_hi:
        return None
    return float(np.mean(labels[i_lo:i_hi]))


def ppg_dalia_generator(root: Path | str) -> Iterator[dict]:
    """Yield 30-second PPG segments (75% overlap) from all 15 PPG-DaLiA subjects.

    Yields
    ------
    signal        : np.ndarray (float32), shape (1920,) — raw PPG at 64 Hz (30 s)
    acc           : np.ndarray (float32), shape (3, 1920) — ACC resampled to 64 Hz
    sampling_rate : 64.0
    ID            : str — subject ID from pickle, e.g. "S1"
    label         : float — mean HR in BPM for the window (omitted if unavailable)
    source        : "ppg-dalia"
    source_file   : str — path to the subject folder
    """
    root = Path(root)
    data_dir = root / "data" / "PPG_FieldStudy"
    pkl_files = sorted(data_dir.glob("S*/*.pkl"))

    if not pkl_files:
        logger.warning("No S*/*.pkl files found under %s", data_dir)
        return

    for pkl_file in pkl_files:
        data = _load_pkl(pkl_file)
        if data is None:
            continue

        try:
            subject_id = str(data.get("subject", pkl_file.parent.name))
            ppg = _to_1d_float32(data["signal"]["wrist"]["BVP"])
            acc_raw = _to_3xN_float32(data["signal"]["wrist"]["ACC"])
        except Exception as exc:
            logger.warning("Bad data in %s: %s", pkl_file, exc)
            continue

        n_ppg = len(ppg)
        acc_64 = _resample_acc(acc_raw, n_ppg)
        labels = np.asarray(data.get("label", []), dtype=np.float32)

        logger.info(
            "%s: PPG=%d samples (%.1f min) → generating 30-s windows",
            subject_id, n_ppg, n_ppg / _PPG_FS / 60,
        )

        for start in range(0, n_ppg - _WINDOW_SAMPLES + 1, _STEP_SAMPLES):
            ppg_window = ppg[start : start + _WINDOW_SAMPLES]
            acc_window = acc_64[:, start : start + _WINDOW_SAMPLES]
            hr = _mean_hr_for_window(labels, start) if len(labels) else None

            result: dict = {
                "signal": ppg_window,
                "acc": acc_window,
                "sampling_rate": _PPG_FS,
                "ID": subject_id,
                "source": "ppg-dalia",
                "source_file": str(pkl_file.parent),
            }
            if hr is not None:
                result["label"] = hr

            yield result
