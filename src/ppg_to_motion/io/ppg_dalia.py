"""IO + segmentation for the PPG-DaLiA dataset.

Data lives in:
  <root>/data/PPG_FieldStudy/S{1..15}/S{1..15}.pkl

Each pickle contains a dict with:
  signal['wrist']['BVP']  — PPG at 64 Hz, shape (N,) or (N,1)
  signal['wrist']['ACC']  — 3-axis accelerometer at 32 Hz, shape (N_acc,3) or (3,N_acc)
  label                   — HR in BPM, one value per 8-second ECG window (2-second shift)
  subject                 — subject ID string, e.g. "S1"

PPG (64 Hz) and ACC (32 Hz) are both upsampled to 100 Hz before windowing.
ACC magnitude sqrt(X²+Y²+Z²) is computed after upsampling.

Segmentation: 30-second windows (3000 samples at 100 Hz) with 50% overlap
  → step = 1500 samples (15 seconds).

Signal images:
  diff_image — computed from the upsampled 100 Hz PPG window.
  stft_image — computed from the raw 64 Hz PPG window (1920 samples, fs=64).

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

_PPG_FS: float = 64.0   # native PPG rate (used for STFT and HR label alignment)
_ACC_FS: float = 32.0   # native ACC rate
_ACT_FS: float = 4.0    # activity label sampling rate
_OUT_FS: float = 100.0  # output rate after upsampling

_ACTIVITY_NAMES: dict[int, str]  = {
    0 : "no_activity",
    1 : "sitting",
    2 : "stairs",
    3 : "table_soccer",
    4 : "cycling",
    5 : "driving",
    6 : "lunch",
    7 : "walking",
    8 : "office_working"   
}
_WINDOW_SEC: int = 30
_OVERLAP: float = 0.50
# Raw-signal window (64 Hz) — used for STFT and HR label indexing only
_WINDOW_SAMPLES: int = int(_PPG_FS * _WINDOW_SEC)        # 1920
_STEP_SAMPLES: int = int(_WINDOW_SAMPLES * (1.0 - _OVERLAP))  # 480
# Output window (100 Hz) — used for signal, acc, and diff_image
_OUT_WINDOW_SAMPLES: int = int(_OUT_FS * _WINDOW_SEC)    # 3000
_OUT_STEP_SAMPLES: int = int(_OUT_WINDOW_SAMPLES * (1.0 - _OVERLAP))  # 750


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


def _upsample_ppg(ppg: np.ndarray) -> np.ndarray:
    """Upsample PPG from 64 Hz to 100 Hz (ratio 25:16)."""
    g = gcd(int(_OUT_FS), int(_PPG_FS))
    return resample_poly(ppg, int(_OUT_FS) // g, int(_PPG_FS) // g).astype(np.float32)


def _resample_acc(acc_3xN: np.ndarray, n_out: int) -> np.ndarray:
    """Upsample ACC from 32 Hz to 100 Hz and trim/pad to n_out samples."""
    g = gcd(int(_OUT_FS), int(_ACC_FS))
    acc_up = resample_poly(acc_3xN, int(_OUT_FS) // g, int(_ACC_FS) // g, axis=1).astype(np.float32)
    if acc_up.shape[1] > n_out:
        return acc_up[:, :n_out]
    if acc_up.shape[1] < n_out:
        return np.pad(acc_up, ((0, 0), (0, n_out - acc_up.shape[1])), mode="edge")
    return acc_up


def _mode_activity(activity: np.ndarray, raw_start: int, n_raw: int) -> str | None:
    """Return the modal activity name for a PPG window.

    Activity labels are at 4 Hz; raw_start and n_raw are in 64 Hz samples.
    """
    if activity is None or len(activity) == 0:
        return None
    act_start = round(raw_start * _ACT_FS / _PPG_FS)
    act_end   = round((raw_start + n_raw) * _ACT_FS / _PPG_FS)
    window = activity[act_start:act_end]
    if len(window) == 0:
        return None
    counts = np.bincount(window.astype(np.intp))
    mode_val = int(counts.argmax())
    return _ACTIVITY_NAMES.get(mode_val)


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
    """Yield 30-second PPG segments (50% overlap) from all 15 PPG-DaLiA subjects.

    PPG and ACC are upsampled from their native rates (64 Hz and 32 Hz) to 100 Hz.

    Yields
    ------
    signal        : np.ndarray (float32), shape (3000,) — PPG upsampled to 100 Hz (30 s)
    acc           : np.ndarray (float32), shape (3000,) — ACC magnitude sqrt(X²+Y²+Z²) at 100 Hz
    sampling_rate : 100.0
    ID            : str — subject ID from pickle, e.g. "S1"
    label         : float — mean HR in BPM for the window (omitted if unavailable)
    source        : "ppg-dalia"
    source_file   : str — path to the subject folder
    diff_image    : np.ndarray (float32), shape (3, 3000) — from upsampled 100 Hz PPG
    stft_image    : np.ndarray (float32), shape (3, n_freqs, n_frames) — from raw 64 Hz PPG
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
        ppg_up = _upsample_ppg(ppg)                               # (N_up,) at 100 Hz
        n_ppg_up = len(ppg_up)
        acc_up = _resample_acc(acc_raw, n_ppg_up)                 # (3, N_up) at 100 Hz
        acc_mag = np.sqrt(np.sum(acc_up ** 2, axis=0))            # (N_up,)
        labels   = np.asarray(data.get("label",    []), dtype=np.float32)
        activity = np.asarray(data.get("activity", []), dtype=np.int32).squeeze()

        logger.info(
            "%s: PPG=%d samples (%.1f min) → generating 30-s windows at 100 Hz",
            subject_id, n_ppg, n_ppg / _PPG_FS / 60,
        )

        # Drive the loop at 100 Hz; derive the matching raw-signal start for STFT and HR labels.
        for up_start in range(0, n_ppg_up - _OUT_WINDOW_SAMPLES + 1, _OUT_STEP_SAMPLES):
            raw_start = round(up_start * _PPG_FS / _OUT_FS)  # always integer (750×64/100=480)
            if raw_start + _WINDOW_SAMPLES > n_ppg:
                break

            ppg_window_up = ppg_up[up_start : up_start + _OUT_WINDOW_SAMPLES]
            acc_window    = acc_mag[up_start : up_start + _OUT_WINDOW_SAMPLES]
            hr  = _mean_hr_for_window(labels, raw_start) if len(labels) else None
            act = _mode_activity(activity, raw_start, _WINDOW_SAMPLES)

            result: dict = {
                "signal": ppg_window_up,
                "acc": acc_window,
                "sampling_rate": _OUT_FS,
                "ID": subject_id,
                "source": "ppg-dalia",
                "source_file": str(pkl_file.parent),
            }
            if hr is not None:
                result["label"] = hr
            if act is not None:
                result["activity"] = act

            yield result
