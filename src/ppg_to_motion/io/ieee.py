"""IO for the IEEE Signal Processing Cup 2015 dataset (.ts format).

The dataset ships as two flat text files (IEEEPPG_TRAIN.ts, IEEEPPG_TEST.ts).
Each data line encodes a pre-segmented ~8-second window (999 samples at 125 Hz)
across 5 channels stored as consecutive blocks:

  PPG1[0:999], PPG2[0:999], AccX[0:999], AccY[0:999], AccZ[0:999] : HR_BPM

Because each row is an independent, pre-segmented window — not a slice of a
continuous recording — 30-second re-segmentation cannot be applied. The generator
yields each 8-second row as one sample. Subject IDs are not present in the .ts
format; a split+row index is used instead.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_FS: float = 125.0
_N_SAMPLES: int = 1000  # samples per channel per row (8 s at 125 Hz)
_N_CHANNELS: int = 5    # PPG1, PPG2, AccX, AccY, AccZ


def ieee_generator(root: Path | str) -> Iterator[dict]:
    """Yield one dict per row of IEEEPPG_TRAIN.ts and IEEEPPG_TEST.ts.

    Yields
    ------
    signal        : np.ndarray (float32), shape (999,) — first PPG channel, 125 Hz
    acc           : np.ndarray (float32), shape (3, 999) — AccX/Y/Z at 125 Hz
    sampling_rate : 125.0
    ID            : str  e.g. "IEEE_TRAIN_000000"
    label         : float — heart rate in BPM (ECG ground truth)
    source        : "ieee"
    source_file   : absolute path to the .ts file
    """
    root = Path(root)
    ts_files = sorted(root.glob("IEEEPPG_*.ts"))
    if not ts_files:
        logger.warning("No IEEEPPG_*.ts files found in %s", root)
        return

    for ts_file in ts_files:
        split = "TRAIN" if "TRAIN" in ts_file.stem.upper() else "TEST"
        logger.info("Reading %s", ts_file)
        row_idx = 0

        with ts_file.open("r", encoding="ascii") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("@"):
                    continue

                # The .ts format uses ':' as dimension separator; final part is the label.
                # Format: dim0_csv : dim1_csv : dim2_csv : dim3_csv : dim4_csv : label
                try:
                    parts = line.split(":")
                    if len(parts) != _N_CHANNELS + 1:
                        raise ValueError(f"expected {_N_CHANNELS + 1} colon-parts, got {len(parts)}")
                    label = float(parts[-1])
                    dims = [np.fromstring(p, sep=",", dtype=np.float32) for p in parts[:-1]]
                except Exception as exc:
                    logger.warning("Row %d parse error: %s", row_idx, exc)
                    row_idx += 1
                    continue

                if any(d.size != _N_SAMPLES for d in dims):
                    sizes = [d.size for d in dims]
                    logger.warning("Row %d: unexpected dim sizes %s — skipping", row_idx, sizes)
                    row_idx += 1
                    continue

                ppg = dims[0].copy()                            # (1000,) PPG channel 1
                acc = np.stack(dims[2:5], axis=0)              # (3, 1000) AccX/Y/Z

                yield {
                    "signal": ppg,
                    "acc": acc,
                    "sampling_rate": _FS,
                    "ID": f"IEEE_{split}_{row_idx:06d}",
                    "label": label,
                    "source": "ieee",
                    "source_file": str(ts_file),
                }
                row_idx += 1

        logger.info("IEEE %s: yielded %d segments", split, row_idx)
