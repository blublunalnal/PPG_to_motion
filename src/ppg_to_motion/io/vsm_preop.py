"""IO + segmentation for the VSM Pre-OP dataset.

Data lives in:
  <root>/VSM_ih_*/Laptop Logged Data/*/CSV/*CombinedData*.csv

Each CSV has metadata in rows 0-3; row 4 is column headers; rows 5+ are data.
PPG channels: S1, S2 (green ~530 nm), S1.1, S2.1 (red ~660 nm), S1.2, S2.2 (IR ~880 nm).
All 6 channels are read and returned as a stacked (6, N) array; S1 is channel 0.
Missing or unrepair-able channels are zero-filled.

No bandpass filtering, upsampling, or padding is applied.

Segmentation: 30-second windows (3000 samples at 100 Hz) with 50% overlap
  → step = 1500 samples (15 seconds).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


logger = logging.getLogger(__name__)

_PPG_FS: float = 100.0
_WINDOW_SEC: int = 30
_OVERLAP: float = 0.50
_WINDOW_SAMPLES: int = int(_PPG_FS * _WINDOW_SEC)       # 3000
_STEP_SAMPLES: int = int(_WINDOW_SAMPLES * (1.0 - _OVERLAP))  # 750

_CSV_SKIP_ROWS = [0, 1, 2, 3]
_ACC_COLUMNS = ["X", "Y", "Z"]
# Pandas renames the 7th duplicate "Epoch Delta TS" column (0-indexed: index 6) to this.
_ACC_TS_COLUMN = "Epoch Delta TS.6"
_MAX_BAD_FRACTION = 0.05

_PPG_COLUMNS = ["S1", "S2", "S1.1", "S2.1", "S1.2", "S2.2"]
_N_PPG_CHANNELS = len(_PPG_COLUMNS)  # 6

_EXCLUDE_PATTERN = re.compile(
    "|".join([
        r"nand\s*flash\s*(?:data\s*)?(?:not\s*)?(?:available|only)",
        r"excluded",
        r"only\s*(?:logged\s*)?(?:via\s*)?(?:the\s*)?(?:watch'?s?\s*)?nand",
        r"nand\s*only",
        r"no\s*laptop\s*data",
        r"data\s*not\s*available",
    ]),
    re.IGNORECASE,
)

VSM_preop_MAPPING = {
    'afib': 'AF',
    'svt': 'SVTACH',
    'pvc': 'PVC',
    'atrial flutter': 'AFLT',
    'vt': 'VTACH',
    'afib ablation': 'ABLATION',
    'aflutter ablation': 'ABLATION',
    'afib/flutter': 'AF',
    'aflutter': 'AFLT',
    'ventricular tachycardia': 'VTACH',
    'a flutter': 'AFLT',
    'atypical atrial flutter/ afib': 'AF',
    'a flutter with predominant 2:1 av block': 'AFLT',
    'a flutter w/ predominant 3:1 av block': 'AFLT',
    'afib/flutter ablation': 'AF',
    'a flutter with predominant 4:1 av block': 'AFLT',
    'a flutter w/ 4:1 av block': 'AFLT',
    'atrial fibrillation ablation': 'ABLATION',
    'pvc ablation': 'ABLATION',
    'afib/aflutter': 'AF',
    'svt (pre-atrial contractions)': 'SVTACH',
    'atrial fibrillation / typical atrial flutter': 'AF',
}

def _filter_acc(acc_3xN: np.ndarray, fs: float) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass (0.5–20 Hz) applied per axis."""
    sos = butter(4, [0.5, 20.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, acc_3xN, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Master log
# ---------------------------------------------------------------------------

def _load_master_log(master_log_path: Path) -> dict[str, dict]:
    """Return {folder_id: {participant_id, rhythm_label, usable}} from Excel master log."""
    df = pd.read_excel(master_log_path, sheet_name="Sheet1")
    if df.empty:
        return {}

    def _folder_id(row: pd.Series) -> str | None:
        sub_id = row.get("Sub_ID (for model development)")
        if pd.notna(sub_id) and str(sub_id).strip():
            s = str(sub_id).strip()
            return s.split(" (")[0].strip() if " (" in s else s
        sub_code = row.get("Sub_code")
        if pd.notna(sub_code) and str(sub_code).strip():
            return str(sub_code).strip()
        return None

    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        fid = _folder_id(row)
        if not fid:
            continue
        mrn = row.get("MRN")
        participant_id = str(mrn).strip() if pd.notna(mrn) and str(mrn).strip() else fid
        diag = row.get("Diagnosis (for EP procedure)") or row.get("On-spot hospital ECG outcome")
        rhythm_label = str(diag).strip().lower() if pd.notna(diag) else ""
        notes = (
            str(row.get("Notes (Pre-OP)") or "") + " " +
            str(row.get("Notes (Post-OP)") or "")
        )
        usable = not bool(_EXCLUDE_PATTERN.search(notes))
        lookup[fid] = {
            "participant_id": participant_id,
            "rhythm_label": rhythm_label,
            "usable": usable,
        }
    return lookup


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _find_combined_csv(patient_folder: Path) -> Path | None:
    """Return the Pre-OP *CombinedData*.csv from Laptop Logged Data, or None.

    Skips Post-OP sessions (rhythm has been corrected, label would be wrong).
    """
    for sub in patient_folder.iterdir():
        if not sub.is_dir() or "Laptop" not in sub.name or "Data" not in sub.name:
            continue
        csvs: list[Path] = []
        for session in sub.iterdir():
            if not session.is_dir() or "post" in session.name.lower():
                continue
            csv_dir = session / "CSV"
            if csv_dir.is_dir():
                csvs.extend(csv_dir.glob("*CombinedData*.csv"))
        if len(csvs) > 1:
            logger.warning(
                "Multiple Pre-OP CombinedData CSVs for %s — using first: %s",
                patient_folder.name, [f.name for f in csvs],
            )
        if csvs:
            return csvs[0]
    return None


def _repair(arr: np.ndarray, name: str, path: str) -> np.ndarray | None:
    """Replace non-finite values with channel median; return None if > _MAX_BAD_FRACTION bad."""
    bad = ~np.isfinite(arr)
    if bad.mean() > _MAX_BAD_FRACTION:
        logger.warning("Skipping %s: %.1f%% non-finite in %s", path, 100.0 * bad.mean(), name)
        return None
    if bad.any():
        med = float(np.nanmedian(arr))
        arr = np.where(bad, med if np.isfinite(med) else 0.0, arr).astype(np.float32)
    return arr


def _read_signals(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read all PPG channels and XYZ ACC from CSV.

    Returns (ppg (6, N), acc_mag (N,), acc_xyz (3, N)) float32, or None on failure.
    PPG channel order matches _PPG_COLUMNS: S1, S2, S1.1, S2.1, S1.2, S2.2.
    Missing or unrepair-able channels are zero-filled; S1 (channel 0) missing → return None.
    ACC is upsampled to the PPG timestamp grid via linear interpolation.
    acc_mag = sqrt(X² + Y² + Z²), z-scored. acc_xyz is bandpassed per-axis values.
    """
    try:
        df = pd.read_csv(csv_path, skiprows=_CSV_SKIP_ROWS, header=0, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None
    if df.empty:
        logger.warning("Empty CSV (no rows): %s", csv_path)
        return None
    df.columns = df.columns.astype(str).str.strip()

    # --- PPG: all 6 channels; S1 (ch 0) must be present to determine n_ppg ---
    if _PPG_COLUMNS[0] not in df.columns:
        logger.warning("Column %s missing in %s", _PPG_COLUMNS[0], csv_path.name)
        return None
    ch0_raw = pd.to_numeric(df[_PPG_COLUMNS[0]], errors="coerce").values.astype(np.float32)
    ch0_raw = _repair(ch0_raw, _PPG_COLUMNS[0], csv_path.name)
    if ch0_raw is None:
        return None
    n_ppg = len(ch0_raw)

    ppg = np.zeros((_N_PPG_CHANNELS, n_ppg), dtype=np.float32)
    ppg[0] = ch0_raw
    for i, col in enumerate(_PPG_COLUMNS[1:], start=1):
        if col not in df.columns:
            logger.warning("PPG column %s missing in %s; zero-filling", col, csv_path.name)
            continue
        ch = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        repaired = _repair(ch, col, csv_path.name)
        ppg[i] = repaired if repaired is not None else np.zeros(n_ppg, dtype=np.float32)

    # --- PPG timestamps (for interpolating ACC onto PPG grid) ---
    if "Epoch Delta TS" in df.columns:
        ppg_ts = pd.to_numeric(df["Epoch Delta TS"], errors="coerce")
        ppg_ts = ppg_ts.interpolate(limit_direction="both").values.astype(np.float64)
    else:
        ppg_ts = np.arange(n_ppg, dtype=np.float64) * 10.0  # 100 Hz → 10 ms/sample

    # --- ACC (X, Y, Z): upsample to PPG grid, compute magnitude ---
    missing = [c for c in _ACC_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("ACC columns %s missing in %s; storing zero ACC", missing, csv_path.name)
        return ppg, np.zeros(n_ppg, dtype=np.float32), np.zeros((3, n_ppg), dtype=np.float32)

    if _ACC_TS_COLUMN in df.columns:
        acc_ts_raw = pd.to_numeric(df[_ACC_TS_COLUMN], errors="coerce").values
        valid = np.isfinite(acc_ts_raw)
        acc_valid_ts = acc_ts_raw[valid]
        acc_xyz = df[_ACC_COLUMNS].values[valid].astype(np.float64)  # (M, 3)
        if len(acc_valid_ts) >= 2:
            sort_idx = np.argsort(acc_valid_ts)
            acc_valid_ts = acc_valid_ts[sort_idx]
            acc_xyz = acc_xyz[sort_idx]
            acc_up = np.stack(
                [np.interp(ppg_ts, acc_valid_ts, acc_xyz[:, i]) for i in range(3)],
                axis=0,
            ).astype(np.float32)  # (3, N)
        else:
            logger.warning("Too few valid ACC timestamps in %s; using raw ACC", csv_path.name)
            acc_up = df[_ACC_COLUMNS].values[:n_ppg].T.astype(np.float32)
    else:
        logger.warning("ACC timestamp column not found in %s; using raw ACC", csv_path.name)
        acc_up = df[_ACC_COLUMNS].values[:n_ppg].T.astype(np.float32)

    # Trim or edge-pad to match PPG length
    if acc_up.shape[1] > n_ppg:
        acc_up = acc_up[:, :n_ppg]
    elif acc_up.shape[1] < n_ppg:
        acc_up = np.pad(acc_up, ((0, 0), (0, n_ppg - acc_up.shape[1])), mode="edge")

    # Repair non-finite ACC values per channel; zero-fill if beyond threshold
    for i, col in enumerate(_ACC_COLUMNS):
        repaired = _repair(acc_up[i], col, csv_path.name)
        acc_up[i] = repaired if repaired is not None else np.zeros(n_ppg, dtype=np.float32)

    acc_up = _filter_acc(acc_up, _PPG_FS)
    acc_mag = np.sqrt(np.sum(acc_up ** 2, axis=0))
    _std = acc_mag.std()
    acc_mag = ((acc_mag - acc_mag.mean()) / (_std if _std > 0 else 1.0)).astype(np.float32)
    return ppg, acc_mag, acc_up  # ppg: (6, N), acc_mag: (N,), acc_up: (3, N)


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def vsm_generator(
    root: Path | str,
    master_log_path: Path | str | None = None,
    only_usable: bool = True,
) -> Iterator[dict]:
    """Yield 30-second PPG segments (50% overlap) from all VSM Pre-OP subjects.

    All 6 PPG channels (S1, S2, S1.1, S2.1, S1.2, S2.2) at 100 Hz are returned.
    No bandpass filtering, upsampling, or padding is applied.

    Yields
    ------
    signal        : np.ndarray (float32), shape (6, 3000) — all PPG channels at 100 Hz (30 s); ch 0 = S1 green
    acc           : np.ndarray (float32), shape (3000,) — ACC magnitude sqrt(X²+Y²+Z²), upsampled to 100 Hz
    acc_xyz       : np.ndarray (float32), shape (3, 3000) — bandpassed ACC [X, Y, Z] at 100 Hz
    sampling_rate : 100.0
    ID            : str — stable participant ID (MRN from master log, else folder name)
    rhythm_label  : str — from master log, empty string if unavailable
    source        : "vsm_preop"
    source_file   : str — path to CombinedData CSV relative to root
    """
    root = Path(root)

    log_lookup: dict[str, dict] = {}
    if master_log_path is not None:
        master_log_path = Path(master_log_path)
        if master_log_path.exists():
            log_lookup = _load_master_log(master_log_path)
        else:
            logger.warning("Master log not found: %s", master_log_path)

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("VSM_ih_"):
            continue
        folder_id = sub.name
        meta = log_lookup.get(folder_id)
        if meta is None:
            if log_lookup:
                logger.warning("Folder %s not in master log — skipping", folder_id)
                continue
            meta = {"participant_id": folder_id, "rhythm_label": "", "usable": True}
        elif only_usable and not meta["usable"]:
            logger.warning("Folder %s marked excluded in master log — skipping", folder_id)
            continue

        csv_path = _find_combined_csv(sub)
        if csv_path is None:
            logger.warning("No Pre-OP CombinedData CSV found for %s — skipping", folder_id)
            continue

        result = _read_signals(csv_path)
        if result is None:
            continue
        ppg, acc_mag, acc_xyz_full = result

        n = ppg.shape[1]
        if n < _WINDOW_SAMPLES:
            logger.warning("%s: too short (%d < %d samples), skipping", folder_id, n, _WINDOW_SAMPLES)
            continue

        logger.info(
            "%s: PPG=%d samples (%.1f min) → generating 30-s windows",
            folder_id, n, n / _PPG_FS / 60,
        )

        source_file = str(csv_path.relative_to(root).as_posix())
        participant_id = meta["participant_id"]
        rhythm_label = meta["rhythm_label"]
        mapped_label = VSM_preop_MAPPING.get(rhythm_label)
        if mapped_label is None and rhythm_label:
            logger.warning("%s: unmapped rhythm_label %r — stored as None", folder_id, rhythm_label)

        for start in range(0, n - _WINDOW_SAMPLES + 1, _STEP_SAMPLES):
            ppg_window     = ppg[:, start : start + _WINDOW_SAMPLES]          # (6, 3000)
            acc_window     = acc_mag[start : start + _WINDOW_SAMPLES]         # (3000,)
            acc_xyz_window = acc_xyz_full[:, start : start + _WINDOW_SAMPLES] # (3, 3000)
            activity = "yes" if float(acc_window.max()) >= 2.5 else "no"

            yield {
                "signal":  ppg_window,
                "acc":     acc_window,
                "acc_xyz": acc_xyz_window,
                "sampling_rate": _PPG_FS,
                "ID": participant_id,
                "rhythm_label": mapped_label,
                "activity": activity,
                "source": "vsm_preop",
                "source_file": source_file,
            }
