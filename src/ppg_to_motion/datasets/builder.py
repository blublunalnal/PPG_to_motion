"""Build unified PPG-DaLiA + VSM Pre-OP dataset as a HuggingFace Dataset.

Signals and pre-computed images are stored at 100 Hz (30 s = 3000 samples).
STFT and differential images are computed once during build and stored in the
Arrow files so DataLoader workers only do I/O during training.

Output
------
<output>/
  dataset/   HuggingFace Dataset (Arrow files, memory-mapped)
  build.log

Stored arrays per segment
-------------------------
  signal        float32 (3000,)        — raw PPG channel 0 at 100 Hz
  acc           float32 (3000,)        — ACC magnitude at 100 Hz
  ppg_channels  float32 (6, 3000)      — all 6 PPG channels (VSM only; None for DaLiA)
  diff_image    float32 (3, 3000)      — [x, dx, d²x] z-scored, built from channel 0
  stft_image    float32 (3, 65, 188)   — [z(log|X|), z(cos φ), z(sin φ)], built from channel 0

STFT parameters (n_fft=128, hop=16, center=True, onesided → 65 freq bins × 188 frames)
are fixed at build time and baked into the stored shape.

Usage
-----
  python -m ppg_to_motion.datasets.builder \\
      [--dalia-root DIR] [--vsm-root DIR] [--vsm-log FILE] \\
      --output DIR
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import Array2D, Array3D, Dataset, Features, Sequence, Value
from tqdm import tqdm

from ppg_to_motion.io.ppg_dalia import ppg_dalia_generator
from ppg_to_motion.io.vsm_preop import vsm_generator
from ppg_to_motion.preprocessing.signal_image_2d import (
    generate_differential_signal_image,
    generate_stft_features,
    normalize_multichannel_image,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed dataset constants (must match signal_image_2d defaults)
# ---------------------------------------------------------------------------
_SIGNAL_LEN      = 3000   # 30 s × 100 Hz
_STFT_N_FFT      = 128
_STFT_HOP        = 16
_STFT_FREQS      = 65     # n_fft // 2 + 1
_STFT_FRAMES     = 188    # ceil(3000 / 16) + 1 with center=True
_N_PPG_CHANNELS  = 6      # VSM: S1, S2, S1.1, S2.1, S1.2, S2.2

FEATURES = Features({
    "source":        Value("string"),
    "subject_id":    Value("string"),
    "sampling_rate": Value("float32"),
    "label":         Value("float32"),   # HR in BPM — null for VSM
    "activity":      Value("string"),    # DaLiA only — null for VSM
    "rhythm_label":  Value("string"),    # VSM only — null for DaLiA
    "source_file":   Value("string"),
    "segment_index": Value("int32"),
    "signal":        Sequence(Value("float32"), length=_SIGNAL_LEN),
    "acc":           Sequence(Value("float32"), length=_SIGNAL_LEN),
    "ppg_channels":  Array2D(shape=(_N_PPG_CHANNELS, _SIGNAL_LEN), dtype="float32"),  # VSM only — null for DaLiA
    "diff_image":    Array2D(shape=(3, _SIGNAL_LEN), dtype="float32"),
    "stft_image":    Array3D(shape=(3, _STFT_FREQS, _STFT_FRAMES), dtype="float32"),
})


def _fix_len(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)), mode="edge")


def _iter_samples(
    dalia_root: str | None,
    vsm_root: str | None,
    vsm_log: str | None,
) -> Iterator[dict]:
    generators: list[tuple[str, Iterator[dict]]] = []
    if dalia_root:
        generators.append(("ppg-dalia", ppg_dalia_generator(Path(dalia_root))))
    if vsm_root:
        generators.append(("vsm_preop", vsm_generator(
            Path(vsm_root), Path(vsm_log) if vsm_log else None
        )))

    seg_counters: dict[str, int] = {}
    for gen_label, gen in generators:
        n_gen = 0
        for sample in tqdm(gen, desc=gen_label, unit="seg"):
            subject_id = str(sample.get("ID", ""))
            seg_idx = seg_counters.get(subject_id, 0)
            seg_counters[subject_id] = seg_idx + 1

            signal_raw = np.asarray(sample["signal"], dtype=np.float32)
            if signal_raw.ndim == 2:  # VSM: (n_channels, N)
                ppg_channels = np.stack(
                    [_fix_len(signal_raw[i], _SIGNAL_LEN) for i in range(signal_raw.shape[0])],
                    axis=0,
                )
                ch0 = ppg_channels[0]
            else:                     # DaLiA: (N,)
                ch0 = _fix_len(signal_raw, _SIGNAL_LEN)
                ppg_channels = None
            signal = ch0

            acc    = _fix_len(
                np.asarray(sample.get("acc", np.zeros(_SIGNAL_LEN, np.float32)), dtype=np.float32),
                _SIGNAL_LEN,
            )

            diff_image = normalize_multichannel_image(
                generate_differential_signal_image(ch0), method="zscore"
            )
            stft_image = generate_stft_features(
                ch0, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP
            )

            label_val = sample.get("label")
            n_gen += 1
            yield {
                "source":        sample["source"],
                "subject_id":    subject_id,
                "sampling_rate": float(sample.get("sampling_rate", 100.0)),
                "label":         float(label_val) if label_val is not None else None,
                "activity":      sample.get("activity") or None,
                "rhythm_label":  sample.get("rhythm_label") or None,
                "source_file":   sample.get("source_file") or None,
                "segment_index": seg_idx,
                "signal":        signal,
                "acc":           acc,
                "ppg_channels":  ppg_channels,
                "diff_image":    diff_image,
                "stft_image":    stft_image,
            }
        logger.info("[%s] %d segments", gen_label, n_gen)


def build(
    output: Path,
    dalia_root: Path | None = None,
    vsm_root: Path | None = None,
    vsm_log: Path | None = None,
    writer_batch_size: int = 1000,
) -> Dataset:
    """Build and save the HuggingFace dataset; return it."""
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir = output / "dataset"

    ds = Dataset.from_generator(
        _iter_samples,
        gen_kwargs={
            "dalia_root": str(dalia_root) if dalia_root else None,
            "vsm_root":   str(vsm_root)   if vsm_root   else None,
            "vsm_log":    str(vsm_log)    if vsm_log    else None,
        },
        features=FEATURES,
        writer_batch_size=writer_batch_size,
    )
    ds.save_to_disk(str(dataset_dir))
    logger.info("Build complete: %d segments → %s", len(ds), dataset_dir)
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PPG-to-motion dataset from PPG-DaLiA and VSM Pre-OP"
    )
    p.add_argument("--dalia-root", type=Path, metavar="DIR")
    p.add_argument("--vsm-root",   type=Path, metavar="DIR")
    p.add_argument("--vsm-log",    type=Path, metavar="FILE",
                   help="Optional Excel master log for VSM participant IDs and labels")
    p.add_argument("--output",     type=Path, required=True, metavar="DIR")
    p.add_argument("--writer-batch-size", type=int, default=1000, metavar="N",
                   help="Rows buffered before each Arrow flush (default: 1000)")
    return p.parse_args()


def main():
    args = _parse_args()

    log_path = args.output / "build.log" if args.output.exists() else Path("build.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    if not args.dalia_root and not args.vsm_root:
        print("Error: specify at least one of --dalia-root or --vsm-root")
        raise SystemExit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    ds = build(
        args.output,
        dalia_root=args.dalia_root,
        vsm_root=args.vsm_root,
        vsm_log=args.vsm_log,
        writer_batch_size=args.writer_batch_size,
    )
    print(f"\nDone. {len(ds)} segments → {args.output / 'dataset'}")


if __name__ == "__main__":
    main()
