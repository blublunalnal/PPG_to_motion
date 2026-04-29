"""Build a unified PPG-to-motion dataset from IEEE DSP Cup and PPG-DaLiA.

Output layout
-------------
<output>/
  metadata.db          SQLite database (table: segments)
  metadata.csv         CSV mirror of the database
  build.log            Warnings / errors from IO modules
  signals/
    ieee/
      shard_000000.tar
      ...
    ppg-dalia/
      shard_000000.tar
      ...

Each tar shard contains numpy arrays named 000001.npy, 000002.npy, … (per shard).

Usage
-----
  python -m ppg_to_motion.datasets.builder \\
      --ieee-root  C:/datasets/IEEE_DSP_CUP \\
      --dalia-root C:/datasets/ppg+dalia \\
      --output     C:/develop/PPG_to_motion/output \\
      [--shard-size 5000] [--compress]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from ppg_to_motion.io.ieee import ieee_generator
from ppg_to_motion.io.ppg_dalia import ppg_dalia_generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    subject_id    TEXT,
    sampling_rate REAL,
    label         REAL,
    source_file   TEXT,
    segment_index INTEGER,
    tar_file      TEXT    NOT NULL,
    signal_key    TEXT    NOT NULL,
    signal_shape  TEXT    NOT NULL,
    signal_dtype  TEXT    NOT NULL DEFAULT 'float32'
);
"""

_DB_COLUMNS = (
    "source", "subject_id", "sampling_rate", "label",
    "source_file", "segment_index", "tar_file", "signal_key",
    "signal_shape", "signal_dtype",
)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------
class _ShardWriter:
    """Write float32 numpy arrays into rotating per-source tar shards."""

    def __init__(self, signals_dir: Path, shard_size: int, compress: bool):
        self._signals_dir = signals_dir
        self._shard_size = shard_size
        self._mode = "w:gz" if compress else "w:"
        self._ext = ".tar.gz" if compress else ".tar"

        self._shard_idx: dict[str, int] = {}
        self._entry_idx: dict[str, int] = {}    # sequential across all shards
        self._shard_count: dict[str, int] = {}  # entries in current shard
        self._tars: dict[str, tarfile.TarFile] = {}

    # ------------------------------------------------------------------
    def write(self, source: str, signal: np.ndarray) -> tuple[str, str]:
        """Persist *signal* and return (tar_rel_path, signal_key)."""
        if source not in self._tars or self._shard_count.get(source, 0) >= self._shard_size:
            self._rotate(source)

        sig_f32 = np.asarray(signal, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, sig_f32)
        data = buf.getvalue()

        # Keys restart at 000001 inside every new shard
        local_idx = self._shard_count[source]  # 0-based inside shard
        key = f"{local_idx + 1:06d}.npy"
        info = tarfile.TarInfo(name=key)
        info.size = len(data)
        self._tars[source].addfile(info, io.BytesIO(data))

        tar_rel = self._current_tar_rel(source)
        self._shard_count[source] += 1
        return tar_rel, key

    def close(self):
        for tf in self._tars.values():
            tf.close()
        self._tars.clear()

    def __enter__(self) -> "_ShardWriter":
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    def _rotate(self, source: str):
        if source in self._tars:
            self._tars[source].close()
        shard_no = self._shard_idx.get(source, 0)
        source_dir = self._signals_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"shard_{shard_no:06d}{self._ext}"
        self._tars[source] = tarfile.open(path, self._mode)
        self._shard_idx[source] = shard_no + 1
        self._shard_count[source] = 0

    def _current_tar_rel(self, source: str) -> str:
        shard_no = self._shard_idx[source] - 1   # current shard
        return str(
            Path("signals") / source / f"shard_{shard_no:06d}{self._ext}"
        )


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------
def build(
    output: Path,
    generators: list[tuple[str, Iterator[dict]]],
    shard_size: int = 5000,
    compress: bool = False,
) -> int:
    """Run the build and return the total number of segments written."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir(exist_ok=True)

    db_path = output / "metadata.db"
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA)
    con.commit()

    total = 0

    with _ShardWriter(output / "signals", shard_size, compress) as writer:
        for gen_label, gen in generators:
            seg_counters: dict[str, int] = {}

            for sample in tqdm(gen, desc=gen_label, unit="seg"):
                source: str = sample["source"]
                subject_id: str = str(sample.get("ID", ""))
                seg_idx = seg_counters.get(subject_id, 0)
                seg_counters[subject_id] = seg_idx + 1

                sig = np.asarray(sample["signal"], dtype=np.float32)
                tar_file, signal_key = writer.write(source, sig)

                label_val = sample.get("label")

                con.execute(
                    f"INSERT INTO segments ({','.join(_DB_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(_DB_COLUMNS))})",
                    (
                        source,
                        subject_id,
                        float(sample.get("sampling_rate", 0.0)),
                        float(label_val) if label_val is not None else None,
                        sample.get("source_file"),
                        seg_idx,
                        tar_file,
                        signal_key,
                        json.dumps(list(sig.shape)),
                        "float32",
                    ),
                )
                total += 1

            con.commit()
            logger.info("[%s] committed %d segments", gen_label, sum(seg_counters.values()))

    # Mirror to CSV
    df = pd.read_sql("SELECT * FROM segments", con)
    df.to_csv(output / "metadata.csv", index=False)
    con.close()

    logger.info("Build complete: %d total segments → %s", total, output)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PPG-to-motion dataset from IEEE DSP Cup and PPG-DaLiA"
    )
    p.add_argument("--ieee-root", type=Path, metavar="DIR",
                   help="Path to IEEE DSP Cup directory")
    p.add_argument("--dalia-root", type=Path, metavar="DIR",
                   help="Path to PPG-DaLiA root directory")
    p.add_argument("--output", type=Path, required=True, metavar="DIR",
                   help="Output directory for dataset")
    p.add_argument("--shard-size", type=int, default=5000, metavar="N",
                   help="Maximum segments per tar shard (default: 5000)")
    p.add_argument("--compress", action="store_true",
                   help="Gzip-compress tar shards (.tar.gz)")
    return p.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output / "build.log" if args.output.exists()
                                else Path("build.log")),
        ],
    )

    generators: list[tuple[str, Iterator[dict]]] = []
    if args.ieee_root:
        generators.append(("ieee", ieee_generator(args.ieee_root)))
    if args.dalia_root:
        generators.append(("ppg-dalia", ppg_dalia_generator(args.dalia_root)))

    if not generators:
        print("Error: specify at least one of --ieee-root or --dalia-root")
        raise SystemExit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    total = build(args.output, generators, args.shard_size, args.compress)
    print(f"\nDone. {total} segments written to {args.output}")


if __name__ == "__main__":
    main()
