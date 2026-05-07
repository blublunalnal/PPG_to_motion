"""Create train/val/test splits from the HuggingFace dataset built by builder.py.

Rules
-----
- Subject-level: every segment from a subject goes to exactly one split.
- No subject appears in more than one split (verified at the end).
- ppg-dalia  : subjects split randomly (no activity stratification — every
               subject does every activity, so a random subject split already
               keeps activity balance across splits).
- vsm_preop  : subjects stratified by rhythm_label.
- Each source is split 70/10/20 independently, preserving the source ratio
  across all three splits.

Output
------
<dataset_dir>/splits/
  train/   HuggingFace Dataset (Arrow, memory-mapped)
  val/
  test/
  summary.txt

Loading for training
--------------------
  from datasets import load_from_disk

  train_ds = load_from_disk("output/splits/train").with_format("torch")
  loader = torch.utils.data.DataLoader(
      train_ds, batch_size=32, num_workers=4, shuffle=True
  )
  for batch in loader:
      stft  = batch["stft_image"]   # [B, 3, 65, 188]
      diff  = batch["diff_image"]   # [B, 3, 3000]
      label = batch["label"]        # [B]

Usage
-----
  python -m ppg_to_motion.datasets.make_splits --dataset-dir /path/to/output
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


# ---------------------------------------------------------------------------
# Subject-level stratified split
# ---------------------------------------------------------------------------

def _subject_split(
    subjects: np.ndarray,
    strat_labels: np.ndarray,
    seg_counts: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[set, set, set]:
    """Assign subjects to train/val/test, balancing segment counts within each stratum.

    Within each stratum subjects are sorted largest-first (ties broken randomly),
    then each subject is assigned to whichever split is furthest below its target
    segment fraction. This avoids one large subject skewing a split.
    """
    train_s, val_s, test_s = set(), set(), set()
    split_sets = [train_s, val_s, test_s]

    for label in np.unique(strat_labels):
        mask = strat_labels == label
        grp_subjects = subjects[mask]
        grp_counts   = seg_counts[mask].astype(np.int64)

        # Shuffle first so ties in segment count are broken randomly
        shuf = rng.permutation(len(grp_subjects))
        grp_subjects = grp_subjects[shuf]
        grp_counts   = grp_counts[shuf]

        # Sort descending by segment count (stable preserves shuffle order for ties)
        order = np.argsort(-grp_counts, kind="stable")
        grp_subjects = grp_subjects[order]
        grp_counts   = grp_counts[order]

        assigned = np.zeros(3, dtype=np.int64)  # cumulative segments per split

        for subj, cnt in zip(grp_subjects, grp_counts):
            total = assigned.sum()
            fracs    = assigned / total if total > 0 else np.zeros(3)
            deficits = np.array(ratios) - fracs
            idx = int(np.argmax(deficits))
            split_sets[idx].add(subj)
            assigned[idx] += cnt

    return train_s, val_s, test_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_splits(
    dataset_dir: Path,
    ratios: tuple[float, float, float] = (0.70, 0.10, 0.20),
    seed: int = 42,
) -> None:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)

    ds = load_from_disk(str(dataset_dir / "dataset"))
    df = ds.select_columns(
        ["source", "subject_id", "activity", "rhythm_label"]
    ).to_pandas()
    df["_idx"] = np.arange(len(df))

    df = df[df["source"].isin(["ppg-dalia", "vsm_preop"])]
    if df.empty:
        raise RuntimeError("No ppg-dalia or vsm_preop segments found in dataset")

    split_col = pd.Series("", index=df.index, dtype=str)

    # --- PPG-DaLiA: no activity stratification; every subject does every activity
    #     so a random subject split already balances activity across splits ---
    dalia = df[df["source"] == "ppg-dalia"]
    if not dalia.empty:
        dalia_subjects = dalia["subject_id"].unique()
        dalia_labels   = np.full(len(dalia_subjects), "dalia")
        dalia_counts   = dalia.groupby("subject_id").size().reindex(dalia_subjects).values
        tr, va, te = _subject_split(
            dalia_subjects, dalia_labels, dalia_counts, ratios, rng
        )
        split_col[dalia[dalia["subject_id"].isin(tr)].index] = "train"
        split_col[dalia[dalia["subject_id"].isin(va)].index] = "val"
        split_col[dalia[dalia["subject_id"].isin(te)].index] = "test"

    # --- VSM Pre-OP: rhythm_label per subject ---
    vsm = df[df["source"] == "vsm_preop"]
    if not vsm.empty:
        subj_rhythm = (
            vsm.groupby("subject_id")["rhythm_label"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown")
        )
        vsm_counts = vsm.groupby("subject_id").size().reindex(subj_rhythm.index).values
        tr, va, te = _subject_split(
            subj_rhythm.index.values, subj_rhythm.values, vsm_counts, ratios, rng
        )
        split_col[vsm[vsm["subject_id"].isin(tr)].index] = "train"
        split_col[vsm[vsm["subject_id"].isin(va)].index] = "val"
        split_col[vsm[vsm["subject_id"].isin(te)].index] = "test"

    df["split"] = split_col

    # Verify no subject leakage
    for src in ["ppg-dalia", "vsm_preop"]:
        src_df = df[df["source"] == src]
        sets = {
            name: set(src_df[src_df["split"] == name]["subject_id"])
            for name in ("train", "val", "test")
        }
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise RuntimeError(
                    f"Subject leak in {src} between {a} and {b}: {overlap}"
                )

    # Save split datasets to disk
    out_dir = dataset_dir / "splits"
    out_dir.mkdir(exist_ok=True)

    lines: list[str] = []
    for name in ("train", "val", "test"):
        idx = df[df["split"] == name]["_idx"].tolist()
        ds.select(idx).save_to_disk(str(out_dir / name))

        split_df = df[df["split"] == name]
        n_dalia = (split_df["source"] == "ppg-dalia").sum()
        n_vsm   = (split_df["source"] == "vsm_preop").sum()
        total   = len(split_df)
        lines.append(
            f"{name:5s}: {total:6d} segments  "
            f"ppg-dalia={n_dalia} ({100*n_dalia/max(1,total):.1f}%)  "
            f"vsm_preop={n_vsm} ({100*n_vsm/max(1,total):.1f}%)"
        )

    lines.append("")
    for src in ["ppg-dalia", "vsm_preop"]:
        src_df = df[df["source"] == src]
        if src_df.empty:
            continue
        lines.append(f"--- {src} subjects ---")
        for name in ("train", "val", "test"):
            subs = sorted(src_df[src_df["split"] == name]["subject_id"].unique())
            lines.append(f"  {name:5s}: {len(subs)} subjects — {subs}")

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nSplits written to {out_dir}")
    print("Load with: load_from_disk('output/splits/train').with_format('torch')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Subject-level 70/10/20 train/val/test split of the HF dataset.
            ppg-dalia split randomly (activity balance follows naturally);
            vsm_preop stratified by rhythm_label per subject.
        """)
    )
    p.add_argument(
        "--dataset-dir", type=Path, required=True, metavar="DIR",
        help="Output directory produced by builder.py (contains dataset/)"
    )
    p.add_argument("--train-frac", type=float, default=0.70, metavar="F")
    p.add_argument("--val-frac",   type=float, default=0.10, metavar="F")
    p.add_argument("--test-frac",  type=float, default=0.20, metavar="F")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ratios = (args.train_frac, args.val_frac, args.test_frac)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(
            f"Fractions must sum to 1.0 (got {sum(ratios):.4f})"
        )
    make_splits(args.dataset_dir, ratios=ratios, seed=args.seed)
