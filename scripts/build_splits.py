"""Construye data/processed/splits.parquet con folds train/val/test_clean/test_hard.

Política tiered (por especie según N):
    N >= 50: 70/15/15 puro
    20 <= N < 50: 70/15/15 con min 3 en val y 3 en test
    10 <= N < 20: 70/15/15 con min 2 en val y 2 en test
    N < 10:      todo a train (0 val, 0 test)

Las filas con split=test_hard quedan intactas (fold=test_hard).
Seed fijo (42) para reproducibilidad.

Uso:
    python scripts/build_splits.py             # dry-run, muestra el plan
    python scripts/build_splits.py --apply     # escribe splits.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
OUT_PATH = ROOT / "data" / "processed" / "splits.parquet"
SEED = 42


def tier_counts(n: int) -> tuple[int, int, int]:
    """Devuelve (train, val, test_clean) para un species con N samples."""
    if n >= 50:
        n_test = round(n * 0.15)
        n_val = round(n * 0.15)
    elif n >= 20:
        n_test = max(3, round(n * 0.15))
        n_val = max(3, round(n * 0.15))
    elif n >= 10:
        n_test = max(2, round(n * 0.15))
        n_val = max(2, round(n * 0.15))
    else:
        n_test = 0
        n_val = 0
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def assign_folds(emb_df: pd.DataFrame) -> pd.DataFrame:
    """Genera DataFrame con (filepath, fold). Determinístico."""
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    train_only = emb_df[emb_df["split"] == "train"]
    test_hard_only = emb_df[emb_df["split"] == "test_hard"]

    for sp, group in train_only.groupby("species"):
        n = len(group)
        n_train, n_val, n_test = tier_counts(n)
        files = group["filepath"].to_numpy()
        rng.shuffle(files)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        for f in train_files:
            rows.append({"filepath": f, "fold": "train"})
        for f in val_files:
            rows.append({"filepath": f, "fold": "val"})
        for f in test_files:
            rows.append({"filepath": f, "fold": "test_clean"})

    for f in test_hard_only["filepath"]:
        rows.append({"filepath": f, "fold": "test_hard"})

    return pd.DataFrame(rows)


def report(emb_df: pd.DataFrame, splits_df: pd.DataFrame) -> None:
    merged = emb_df.merge(splits_df, on="filepath")
    pivot = (
        merged.groupby(["species", "fold"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test_clean", "test_hard"], fill_value=0)
    )
    pivot["total_train_pool"] = pivot["train"] + pivot["val"] + pivot["test_clean"]
    pivot = pivot.sort_values("total_train_pool", ascending=False)
    print(pivot.to_string())
    print()
    print("Totales por fold:")
    print(merged.groupby("fold").size().to_string())
    print(f"\nGran total: {len(merged)} filas")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Escribir splits.parquet (default: dry-run)")
    args = parser.parse_args()

    if not EMB_PATH.exists():
        print(f"! Falta {EMB_PATH}", file=sys.stderr)
        return 1

    emb = pd.read_parquet(EMB_PATH, columns=["filepath", "species", "split"])
    splits = assign_folds(emb)

    # Sanity checks
    assert len(splits) == len(emb), \
        f"counts mismatch: splits={len(splits)} vs embeddings={len(emb)}"
    assert splits["filepath"].is_unique, "filepath duplicado en splits"
    assert set(splits["filepath"]) == set(emb["filepath"]), "filepaths no coinciden"

    report(emb, splits)

    if args.apply:
        splits.to_parquet(OUT_PATH, index=False)
        print(f"\nEscrito: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    else:
        print("\nDry-run: no se escribió. Re-ejecutar con --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
