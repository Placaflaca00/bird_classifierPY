"""Detecta samples sospechosos de label noise en test_clean y test_hard.

Approach simple, sin cleanlab: como test_clean y test_hard nunca se usaron
para training (el modelo wa-drop3-v1 no los vio), las probas de softmax sobre
esos folds son out-of-sample naturalmente. No hace falta cross-val.

Por cada sample del test:
    - max_prob:        confianza del modelo en su predicción (max softmax)
    - true_class_prob: prob que el modelo le asignó a la clase verdadera
    - correct:         si pred == true

Score de severity (mayor = más sospechoso):
    severity = max_prob * (1 - true_class_prob) si correct=False, else 0

Interpretación: un sample con severity alto significa "el modelo está MUY
seguro de otra clase Y casi descartó la clase de la etiqueta". Eso huele a
label noise (audio mal etiquetado, mala calidad, o pájaro no audible).

Salida en reports/label_issues/:
    - test_hard_suspicious.csv  (top-N candidatos)
    - test_clean_suspicious.csv
    - per_species_summary.csv   (cuántos sospechosos por (fold, especie))

Uso:
    python scripts/find_label_issues.py
    python scripts/find_label_issues.py --top-n 50 --conf-threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"
DEFAULT_CKPT_GLOB = "checkpoints/wa-drop3-v1/best-*.ckpt"
OUT_DIR = ROOT / "reports" / "label_issues"


def species_mapping(emb_path: Path) -> dict[str, int]:
    df = pd.read_parquet(emb_path, columns=["species"])
    species = sorted(df["species"].unique())
    return {sp: i for i, sp in enumerate(species)}


def load_eval_df() -> pd.DataFrame:
    emb = pd.read_parquet(EMB_PATH)
    splits = pd.read_parquet(SPLITS_PATH)
    df = emb.merge(splits, on="filepath", how="inner")
    if "is_aug" in df.columns:
        df = df[~df["is_aug"]]
    df = df[df["fold"].isin(["test_clean", "test_hard"])]
    return df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help=f"Default: glob {DEFAULT_CKPT_GLOB}")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Top N sospechosos por fold a reportar.")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Para considerar 'modelo confiado pero incorrecto'.")
    parser.add_argument("--drop-threshold", type=float, default=0.1,
                        help="Para considerar 'modelo descartó clase real'.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ckpt is None:
        cands = sorted(glob(str(ROOT / DEFAULT_CKPT_GLOB)))
        if not cands:
            print(f"No checkpoint en {DEFAULT_CKPT_GLOB}", file=sys.stderr)
            return 2
        ckpt = cands[-1]
    else:
        ckpt = args.ckpt
    print(f"Checkpoint: {ckpt}")

    species_to_idx = species_mapping(EMB_PATH)
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    num_classes = len(species_to_idx)
    print(f"num_classes: {num_classes}")

    model = BirdClassifier.load_from_checkpoint(ckpt, strict=False, map_location="cpu")
    model.eval()
    if model.net[-1].out_features != num_classes:
        print(f"! ckpt num_classes mismatch", file=sys.stderr)
        return 3

    df = load_eval_df()
    print(f"Eval rows: {len(df)}  (clean={int((df['fold']=='test_clean').sum())}, "
          f"hard={int((df['fold']=='test_hard').sum())})")

    emb_arr = np.stack(df["embedding"].apply(np.asarray, args=(np.float32,)).to_numpy())
    with torch.no_grad():
        logits = model(torch.from_numpy(emb_arr))
        probs = F.softmax(logits, dim=1).numpy()

    df = df[["filepath", "fold", "species"]].copy()
    df["true_idx"] = df["species"].map(species_to_idx).astype(int)
    df["pred_idx"] = probs.argmax(axis=1)
    df["pred_species"] = [idx_to_species[i] for i in df["pred_idx"]]
    df["max_prob"] = probs.max(axis=1)
    df["true_class_prob"] = probs[np.arange(len(df)), df["true_idx"].to_numpy()]
    df["correct"] = df["true_idx"] == df["pred_idx"]
    # severity 0 si acertó, sino max_prob * (1 - true_class_prob)
    df["severity"] = np.where(
        df["correct"],
        0.0,
        df["max_prob"] * (1.0 - df["true_class_prob"]),
    )

    # === Per-species summary ===
    summary_rows = []
    for fold in ("test_clean", "test_hard"):
        sub = df[df["fold"] == fold]
        for sp, g in sub.groupby("species"):
            n = len(g)
            n_wrong = int((~g["correct"]).sum())
            n_high_conf_wrong = int(((~g["correct"]) & (g["max_prob"] >= args.conf_threshold)).sum())
            n_dropped = int((g["true_class_prob"] < args.drop_threshold).sum())
            summary_rows.append({
                "fold": fold,
                "species": sp,
                "n": n,
                "n_wrong": n_wrong,
                "n_high_conf_wrong": n_high_conf_wrong,
                "n_dropped_true_class": n_dropped,
                "max_severity": float(g["severity"].max()),
                "mean_severity": float(g["severity"].mean()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["fold", "max_severity"], ascending=[True, False])
    summary.to_csv(OUT_DIR / "per_species_summary.csv", index=False)
    print("\n=== Per-species summary (orden: max severity descendente) ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(summary.round(4).to_string(index=False))

    # === Top-N suspicious por fold ===
    for fold in ("test_clean", "test_hard"):
        sub = df[df["fold"] == fold].copy()
        suspicious = sub[~sub["correct"]].sort_values("severity", ascending=False)
        cols = ["filepath", "fold", "species", "pred_species", "max_prob", "true_class_prob", "severity"]
        out_csv = OUT_DIR / f"{fold}_suspicious.csv"
        suspicious[cols].to_csv(out_csv, index=False)
        print(f"\n=== Top-{args.top_n} sospechosos en {fold} ===")
        print(suspicious[cols].head(args.top_n).round(4).to_string(index=False))
        print(f"\n  total wrong: {len(suspicious)}/{len(sub)}  "
              f"(high-conf wrong: {int((suspicious['max_prob'] >= args.conf_threshold).sum())})")
        print(f"  guardado: {out_csv}")

    print(f"\nReportes en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
