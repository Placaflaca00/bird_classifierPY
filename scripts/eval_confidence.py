"""Diagnóstico de confianza del modelo sobre test_clean vs test_hard.

Carga el checkpoint indicado (default: ``checkpoints/baseline-v0/best-*.ckpt``)
y los embeddings/splits locales. Para cada sample no-aumentado de test_clean y
test_hard computa softmax y reporta:

    * Counts por especie x fold (insumo para decidir clase "Otros").
    * Confianza promedio (max softmax) y proba de la clase verdadera por
      (especie, fold).
    * Resumen agregado para las especies "well-supported" (>=100 train).
    * Boxplot de confianza por fold (well-supported vs pobres).

Salida: CSVs + PNG en ``reports/confidence_diagnostic/``.

Uso:
    python scripts/eval_confidence.py
    python scripts/eval_confidence.py --ckpt path/al/best.ckpt
    python scripts/eval_confidence.py --emb data/processed/embeddings.baseline-v0.parquet
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

DEFAULT_EMB = ROOT / "data" / "processed" / "embeddings.parquet"
DEFAULT_SPLITS = ROOT / "data" / "processed" / "splits.parquet"
DEFAULT_CKPT_GLOB = "checkpoints/baseline-v0/best-*.ckpt"
OUT_DIR = ROOT / "reports" / "confidence_diagnostic"


def species_mapping_from(emb_path: Path) -> dict[str, int]:
    """Mismo orden alfabético que ``src.data.dataset.load_species_mapping``,
    pero leyendo desde un parquet arbitrario (para soportar la versión
    congelada de embeddings.baseline-v0.parquet)."""
    df = pd.read_parquet(emb_path, columns=["species"])
    species = sorted(df["species"].unique())
    return {sp: i for i, sp in enumerate(species)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help=f"Checkpoint. Default: glob {DEFAULT_CKPT_GLOB}")
    parser.add_argument("--emb", default=str(DEFAULT_EMB),
                        help="Parquet de embeddings.")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS),
                        help="Parquet de splits.")
    args = parser.parse_args()

    if args.ckpt is None:
        cands = sorted(glob(str(ROOT / DEFAULT_CKPT_GLOB)))
        if not cands:
            raise SystemExit(f"No se encontró checkpoint en {DEFAULT_CKPT_GLOB}")
        ckpt_path = cands[-1]
    else:
        ckpt_path = args.ckpt
    emb_path = Path(args.emb)
    splits_path = Path(args.splits)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Embeddings: {emb_path}")
    print(f"Splits:     {splits_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    species_to_idx = species_mapping_from(emb_path)
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    num_classes = len(species_to_idx)

    model = BirdClassifier.load_from_checkpoint(
        ckpt_path, strict=False, map_location="cpu"
    )
    model.eval()
    ckpt_num_classes = model.net[-1].out_features
    if ckpt_num_classes != num_classes:
        raise SystemExit(
            f"num_classes mismatch: ckpt tiene {ckpt_num_classes} pero el "
            f"mapping de embeddings tiene {num_classes}. Probá pasar "
            f"--emb data/processed/embeddings.baseline-v0.parquet"
        )
    print(f"num_classes (ckpt y mapping coinciden): {num_classes}")

    emb = pd.read_parquet(emb_path)
    splits = pd.read_parquet(splits_path)
    full = emb.merge(splits, on="filepath", how="inner")
    if "is_aug" in full.columns:
        full = full[~full["is_aug"]]
    full = full.reset_index(drop=True)

    eval_df = full[full["fold"].isin(["test_clean", "test_hard"])].reset_index(drop=True)
    print(f"\nEval rows (no-aug, test_clean+test_hard): {len(eval_df)}")
    print("Per-fold:", eval_df.groupby("fold").size().to_dict())

    emb_arr = np.stack(eval_df["embedding"].to_numpy()).astype(np.float32)
    x = torch.from_numpy(emb_arr)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    eval_df["pred_idx"] = pred_idx
    eval_df["pred_species"] = [idx_to_species[i] for i in pred_idx]
    eval_df["true_idx"] = eval_df["species"].map(species_to_idx).astype(int)
    eval_df["correct"] = eval_df["true_idx"] == eval_df["pred_idx"]
    eval_df["max_prob"] = max_prob
    eval_df["true_class_prob"] = probs[np.arange(len(eval_df)), eval_df["true_idx"].to_numpy()]

    # Counts globales por especie x fold (insumo para PASO 2 / "Otros")
    counts = (
        full.groupby(["species", "fold"]).size().unstack(fill_value=0)
            .reindex(columns=["train", "val", "test_clean", "test_hard"], fill_value=0)
            .assign(total=lambda d: d.sum(axis=1))
            .sort_values("train", ascending=False)
    )
    counts.to_csv(OUT_DIR / "species_counts.csv")
    print("\n=== Counts por especie (no-aug) ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(counts.to_string())

    # Confianza por especie x fold
    g = (eval_df.groupby(["species", "fold"]).agg(
            n=("max_prob", "size"),
            conf_mean=("max_prob", "mean"),
            conf_median=("max_prob", "median"),
            true_prob_mean=("true_class_prob", "mean"),
            acc=("correct", "mean"),
        ).round(4).unstack("fold"))
    g.to_csv(OUT_DIR / "confidence_by_species.csv")
    print("\n=== Confianza + accuracy por especie x fold ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(g.to_string())

    # Well-supported = >=100 train (criterio inicial; ajustable después)
    well = counts[counts["train"] >= 100].index.tolist()
    poor = [s for s in counts.index if s not in well]
    print(f"\nWell-supported (>=100 train): {len(well)} especies")
    for s in well:
        print(f"  - {s} (train={counts.loc[s,'train']})")
    print(f"\nPobres en data: {len(poor)} especies")
    for s in poor:
        print(f"  - {s} (train={counts.loc[s,'train']}  hard={counts.loc[s,'test_hard']})")

    well_df = eval_df[eval_df["species"].isin(well)]
    summary = well_df.groupby("fold").agg(
        n=("max_prob", "size"),
        conf_mean=("max_prob", "mean"),
        conf_median=("max_prob", "median"),
        true_prob_mean=("true_class_prob", "mean"),
        acc=("correct", "mean"),
    ).round(4)
    summary.to_csv(OUT_DIR / "well_supported_summary.csv")
    print("\n=== Resumen well-supported (clean vs hard) ===")
    print(summary.to_string())
    if {"test_clean", "test_hard"}.issubset(summary.index):
        d_conf = summary.loc["test_clean", "conf_mean"] - summary.loc["test_hard", "conf_mean"]
        d_acc = summary.loc["test_clean", "acc"] - summary.loc["test_hard", "acc"]
        print(f"\nDelta well-supported  conf_mean(clean - hard) = {d_conf:+.4f}")
        print(f"Delta well-supported  acc(clean - hard)       = {d_acc:+.4f}")

    if poor:
        poor_df = eval_df[eval_df["species"].isin(poor)]
        poor_summary = poor_df.groupby("fold").agg(
            n=("max_prob", "size"),
            conf_mean=("max_prob", "mean"),
            true_prob_mean=("true_class_prob", "mean"),
            acc=("correct", "mean"),
        ).round(4)
        poor_summary.to_csv(OUT_DIR / "poor_summary.csv")
        print("\n=== Resumen pobres-en-data (clean vs hard) ===")
        print(poor_summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    folds = ["test_clean", "test_hard"]
    data_well = [well_df[well_df["fold"] == f]["max_prob"].to_numpy() for f in folds]
    axes[0].boxplot(data_well, tick_labels=folds, showmeans=True)
    axes[0].set_title(f"Confianza (max softmax) — well-supported ({len(well)} sp)")
    axes[0].set_ylabel("max softmax")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)

    if poor:
        poor_df_local = eval_df[eval_df["species"].isin(poor)]
        data_poor = [poor_df_local[poor_df_local["fold"] == f]["max_prob"].to_numpy() for f in folds]
        axes[1].boxplot(data_poor, tick_labels=folds, showmeans=True)
        axes[1].set_title(f"Confianza (max softmax) — pobres ({len(poor)} sp)")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = OUT_DIR / "confidence_boxplot.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nReportes guardados en: {OUT_DIR}")
    print(f"  - species_counts.csv")
    print(f"  - confidence_by_species.csv")
    print(f"  - well_supported_summary.csv")
    print(f"  - poor_summary.csv")
    print(f"  - confidence_boxplot.png")


if __name__ == "__main__":
    main()
