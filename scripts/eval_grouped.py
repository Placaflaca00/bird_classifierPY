"""Evalúa baseline-v0 con tres configuraciones para responder:
¿el modelo confunde las "raras" entre sí, o las confunde con no-raras?

  as-is : 23 clases, métricas normales.
  group : las especies en OTHERS se agrupan a UNA clase virtual "Otros"
          (en truth y pred). Métricas sobre 18 clases efectivas.
          Equivale a "el modelo no sabe cuál exactamente, pero acertó al grupo".
  drop  : samples cuya verdad ∈ OTHERS se eliminan del test set.
          Métricas sobre las 17 clases reales (cómo le va al modelo cuando
          olvidamos las raras del eval).

Salida: tabla CSV + print por (modo, fold) con accuracy y macro_f1.

Uso:
    python scripts/eval_grouped.py
    python scripts/eval_grouped.py --vad-cache reports/cleaning_eval/cleaning_per_window.parquet --vad-threshold -2.0
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
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"
DEFAULT_CKPT_GLOB = "checkpoints/baseline-v0/best-*.ckpt"
OUT_DIR = ROOT / "reports" / "grouping_eval"

OTHERS = [
    "Pipile jacutinga",
    "Rhea americana",
    "Phoenicopterus chilensis",
    "Jabiru mycteria",
    "Mycteria americana",
    "Eudromia formosa",
]


def species_mapping() -> dict[str, int]:
    df = pd.read_parquet(EMB_PATH, columns=["species"])
    species = sorted(df["species"].unique())
    return {sp: i for i, sp in enumerate(species)}


def load_eval_embeddings_raw() -> pd.DataFrame:
    """Embeddings originales (sin cleaning) para test_clean + test_hard, no-aug."""
    emb = pd.read_parquet(EMB_PATH)
    splits = pd.read_parquet(SPLITS_PATH)
    df = emb.merge(splits, on="filepath", how="inner")
    if "is_aug" in df.columns:
        df = df[~df["is_aug"]]
    df = df[df["fold"].isin(["test_clean", "test_hard"])]
    return df.reset_index(drop=True)


def load_eval_embeddings_vad(cache_path: Path, threshold: float) -> pd.DataFrame:
    """Reagrega cache per-window con VAD threshold y devuelve un DataFrame
    con la misma forma que load_eval_embeddings_raw (filepath, species, fold,
    embedding)."""
    cache = pd.read_parquet(cache_path)
    sub = cache[cache["source"] == "raw"]
    rows = []
    for (fp, fold, sp), grp in sub.groupby(["filepath", "fold", "species"], sort=False):
        embs = np.stack(grp["embedding"].apply(np.asarray, args=(np.float32,)).to_numpy())
        scores = grp["score"].to_numpy()
        mask = scores >= threshold
        emb_pooled = embs[mask].mean(axis=0) if mask.any() else embs.mean(axis=0)
        rows.append({"filepath": fp, "species": sp, "fold": fold, "embedding": emb_pooled.tolist()})
    return pd.DataFrame(rows)


def metrics_for(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> dict:
    if len(true_idx) == 0:
        return {"n": 0, "acc": float("nan"), "macro_f1": float("nan")}
    acc = float((true_idx == pred_idx).mean())
    f1 = float(f1_score(true_idx, pred_idx, average="macro",
                        labels=list(range(n_classes)), zero_division=0))
    return {"n": int(len(true_idx)), "acc": round(acc, 4), "macro_f1": round(f1, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help=f"Checkpoint. Default: glob {DEFAULT_CKPT_GLOB}")
    parser.add_argument("--vad-cache", default=None,
                        help="Si se pasa, evalúa también con VAD aplicado (cache per-window).")
    parser.add_argument("--vad-threshold", type=float, default=-2.0)
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

    species_to_idx = species_mapping()
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    num_classes = len(species_to_idx)
    print(f"num_classes: {num_classes}")
    print(f"OTHERS ({len(OTHERS)}): {OTHERS}")

    others_orig_idx = {species_to_idx[s] for s in OTHERS if s in species_to_idx}
    print(f"OTHERS idx en mapping original: {sorted(others_orig_idx)}")

    # Mapping para "group": las clases reales mantienen su orden alfabético,
    # "Otros" toma el último índice
    real_species = sorted(s for s in species_to_idx if s not in OTHERS)
    grouped_remap = {s: i for i, s in enumerate(real_species)}
    others_grouped_idx = len(real_species)
    grouped_remap.update({s: others_grouped_idx for s in OTHERS})
    grouped_n_classes = len(real_species) + 1
    print(f"Grouped classes: {grouped_n_classes} ({len(real_species)} reales + 1 Otros)")

    # Mapping para "drop": las clases reales con índice 0..16
    drop_n_classes = len(real_species)

    # Modelo
    model = BirdClassifier.load_from_checkpoint(ckpt, strict=False, map_location="cpu")
    model.eval()
    if model.net[-1].out_features != num_classes:
        print(f"! num_classes mismatch", file=sys.stderr)
        return 3

    sources: list[tuple[str, pd.DataFrame]] = []
    sources.append(("raw", load_eval_embeddings_raw()))
    if args.vad_cache:
        cache_path = Path(args.vad_cache)
        if cache_path.exists():
            print(f"\nCargando cache VAD: {cache_path}  (threshold={args.vad_threshold})")
            sources.append((f"vad@{args.vad_threshold}", load_eval_embeddings_vad(cache_path, args.vad_threshold)))
        else:
            print(f"! cache VAD no existe: {cache_path}", file=sys.stderr)

    all_rows = []
    for source_name, df in sources:
        emb_arr = np.stack(df["embedding"].apply(np.asarray, args=(np.float32,)).to_numpy())
        with torch.no_grad():
            probs = F.softmax(model(torch.from_numpy(emb_arr)), dim=1).numpy()
        df = df.copy()
        df["pred_orig"] = probs.argmax(axis=1)
        df["true_orig"] = df["species"].map(species_to_idx).astype(int)

        # Mapeos derivados
        # grouped
        idx_orig_to_grouped = np.array([grouped_remap[idx_to_species[i]] for i in range(num_classes)])
        df["true_grouped"] = idx_orig_to_grouped[df["true_orig"].to_numpy()]
        df["pred_grouped"] = idx_orig_to_grouped[df["pred_orig"].to_numpy()]
        # drop: filtramos abajo

        for fold in ("test_clean", "test_hard"):
            f = df[df["fold"] == fold]
            if f.empty:
                continue
            # as-is
            m_asis = metrics_for(f["true_orig"].to_numpy(), f["pred_orig"].to_numpy(), num_classes)
            # group
            m_group = metrics_for(f["true_grouped"].to_numpy(), f["pred_grouped"].to_numpy(), grouped_n_classes)
            # drop: descartar samples cuya verdad ∈ OTHERS
            keep = ~f["true_orig"].isin(others_orig_idx)
            f_drop = f[keep]
            # En "drop" trabajamos solo con las 17 clases reales. Re-indexamos.
            real_to_idx = {s: i for i, s in enumerate(real_species)}
            true_drop = f_drop["species"].map(real_to_idx).to_numpy()
            # Pred: si predijo una rara (no en real_to_idx), eso es error en este modo
            # Mapeamos preds que no están en real_species a un índice "wrong" (-1)
            # El f1 con label list que NO incluye -1 trata esos como wrong.
            pred_drop = np.array([
                real_to_idx.get(idx_to_species[p], -1)
                for p in f_drop["pred_orig"].to_numpy()
            ])
            m_drop = metrics_for(true_drop, pred_drop, drop_n_classes)

            for mode, m in [("as-is", m_asis), ("group", m_group), ("drop", m_drop)]:
                all_rows.append({
                    "source": source_name,
                    "fold": fold,
                    "mode": mode,
                    **m,
                })

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== Resultado ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(summary.to_string(index=False))

    # Pivote para lectura rápida
    print("\n=== Lectura por modo (acc / macro_f1) ===")
    for source_name, _ in sources:
        sub = summary[summary["source"] == source_name]
        pivot = sub.pivot(index="mode", columns="fold", values=["acc", "macro_f1", "n"])
        print(f"\n[{source_name}]")
        print(pivot.to_string())

    print(f"\nGuardado en: {OUT_DIR / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
