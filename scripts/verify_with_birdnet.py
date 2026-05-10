"""Verifica samples sospechosos con BirdNET como segundo opinador.

BirdNET (clasificador completo de 6000 especies, entrenado por Cornell Lab
con data independiente) actúa como oracle externo. Si BirdNET coincide con
nuestro modelo contra la etiqueta original, hay fuerte evidencia de label
noise (dos modelos independientes dicen lo mismo).

Verdict:
    DROP    : BirdNET dice lo mismo que nuestro modelo (ambos contra label).
    KEEP    : BirdNET coincide con el label original (nuestro modelo se equivoca
              por similitud acústica). NO es label noise.
    UNCLEAR : Las 3 opiniones difieren, o BirdNET no detecta nada de la lista.

Lee `reports/label_issues/{test_clean,test_hard}_suspicious.csv` (genera con
scripts/find_label_issues.py primero). Procesa los top-N por severity.

Salida: `reports/label_issues/{fold}_verdict.csv` con columnas extra:
    birdnet_top1_species, birdnet_top1_score, birdnet_in_our_top3, verdict

Uso:
    python scripts/verify_with_birdnet.py
    python scripts/verify_with_birdnet.py --top-n 50
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.precompute_embeddings import (  # noqa: E402
    SAMPLE_RATE,
    chunk_audio,
    load_waveform,
)

RAW_DIR = ROOT / "data" / "raw"
TEST_HARD_DIR = ROOT / "data" / "test_sets" / "xc_hard"
ISSUES_DIR = ROOT / "reports" / "label_issues"


def build_interpreter_with_classifier_and_labels():
    """BirdNET interpreter + lista de labels (6000 species)."""
    import tensorflow as tf
    from birdnetlib.analyzer import Analyzer

    a = Analyzer()
    interpreter = tf.lite.Interpreter(
        model_path=a.model_path,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]["index"]
    classifier_out_idx = interpreter.get_output_details()[0]["index"]
    return interpreter, input_idx, classifier_out_idx, a.labels


def parse_sci_name(label_str: str) -> str:
    """'Tringa flavipes_Lesser Yellowlegs' -> 'Tringa flavipes'"""
    return label_str.split("_", 1)[0].strip()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_file(audio_path: Path, interpreter, input_idx, cls_idx) -> np.ndarray:
    """Forward BirdNET sobre todas las ventanas, devuelve probas mean-pool."""
    y = load_waveform(audio_path)
    windows = chunk_audio(y).astype(np.float32)
    all_scores: list[np.ndarray] = []
    for w in windows:
        interpreter.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interpreter.invoke()
        out = interpreter.get_tensor(cls_idx)[0]
        all_scores.append(np.asarray(out))
    arr = np.stack(all_scores)  # (n_windows, num_classes)
    # BirdNET es multi-label con sigmoid; los outputs en general son logits
    return sigmoid(arr).mean(axis=0)


def resolve_path(filepath: str, fold: str) -> Path:
    base = TEST_HARD_DIR if fold == "test_hard" else RAW_DIR
    return base / filepath


def verdict(label_orig: str, our_pred: str, birdnet_top1: str) -> str:
    if birdnet_top1 == label_orig:
        return "KEEP"  # BirdNET avala label original
    if birdnet_top1 == our_pred:
        return "DROP"  # BirdNET avala que NO es la especie etiquetada
    return "UNCLEAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=25,
                        help="Procesar top-N por severity de cada fold.")
    args = parser.parse_args()

    csvs = {
        "test_clean": ISSUES_DIR / "test_clean_suspicious.csv",
        "test_hard": ISSUES_DIR / "test_hard_suspicious.csv",
    }
    for f, p in csvs.items():
        if not p.exists():
            print(f"! Falta {p}. Ejecutá scripts/find_label_issues.py primero.", file=sys.stderr)
            return 1

    print("Cargando BirdNET...")
    interpreter, input_idx, cls_idx, labels = build_interpreter_with_classifier_and_labels()
    print(f"BirdNET cargado. {len(labels)} labels.")
    label_to_sci = [parse_sci_name(l) for l in labels]

    all_results = []
    for fold, csv_path in csvs.items():
        df = pd.read_csv(csv_path).head(args.top_n)
        print(f"\n=== {fold}: procesando {len(df)} archivos sospechosos ===")
        rows = []
        for i, row in enumerate(df.itertuples(index=False), 1):
            audio_path = resolve_path(row.filepath, fold)
            try:
                probs = score_file(audio_path, interpreter, input_idx, cls_idx)
            except Exception as e:
                print(f"  [{i}] {row.filepath}: FAIL {type(e).__name__}: {e}")
                continue
            top3_idx = np.argsort(-probs)[:3]
            top3 = [(label_to_sci[i], float(probs[i])) for i in top3_idx]
            birdnet_top1_sp, birdnet_top1_score = top3[0]
            v = verdict(row.species, row.pred_species, birdnet_top1_sp)
            rows.append({
                "filepath": row.filepath,
                "label_original": row.species,
                "our_pred": row.pred_species,
                "our_pred_conf": row.max_prob,
                "birdnet_top1": birdnet_top1_sp,
                "birdnet_top1_score": birdnet_top1_score,
                "birdnet_top2": top3[1][0],
                "birdnet_top2_score": top3[1][1],
                "birdnet_top3": top3[2][0],
                "birdnet_top3_score": top3[2][1],
                "verdict": v,
                "severity": row.severity,
            })
            print(f"  [{i}/{len(df)}] {row.filepath:55s}  label={row.species:25s}  our={row.pred_species:25s}  birdnet={birdnet_top1_sp:25s} ({birdnet_top1_score:.3f})  -> {v}")

        out_df = pd.DataFrame(rows)
        out_csv = ISSUES_DIR / f"{fold}_verdict.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\n  Guardado: {out_csv}")
        verdict_counts = out_df["verdict"].value_counts().to_dict()
        print(f"  Verdicts: {verdict_counts}")
        all_results.append((fold, out_df))

    # Resumen
    print("\n=== RESUMEN ===")
    for fold, df in all_results:
        n_drop = int((df["verdict"] == "DROP").sum())
        n_keep = int((df["verdict"] == "KEEP").sum())
        n_unclear = int((df["verdict"] == "UNCLEAR").sum())
        print(f"  {fold}: DROP={n_drop}  KEEP={n_keep}  UNCLEAR={n_unclear}  total={len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
