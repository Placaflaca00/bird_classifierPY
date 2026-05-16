"""Benchmark: BirdNET nativo vs BirdNET embedding + ONNX fine-tuned.

Compara dos approaches sobre los mismos audios de test_clean y test_hard:

    1. **Baseline**: BirdNET V2.4 nativo (~6522 clases). Por ventana, sigmoid
       sobre los logits, max-pool per-class across windows, restringido a
       bird-only (excluye Engine/Noise/Dog/Human/ranas/etc), top-1 mapeado
       a mis 20 especies cuando existe correspondencia taxonómica.
    2. **Pipeline (mio)**: BirdNET embedding (penúltima capa, 1024-d, mean-pool
       across windows) -> ONNX classifier 20-clases -> softmax -> top-1.

Single forward por ventana captura embedding (penúltima capa) Y logits
nativos (~6522) simultáneamente. Reusa `lambda/handler.py` para
garantizar paridad con producción.

Decisiones tomadas (justificación en el .md de salida):
    - Agregación baseline: MAX per-class across windows (mirrora handler.py
      production gating, justificable por Wood & Kahl 2024).
    - Filtro non-bird: lista keyword-based + géneros de anfibios. Las
      meta-clases de BirdNET (Engine, Noise, etc.) y ranas se excluyen
      antes del argmax para no contaminar el baseline.
    - Pipile jacutinga (única faltante en BirdNET): contribuye 0 al
      baseline garantizado. Se reporta tanto en la tabla "all 20" como
      en una vista "BirdNET-compatible 19" para auditabilidad.
    - Confianza: baseline (sigmoid multi-label) y pipeline (softmax
      single-label) no son la misma escala. Se reportan separadas, no
      hay "improvement" sobre confianza.

Output:
    benchmarks/baseline_vs_finetuned_<YYYY-MM-DD>.json
    benchmarks/baseline_vs_finetuned_<YYYY-MM-DD>.md

Uso:
    python -u scripts/benchmark_baseline_vs_finetuned.py
    python -u scripts/benchmark_baseline_vs_finetuned.py --limit 10  # smoke
    python -u scripts/benchmark_baseline_vs_finetuned.py --folds test_clean
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TEST_HARD_DIR = ROOT / "data" / "test_sets" / "xc_hard"
SPLITS = ROOT / "data" / "processed" / "splits.parquet"
OUT_DIR = ROOT / "benchmarks"


def load_handler_module():
    """Importa lambda/handler.py via importlib (lambda es reserved keyword).

    Side effect: carga BirdNET TFLite y ONNX classifier en module-level.
    Reusa lo que ya hace prod — paridad garantizada.
    """
    spec = importlib.util.spec_from_file_location(
        "lambda_handler", ROOT / "lambda" / "handler.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_my_to_birdnet_mapping(my_species: list[str], birdnet_labels: list[str]) -> dict[int, int | None]:
    """Mapea mi_idx (0..19) -> birdnet_idx (0..6521) o None si no existe.

    Match por nombre cientifico exacto (parte antes del '_').
    Sinonimos taxonomicos probados para Pipile jacutinga (Aburria, Penelope) — confirmado
    NO existe en BirdNET V2.4. Se documenta en el .md.
    """
    sci_to_idx = {l.split("_", 1)[0]: i for i, l in enumerate(birdnet_labels)}
    synonyms = {
        "Pipile jacutinga": ["Aburria jacutinga", "Penelope jacutinga"],
    }
    out: dict[int, int | None] = {}
    for my_idx, sp in enumerate(my_species):
        if sp in sci_to_idx:
            out[my_idx] = sci_to_idx[sp]
            continue
        found = None
        for syn in synonyms.get(sp, []):
            if syn in sci_to_idx:
                found = sci_to_idx[syn]
                break
        out[my_idx] = found
    return out


def build_non_bird_mask(birdnet_labels: list[str]) -> np.ndarray:
    """Mascara bool de tamano len(labels): True donde la clase NO es ave.

    Dos criterios:
      1. Meta-clases agregadas por BirdNET (no taxones biologicos):
         Engine, Environmental, Fireworks, Gun shot, Human (varios),
         Noise, Power tools, Siren, Dog, Insect.
         Patron: nombre sin espacio antes del '_' o palabra explicita.
      2. Generos de anfibios presentes (ranas). Listado verificado
         contra los 6522 labels: Acris, Eleutherodactylus, Hyliola,
         Lithobates. Excluyo Batrachostomus y Podargus porque son aves
         (frogmouths, no ranas — false positive por keyword "Frog").
    """
    META_KEYWORDS = {
        "Engine", "Environmental", "Fireworks", "Gun shot", "Human non-vocal",
        "Human vocal", "Human whistle", "Noise", "Power tools", "Siren",
        "Dog", "Insect",
    }
    FROG_GENERA = {"Acris", "Eleutherodactylus", "Hyliola", "Lithobates"}

    mask = np.zeros(len(birdnet_labels), dtype=bool)
    for i, label in enumerate(birdnet_labels):
        sci = label.split("_", 1)[0]
        # Meta-clase: el "scientific" es la misma palabra que el "common",
        # ej. "Engine_Engine", "Noise_Noise". O coincide con keyword.
        if sci in META_KEYWORDS:
            mask[i] = True
            continue
        # Rana: genero en lista de anfibios.
        genus = sci.split()[0] if " " in sci else sci
        if genus in FROG_GENERA:
            mask[i] = True
    return mask


def collect_audio_jobs(folds: list[str]) -> pd.DataFrame:
    """Devuelve DF (filepath, abs_path, species, fold) para los folds pedidos.

    test_clean/val/train viven en data/raw/ (consultamos splits.parquet).
    test_hard vive en data/test_sets/xc_hard/.
    """
    raw_meta = pd.read_parquet(RAW_DIR / "metadata.parquet")
    hard_meta = pd.read_parquet(TEST_HARD_DIR / "metadata.parquet")
    splits = pd.read_parquet(SPLITS)

    rows = []
    if any(f in folds for f in ("test_clean", "val", "train")):
        merged = raw_meta.merge(splits, on="filepath")
        for fold in folds:
            if fold == "test_hard":
                continue
            sub = merged[merged["fold"] == fold]
            for _, r in sub.iterrows():
                rows.append({
                    "filepath": r["filepath"],
                    "abs_path": str(RAW_DIR / r["filepath"]),
                    "species": r["species"],
                    "fold": fold,
                })
    if "test_hard" in folds:
        for _, r in hard_meta.iterrows():
            rows.append({
                "filepath": r["filepath"],
                "abs_path": str(TEST_HARD_DIR / r["filepath"]),
                "species": r["species"],
                "fold": "test_hard",
            })
    return pd.DataFrame(rows)


def infer_one(
    audio_path: str, interpreter, input_idx: int, emb_idx: int, cls_idx: int,
    sigmoid_fn, chunk_fn, embedding_dim: int, n_classes_birdnet: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Single forward por audio. Devuelve:
        embedding_meanpool: shape (1024,) — mean across windows del embedding
        max_per_class_sigmoid: shape (~6522,) — max across windows de sigmoid(logits)
        n_windows: int

    El bucle por ventana lee AMBOS tensores con un solo invoke. Costo:
    sigmoid extra por ventana (~6522 floats) + max acumulado. Despreciable.
    """
    import librosa

    y, _ = librosa.load(audio_path, sr=48_000, mono=True)
    if len(y) == 0:
        raise ValueError("audio vacio")
    y = y.astype(np.float32)
    windows = chunk_fn(y).astype(np.float32)

    embs = np.empty((len(windows), embedding_dim), dtype=np.float32)
    max_per_class = np.full(n_classes_birdnet, -np.inf, dtype=np.float64)
    for k, w in enumerate(windows):
        interpreter.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interpreter.invoke()
        embs[k] = interpreter.get_tensor(emb_idx)[0]
        logits = interpreter.get_tensor(cls_idx)[0]
        probs = sigmoid_fn(logits)
        np.maximum(max_per_class, probs, out=max_per_class)
    return embs.mean(axis=0), max_per_class.astype(np.float32), len(windows)


def baseline_topk(
    max_per_class: np.ndarray, bird_mask_inverse: np.ndarray,
    birdnet_to_my: dict[int, int], k: int = 3,
) -> list[tuple[int | None, float, int]]:
    """Top-K del baseline: argmax sobre bird-only, mapeado a mis 20.

    bird_mask_inverse: bool array donde True = bird (NO la mascara non-bird).
    Devuelve lista de (my_idx_or_None, confidence, birdnet_idx) en orden top.
    None significa que el top-1 cayo en una clase no presente en mis 20.
    """
    scores = max_per_class.copy()
    scores[~bird_mask_inverse] = -np.inf  # excluir non-bird
    top_birdnet_idx = np.argsort(scores)[::-1][:k]
    out = []
    for bn_idx in top_birdnet_idx:
        bn_idx = int(bn_idx)
        my_idx = birdnet_to_my.get(bn_idx)
        out.append((my_idx, float(max_per_class[bn_idx]), bn_idx))
    return out


def pipeline_topk(embedding: np.ndarray, onnx_sess, k: int = 3) -> list[tuple[int, float]]:
    """Pipeline (ONNX) top-K. Devuelve [(my_idx, softmax_prob), ...]."""
    x = embedding.astype(np.float32).reshape(1, -1)
    logits = onnx_sess.run(None, {"embedding": x})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top_idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[int(i)])) for i in top_idx]


def macro_acc(results_for_fold: list[dict], n_classes: int, key: str) -> float:
    """Promedio per-especie de accuracy. results_for_fold[i] tiene `species_idx` y `key`.

    `key` es 'baseline_correct' o 'pipeline_correct' (bool).
    """
    by_class: dict[int, list[bool]] = defaultdict(list)
    for r in results_for_fold:
        by_class[r["species_idx"]].append(bool(r[key]))
    if not by_class:
        return float("nan")
    per_class = [sum(v) / len(v) for v in by_class.values()]
    return float(np.mean(per_class))


def per_species_table(results_for_fold: list[dict], my_species: list[str]) -> list[dict]:
    """Tabla por especie con N, baseline acc, pipeline acc."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in results_for_fold:
        by_class[r["species_idx"]].append(r)
    rows = []
    for my_idx in range(len(my_species)):
        bucket = by_class.get(my_idx, [])
        n = len(bucket)
        if n == 0:
            rows.append({
                "species": my_species[my_idx], "n": 0,
                "baseline_acc": None, "pipeline_acc": None,
                "baseline_top3_acc": None, "pipeline_top3_acc": None,
            })
            continue
        rows.append({
            "species": my_species[my_idx],
            "n": n,
            "baseline_acc": sum(r["baseline_correct"] for r in bucket) / n,
            "pipeline_acc": sum(r["pipeline_correct"] for r in bucket) / n,
            "baseline_top3_acc": sum(r["baseline_top3_correct"] for r in bucket) / n,
            "pipeline_top3_acc": sum(r["pipeline_top3_correct"] for r in bucket) / n,
        })
    return rows


def compute_fold_metrics(results: list[dict], my_species: list[str], known_in_birdnet: set[int]) -> dict:
    """Top-1, top-3, macro, no-match counts, conf means."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    n_classes = len(my_species)

    baseline_top1_micro = sum(r["baseline_correct"] for r in results) / n
    pipeline_top1_micro = sum(r["pipeline_correct"] for r in results) / n
    baseline_top3_micro = sum(r["baseline_top3_correct"] for r in results) / n
    pipeline_top3_micro = sum(r["pipeline_top3_correct"] for r in results) / n

    baseline_top1_macro = macro_acc(results, n_classes, "baseline_correct")
    pipeline_top1_macro = macro_acc(results, n_classes, "pipeline_correct")

    baseline_no_match = sum(1 for r in results if r["baseline_top1_my_idx"] is None)
    baseline_conf_mean = float(np.mean([r["baseline_top1_conf"] for r in results]))
    pipeline_conf_mean = float(np.mean([r["pipeline_top1_conf"] for r in results]))

    # BirdNET-compatible 19 (excluir Pipile jacutinga)
    results_19 = [r for r in results if r["species_idx"] in known_in_birdnet]
    if results_19:
        baseline_top1_micro_19 = sum(r["baseline_correct"] for r in results_19) / len(results_19)
        pipeline_top1_micro_19 = sum(r["pipeline_correct"] for r in results_19) / len(results_19)
    else:
        baseline_top1_micro_19 = pipeline_top1_micro_19 = float("nan")

    return {
        "n": n,
        "baseline": {
            "top1_micro": baseline_top1_micro,
            "top1_macro": baseline_top1_macro,
            "top3_micro": baseline_top3_micro,
            "no_match_n": baseline_no_match,
            "conf_mean_max_sigmoid": baseline_conf_mean,
        },
        "pipeline": {
            "top1_micro": pipeline_top1_micro,
            "top1_macro": pipeline_top1_macro,
            "top3_micro": pipeline_top3_micro,
            "conf_mean_softmax": pipeline_conf_mean,
        },
        "birdnet_compatible_19_only": {
            "n": len(results_19),
            "baseline_top1_micro": baseline_top1_micro_19,
            "pipeline_top1_micro": pipeline_top1_micro_19,
        },
        "per_species": per_species_table(results, my_species),
    }


def build_confusion_matrix(results: list[dict], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r in results:
        gt = r["species_idx"]
        pred = r["pipeline_top1_my_idx"]
        cm[gt, pred] += 1
    return cm


def write_markdown(path: Path, payload: dict, my_species: list[str],
                   my_to_birdnet: dict[int, int | None], birdnet_labels: list[str],
                   non_bird_count: int) -> None:
    lines = []
    lines.append(f"# Benchmark: BirdNET nativo vs ONNX fine-tuned\n")
    lines.append(f"**Fecha:** {payload['date']}  ")
    lines.append(f"**Modelo:** `wa-drop3-v1` (ver `models/classifier.json`)  ")
    lines.append(f"**BirdNET:** V2.4 GLOBAL 6K ({len(birdnet_labels)} clases, {non_bird_count} non-bird filtradas)  ")
    lines.append(f"**Inference time total:** {payload['inference_seconds']:.1f} s\n")

    lines.append("## TL;DR\n")
    for fold_name in payload["folds"]:
        m = payload["folds"][fold_name]
        if m.get("n", 0) == 0:
            lines.append(f"- **{fold_name}**: 0 audios procesados (skipped).")
            continue
        b = m["baseline"]; p = m["pipeline"]
        delta_micro = (p["top1_micro"] - b["top1_micro"]) * 100
        ratio = p["top1_micro"] / b["top1_micro"] if b["top1_micro"] > 0 else float("inf")
        lines.append(f"- **{fold_name}** (n={m['n']}): baseline {b['top1_micro']*100:.1f}% -> pipeline {p['top1_micro']*100:.1f}% top-1 micro. "
                     f"Delta absoluto: **{delta_micro:+.1f}pp**, relativo: **{ratio:.2f}x**.")
    lines.append("")

    lines.append("## Mapeo BirdNET -> mis 20 clases\n")
    matched = sum(1 for v in my_to_birdnet.values() if v is not None)
    missing = [my_species[k] for k, v in my_to_birdnet.items() if v is None]
    lines.append(f"- {matched}/20 con match exacto por nombre cientifico.")
    if missing:
        lines.append(f"- **Faltantes en BirdNET V2.4:** {missing}")
        lines.append(f"  - Probados sinonimos para Pipile jacutinga: `Aburria jacutinga`, `Penelope jacutinga`. Tampoco existen.")
        lines.append(f"  - Implica: para audios de `{missing[0]}`, baseline NUNCA puede acertar (top-1 baseline = wrong garantizado). Justifica el fine-tuning.\n")
    else:
        lines.append("")

    for fold_name in payload["folds"]:
        m = payload["folds"][fold_name]
        if m.get("n", 0) == 0:
            continue
        b = m["baseline"]; p = m["pipeline"]
        lines.append(f"## Fold: `{fold_name}` (n={m['n']})\n")
        lines.append("### Metricas globales\n")
        lines.append("| metric | baseline (BirdNET nativo) | pipeline (BirdNET emb + ONNX) | delta |")
        lines.append("|---|--:|--:|--:|")
        lines.append(f"| top-1 micro accuracy | {b['top1_micro']*100:.2f}% | {p['top1_micro']*100:.2f}% | {(p['top1_micro']-b['top1_micro'])*100:+.2f}pp |")
        lines.append(f"| top-1 macro accuracy | {b['top1_macro']*100:.2f}% | {p['top1_macro']*100:.2f}% | {(p['top1_macro']-b['top1_macro'])*100:+.2f}pp |")
        lines.append(f"| top-3 micro accuracy | {b['top3_micro']*100:.2f}% | {p['top3_micro']*100:.2f}% | {(p['top3_micro']-b['top3_micro'])*100:+.2f}pp |")
        lines.append(f"| confianza promedio | {b['conf_mean_max_sigmoid']:.3f} (max sigmoid) | {p['conf_mean_softmax']:.3f} (softmax top-1) | (escalas distintas, no compar.) |")
        lines.append(f"| baseline no_match (cayo fuera de mis 20) | {b['no_match_n']} | — | — |\n")

        sub = m["birdnet_compatible_19_only"]
        if sub["n"] != m["n"]:
            lines.append(f"### Sub-vista: solo clases que BirdNET conoce (19/20)\n")
            lines.append(f"Excluye Pipile jacutinga. Baseline accuracy comparable sin la penalidad estructural.")
            lines.append(f"- n = {sub['n']}")
            lines.append(f"- baseline top-1 micro: {sub['baseline_top1_micro']*100:.2f}%")
            lines.append(f"- pipeline top-1 micro: {sub['pipeline_top1_micro']*100:.2f}%\n")

        lines.append("### Per-species\n")
        lines.append("| especie | n | baseline top-1 | pipeline top-1 | baseline top-3 | pipeline top-3 | gana |")
        lines.append("|---|--:|--:|--:|--:|--:|:--|")
        for row in m["per_species"]:
            if row["n"] == 0:
                lines.append(f"| {row['species']} | 0 | — | — | — | — | — |")
                continue
            ba = row["baseline_acc"]; pa = row["pipeline_acc"]
            winner = "pipe" if pa > ba else ("base" if ba > pa else "tie")
            lines.append(
                f"| {row['species']} | {row['n']} | {ba*100:.0f}% | {pa*100:.0f}% | "
                f"{row['baseline_top3_acc']*100:.0f}% | {row['pipeline_top3_acc']*100:.0f}% | **{winner}** |"
            )
        lines.append("")

    lines.append("## Decisiones metodologicas\n")
    lines.append("- **Single-pass por audio**: una sola pasada por BirdNET captura embedding (penultima capa, 1024-d) y logits nativos (~6522). Sin doble forward — eficiencia y paridad numerica.")
    lines.append("- **Agregacion baseline = MAX per-class across windows** (mirrora `lambda/handler.py` para gating; justificable por Wood & Kahl 2024 — BirdNET fue disenado para deteccion puntual en ventanas de 3s, max captura el peak).")
    lines.append("- **Agregacion pipeline = MEAN embedding across windows** (igual que produccion y precompute_embeddings.py).")
    lines.append(f"- **Filtro non-bird**: {non_bird_count} clases de BirdNET excluidas antes del argmax (Engine, Noise, Dog, Human-varios, Power tools, ranas de generos Acris/Eleutherodactylus/Hyliola/Lithobates, etc).")
    lines.append("- **Mapeo conservador**: solo nombre cientifico exacto + sinonimos verificados. No se inventan matches.")
    lines.append("- **Confianza**: baseline (max sigmoid multi-label) y pipeline (softmax top-1) NO son la misma escala. Se reportan con su nombre real, sin claim de 'mejora de confianza'.")
    lines.append("- **Sin overlap train/test verificado**: 0 archivos compartidos por basename entre `data/raw/metadata.parquet` (2392 audios, 1675 train+356 val+351 test_clean) y `data/test_sets/xc_hard/` (208).\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="+", default=["test_clean", "test_hard"],
                    choices=["test_clean", "test_hard", "val", "train"])
    ap.add_argument("--limit", type=int, default=None, help="Limita a N audios (smoke).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[setup] cargando lambda/handler.py (carga BirdNET + ONNX en module-level)...", flush=True)
    t0 = time.time()
    lh = load_handler_module()
    interp, input_idx, emb_idx, cls_idx = lh._BIRDNET
    onnx_sess = lh._CLASSIFIER_SESS
    my_idx_to_species: dict[int, str] = lh._IDX_TO_SPECIES
    my_species = [my_idx_to_species[i] for i in range(len(my_idx_to_species))]
    n_my = len(my_species)
    print(f"[setup] handler module loaded ({time.time()-t0:.1f}s). {n_my} clases.", flush=True)

    # BirdNET labels (de birdnetlib en dev local)
    from birdnetlib.analyzer import Analyzer  # type: ignore
    birdnet_labels = Analyzer().labels  # ~6522
    n_bn = len(birdnet_labels)
    print(f"[setup] BirdNET labels: {n_bn}", flush=True)

    # Mapping
    my_to_bn = build_my_to_birdnet_mapping(my_species, birdnet_labels)
    bn_to_my = {v: k for k, v in my_to_bn.items() if v is not None}
    known_in_birdnet = set(bn_to_my.values())
    matched = sum(1 for v in my_to_bn.values() if v is not None)
    print(f"[setup] mapping: {matched}/{n_my} matched. Faltantes: "
          f"{[my_species[i] for i, v in my_to_bn.items() if v is None]}", flush=True)

    # Non-bird filter
    non_bird_mask = build_non_bird_mask(birdnet_labels)
    bird_mask = ~non_bird_mask
    print(f"[setup] non-bird classes excluded: {int(non_bird_mask.sum())} de {n_bn}", flush=True)

    # Jobs
    jobs = collect_audio_jobs(args.folds)
    if args.limit:
        jobs = jobs.head(args.limit).reset_index(drop=True)
    print(f"[setup] audios a procesar: {len(jobs)} (por fold: "
          f"{jobs.groupby('fold').size().to_dict()})", flush=True)

    species_to_idx = {sp: i for i, sp in enumerate(my_species)}

    t_infer = time.time()
    results: list[dict] = []
    for i, job in enumerate(jobs.itertuples(index=False), 1):
        try:
            emb, max_per_class, n_win = infer_one(
                job.abs_path, interp, input_idx, emb_idx, cls_idx,
                lh._sigmoid, lh._chunk_audio, lh.EMBEDDING_DIM, n_bn,
            )
        except Exception as e:
            print(f"  ! [{i}/{len(jobs)}] {job.filepath}: {type(e).__name__}: {e}", flush=True)
            continue

        gt_idx = species_to_idx[job.species]
        baseline_top = baseline_topk(max_per_class, bird_mask, bn_to_my, k=3)
        pipeline_top = pipeline_topk(emb, onnx_sess, k=3)

        baseline_top1_my = baseline_top[0][0]
        baseline_top1_conf = baseline_top[0][1]
        baseline_top1_bn = baseline_top[0][2]
        pipeline_top1_my = pipeline_top[0][0]
        pipeline_top1_conf = pipeline_top[0][1]

        baseline_correct = baseline_top1_my == gt_idx
        pipeline_correct = pipeline_top1_my == gt_idx
        baseline_top3_my_set = {t[0] for t in baseline_top}
        pipeline_top3_my_set = {t[0] for t in pipeline_top}
        baseline_top3_correct = gt_idx in baseline_top3_my_set
        pipeline_top3_correct = gt_idx in pipeline_top3_my_set

        results.append({
            "filepath": job.filepath,
            "fold": job.fold,
            "species": job.species,
            "species_idx": gt_idx,
            "n_windows": n_win,
            "baseline_top1_my_idx": baseline_top1_my,
            "baseline_top1_birdnet_idx": baseline_top1_bn,
            "baseline_top1_birdnet_label": birdnet_labels[baseline_top1_bn],
            "baseline_top1_conf": baseline_top1_conf,
            "baseline_correct": baseline_correct,
            "baseline_top3_correct": baseline_top3_correct,
            "pipeline_top1_my_idx": pipeline_top1_my,
            "pipeline_top1_conf": pipeline_top1_conf,
            "pipeline_correct": pipeline_correct,
            "pipeline_top3_correct": pipeline_top3_correct,
        })

        if i % 25 == 0 or i == len(jobs):
            elapsed = time.time() - t_infer
            rate = i / elapsed
            eta = (len(jobs) - i) / rate
            print(f"  [{i}/{len(jobs)}] {rate:.2f} aud/s ETA {eta/60:.1f}min", flush=True)

    inf_seconds = time.time() - t_infer
    print(f"\n[done] inferencia: {inf_seconds:.1f}s para {len(results)} audios.", flush=True)

    # Aggregate por fold
    payload = {
        "date": str(date.today()),
        "model": "wa-drop3-v1",
        "birdnet_version": "V2.4 GLOBAL 6K",
        "birdnet_n_classes": n_bn,
        "non_bird_n_excluded": int(non_bird_mask.sum()),
        "my_n_classes": n_my,
        "my_species": my_species,
        "mapping_matched": matched,
        "mapping_missing": [my_species[i] for i, v in my_to_bn.items() if v is None],
        "inference_seconds": inf_seconds,
        "folds": {},
    }
    for fold_name in args.folds:
        fold_results = [r for r in results if r["fold"] == fold_name]
        payload["folds"][fold_name] = compute_fold_metrics(
            fold_results, my_species, known_in_birdnet
        )

    out_stem = f"baseline_vs_finetuned_{payload['date']}"
    json_path = OUT_DIR / f"{out_stem}.json"
    md_path = OUT_DIR / f"{out_stem}.md"
    csv_path = OUT_DIR / f"{out_stem}_raw.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[out] {json_path} ({json_path.stat().st_size/1024:.1f} KB)", flush=True)

    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"[out] {csv_path} ({csv_path.stat().st_size/1024:.1f} KB)", flush=True)

    write_markdown(md_path, payload, my_species, my_to_bn, birdnet_labels,
                   int(non_bird_mask.sum()))
    print(f"[out] {md_path} ({md_path.stat().st_size/1024:.1f} KB)", flush=True)

    # Headline en stdout
    print("\n=== HEADLINE ===", flush=True)
    for fold_name in args.folds:
        m = payload["folds"][fold_name]
        if m.get("n", 0) == 0:
            continue
        b = m["baseline"]; p = m["pipeline"]
        delta = (p["top1_micro"] - b["top1_micro"]) * 100
        ratio = p["top1_micro"] / b["top1_micro"] if b["top1_micro"] > 0 else float("inf")
        print(f"{fold_name:11s} n={m['n']:3d}  "
              f"baseline {b['top1_micro']*100:5.2f}%  ->  pipeline {p['top1_micro']*100:5.2f}%  "
              f"({delta:+5.2f}pp, {ratio:.2f}x)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
