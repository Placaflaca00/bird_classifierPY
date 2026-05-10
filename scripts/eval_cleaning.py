"""Evalúa estrategias de cleaning de audio sobre test_clean + test_hard.

Re-computa embeddings BirdNET desde el audio raw aplicando:

    - raw:       sin cleaning (baseline)
    - bandpass:  filtro pasa-banda 1-8 kHz (scipy butterworth, sosfiltfilt)
    - vad@thr:   descarta ventanas donde max(BirdNET classifier) < thr
    - both@thr:  bandpass + vad

El cache es **per-window** (1 fila por (filepath, source, window_idx)) — esto
permite barrer múltiples thresholds VAD post-hoc sin re-pasar BirdNET. El
forward de BirdNET es la operación cara (~1-2 s/audio en CPU).

Smoke test:
    python scripts/eval_cleaning.py --limit 10

Full + sweep:
    python scripts/eval_cleaning.py --thresholds 0.005 0.01 0.03 0.05 0.10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from glob import glob
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.precompute_embeddings import (  # noqa: E402
    EMBEDDING_DIM,
    SAMPLE_RATE,
    chunk_audio,
    load_waveform,
)
from src.models.classifier import BirdClassifier  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
TEST_HARD_DIR = ROOT / "data" / "test_sets" / "xc_hard"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"
EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
DEFAULT_CKPT_GLOB = "checkpoints/baseline-v0/best-*.ckpt"
OUT_DIR = ROOT / "reports" / "cleaning_eval"
DEFAULT_CACHE = OUT_DIR / "cleaning_per_window.parquet"

SOURCES = ("raw", "band")  # 2 forwards por audio


def build_interpreter_with_classifier():
    """Variante de precompute_embeddings.build_interpreter() que devuelve
    también el índice del clasificador (no sólo el del embedding)."""
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
    embedding_idx = classifier_out_idx - 1
    return interpreter, input_idx, embedding_idx, classifier_out_idx


def make_bandpass_sos(low_hz: float = 1000.0, high_hz: float = 8000.0,
                      sr: int = SAMPLE_RATE, order: int = 4):
    nyq = sr / 2
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def apply_bandpass(y: np.ndarray, sos) -> np.ndarray:
    return sosfiltfilt(sos, y).astype(np.float32)


def forward_windows(y: np.ndarray, interpreter, input_idx, emb_idx, cls_idx):
    """Por ventana devuelve embedding (1024-d) y max score del clasificador BirdNET."""
    windows = chunk_audio(y).astype(np.float32)
    embs = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    scores = np.empty(len(windows), dtype=np.float32)
    for k, w in enumerate(windows):
        interpreter.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interpreter.invoke()
        embs[k] = interpreter.get_tensor(emb_idx)[0]
        cls_out = interpreter.get_tensor(cls_idx)[0]
        scores[k] = float(np.asarray(cls_out).max())
    return embs, scores


def resolve_path(filepath: str, fold: str) -> Path:
    base = TEST_HARD_DIR if fold == "test_hard" else RAW_DIR
    return base / filepath


def collect_jobs(limit: int | None = None) -> pd.DataFrame:
    splits = pd.read_parquet(SPLITS_PATH)
    splits = splits[splits["fold"].isin(["test_clean", "test_hard"])].copy()
    emb_meta = pd.read_parquet(EMB_PATH, columns=["filepath", "species"]).drop_duplicates("filepath")
    df = splits.merge(emb_meta, on="filepath", how="left")
    if df["species"].isna().any():
        missing = df[df["species"].isna()]["filepath"].tolist()
        raise RuntimeError(f"Sin species en embeddings.parquet para: {missing[:3]}...")
    df["abs_path"] = [str(resolve_path(fp, fd)) for fp, fd in zip(df["filepath"], df["fold"])]
    df = df.reset_index(drop=True)
    if limit is not None:
        df = df.groupby("fold").head(limit // 2).reset_index(drop=True)
    return df


def species_mapping_from(emb_path: Path) -> dict[str, int]:
    df = pd.read_parquet(emb_path, columns=["species"])
    species = sorted(df["species"].unique())
    return {sp: i for i, sp in enumerate(species)}


def compute_per_window_cache(jobs: pd.DataFrame, cache_path: Path,
                             ignore_cache: bool) -> pd.DataFrame:
    """Pasa cada audio por BirdNET una vez (raw) y otra (bandpass), y guarda
    embedding + score por ventana. Soporta resume: salta filepaths ya cacheados."""
    if not ignore_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        # done = filepaths que tienen filas para AMBOS sources
        done_files = set(
            cached.groupby("filepath")["source"].agg(lambda x: set(x))
            .apply(lambda s: SOURCES[0] in s and SOURCES[1] in s)
            .pipe(lambda s: s[s].index)
        )
        print(f"Cache: {len(cached)} filas, {len(done_files)} archivos completos.")
    else:
        cached = pd.DataFrame(columns=[
            "filepath", "fold", "species", "source", "window_idx", "score", "embedding"
        ])
        done_files = set()

    todo = [j for j in jobs.itertuples(index=False) if j.filepath not in done_files]
    if not todo:
        print("Nada que computar (todo está en cache).")
        return cached

    print(f"\nCargando BirdNET TFLite ({len(todo)} archivos a procesar)...")
    interpreter, input_idx, emb_idx, cls_idx = build_interpreter_with_classifier()
    sos = make_bandpass_sos()
    print("OK.\n")

    new_rows: list[dict] = []
    failures = 0
    t0 = time.time()
    for i, job in enumerate(todo, 1):
        try:
            y = load_waveform(Path(job.abs_path))
        except Exception as e:
            print(f"  ! [{i}/{len(todo)}] {job.filepath} (load): {type(e).__name__}: {e}")
            failures += 1
            continue
        try:
            emb_raw, scores_raw = forward_windows(y, interpreter, input_idx, emb_idx, cls_idx)
            y_band = apply_bandpass(y, sos)
            emb_band, scores_band = forward_windows(y_band, interpreter, input_idx, emb_idx, cls_idx)
        except Exception as e:
            print(f"  ! [{i}/{len(todo)}] {job.filepath} (forward): {type(e).__name__}: {e}")
            failures += 1
            continue

        for source, embs, scores in [("raw", emb_raw, scores_raw), ("band", emb_band, scores_band)]:
            for wi in range(len(embs)):
                new_rows.append({
                    "filepath": job.filepath,
                    "fold": job.fold,
                    "species": job.species,
                    "source": source,
                    "window_idx": wi,
                    "score": float(scores[wi]),
                    "embedding": embs[wi].astype(np.float32).tolist(),
                })

        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(todo)}] {rate:.2f} aud/s, ETA {eta/60:.1f} min")

    print(f"\nNuevas filas (per-window): {len(new_rows)}, fallos: {failures}")
    df_new = pd.DataFrame(new_rows)
    out = pd.concat([cached, df_new], ignore_index=True) if len(new_rows) else cached
    out = out.drop_duplicates(subset=["filepath", "source", "window_idx"], keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"Cache total: {len(out)} filas en {cache_path}")
    return out


def aggregate_strategy(cache: pd.DataFrame, strategy: str, threshold: float) -> pd.DataFrame:
    """Agrega per-window -> per-file aplicando la estrategia indicada.

    raw      : source=raw, todas las ventanas
    bandpass : source=band, todas las ventanas
    vad      : source=raw, ventanas con score >= threshold (fallback: todas)
    both     : source=band, ventanas con score >= threshold (fallback: todas)
    """
    if strategy in ("raw", "vad"):
        sub = cache[cache["source"] == "raw"]
    else:
        sub = cache[cache["source"] == "band"]

    rows: list[dict] = []
    for (filepath, fold, species), grp in sub.groupby(["filepath", "fold", "species"], sort=False):
        embs = np.stack(grp["embedding"].apply(np.asarray, args=(np.float32,)).to_numpy())
        scores = grp["score"].to_numpy()
        n_total = len(embs)
        if strategy in ("vad", "both"):
            mask = scores >= threshold
            if mask.any():
                emb_pooled = embs[mask].mean(axis=0)
                n_active = int(mask.sum())
            else:
                emb_pooled = embs.mean(axis=0)
                n_active = 0  # fallback, marcamos como 0 activas
        else:
            emb_pooled = embs.mean(axis=0)
            n_active = n_total
        rows.append({
            "filepath": filepath,
            "fold": fold,
            "species": species,
            "embedding": emb_pooled,
            "n_active": n_active,
            "n_total": n_total,
        })
    return pd.DataFrame(rows)


def evaluate(model: BirdClassifier, agg: pd.DataFrame,
             species_to_idx: dict[str, int]) -> pd.DataFrame:
    """Forward del modelo sobre embeddings agregados. Devuelve agg con metrics."""
    if agg.empty:
        return agg
    emb_arr = np.stack(agg["embedding"].to_numpy()).astype(np.float32)
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(emb_arr)), dim=1).numpy()
    out = agg.copy()
    out["pred_idx"] = probs.argmax(axis=1)
    out["max_prob"] = probs.max(axis=1)
    out["true_idx"] = out["species"].map(species_to_idx).astype(int)
    out["correct"] = out["true_idx"] == out["pred_idx"]
    out["true_class_prob"] = probs[np.arange(len(out)), out["true_idx"].to_numpy()]
    return out


def summarize(eval_df: pd.DataFrame, strategy: str, threshold: float) -> list[dict]:
    rows = []
    for fold in ("test_clean", "test_hard"):
        f = eval_df[eval_df["fold"] == fold]
        if f.empty:
            continue
        rows.append({
            "strategy": strategy,
            "threshold": threshold if strategy in ("vad", "both") else np.nan,
            "fold": fold,
            "n": len(f),
            "conf_mean": float(f["max_prob"].mean()),
            "conf_median": float(f["max_prob"].median()),
            "true_prob_mean": float(f["true_class_prob"].mean()),
            "acc": float(f["correct"].mean()),
            "n_active_mean": float(f["n_active"].mean()),
            "n_total_mean": float(f["n_total"].mean()),
            "frac_fallback": float((f["n_active"] == 0).mean()) if strategy in ("vad", "both") else 0.0,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo N audios totales (smoke test).")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
                        help="Thresholds VAD a barrer. Los scores de BirdNET son logits "
                             "(no probas), rango aprox [-5, +5]. Threshold=0 separa "
                             "'BirdNET tiende a detectar pájaro' vs 'no'.")
    parser.add_argument("--ckpt", default=None,
                        help=f"Checkpoint. Default: glob {DEFAULT_CKPT_GLOB}")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="Cache per-window.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignorar cache y recomputar todo.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    if args.ckpt is None:
        cands = sorted(glob(str(ROOT / DEFAULT_CKPT_GLOB)))
        if not cands:
            print(f"No checkpoint en {DEFAULT_CKPT_GLOB}", file=sys.stderr)
            return 2
        ckpt_path = cands[-1]
    else:
        ckpt_path = args.ckpt
    print(f"Checkpoint: {ckpt_path}")

    jobs = collect_jobs(limit=args.limit)
    print(f"Audios target: {len(jobs)}  (clean={int((jobs['fold']=='test_clean').sum())}, "
          f"hard={int((jobs['fold']=='test_hard').sum())})")

    cache = compute_per_window_cache(jobs, cache_path, ignore_cache=args.no_cache)
    # Restringir cache a los archivos del job (--limit)
    job_files = set(jobs["filepath"])
    cache_for_eval = cache[cache["filepath"].isin(job_files)].copy()

    species_to_idx = species_mapping_from(EMB_PATH)
    num_classes = len(species_to_idx)
    model = BirdClassifier.load_from_checkpoint(ckpt_path, strict=False, map_location="cpu")
    model.eval()
    if model.net[-1].out_features != num_classes:
        print(f"! num_classes mismatch: ckpt={model.net[-1].out_features} mapping={num_classes}",
              file=sys.stderr)
        return 3

    # ---------- Distribución de scores (informativo) ----------
    print("\n=== Distribución de scores (max BirdNET classifier por ventana) ===")
    for src in SOURCES:
        for fold in ("test_clean", "test_hard"):
            s = cache_for_eval[(cache_for_eval["source"] == src) & (cache_for_eval["fold"] == fold)]["score"]
            if s.empty:
                continue
            qs = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(4).to_dict()
            print(f"  source={src:5s}  fold={fold:10s}  n_windows={len(s):4d}  "
                  f"min={s.min():.4f}  q05={qs[0.05]}  q25={qs[0.25]}  "
                  f"med={qs[0.5]}  q75={qs[0.75]}  q95={qs[0.95]}  max={s.max():.4f}")

    # ---------- Sweep ----------
    print("\n=== Sweep ===")
    all_summaries: list[dict] = []
    # raw y bandpass: 1 vez (no dependen de threshold)
    for strat in ("raw", "bandpass"):
        agg = aggregate_strategy(cache_for_eval, strat, threshold=0.0)
        ev = evaluate(model, agg, species_to_idx)
        all_summaries.extend(summarize(ev, strat, 0.0))
    # vad y both: por threshold
    for thr in args.thresholds:
        for strat in ("vad", "both"):
            agg = aggregate_strategy(cache_for_eval, strat, threshold=thr)
            ev = evaluate(model, agg, species_to_idx)
            all_summaries.extend(summarize(ev, strat, thr))

    summary = pd.DataFrame(all_summaries)
    summary = summary.sort_values(["strategy", "threshold", "fold"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n=== Resumen completo ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(summary.round(4).to_string(index=False))

    # ---------- Delta vs raw ----------
    raw_clean = summary[(summary["strategy"] == "raw") & (summary["fold"] == "test_clean")].iloc[0]
    raw_hard = summary[(summary["strategy"] == "raw") & (summary["fold"] == "test_hard")].iloc[0]
    print(f"\nBaseline raw: clean conf={raw_clean['conf_mean']:.4f} acc={raw_clean['acc']:.4f}  "
          f"hard conf={raw_hard['conf_mean']:.4f} acc={raw_hard['acc']:.4f}  "
          f"GAP_conf={raw_clean['conf_mean']-raw_hard['conf_mean']:.4f}")

    delta_rows = []
    for _, r in summary.iterrows():
        base = raw_clean if r["fold"] == "test_clean" else raw_hard
        delta_rows.append({
            "strategy": r["strategy"],
            "threshold": r["threshold"],
            "fold": r["fold"],
            "d_conf": r["conf_mean"] - base["conf_mean"],
            "d_acc": r["acc"] - base["acc"],
            "frac_fallback": r["frac_fallback"],
            "n_active_mean": r["n_active_mean"],
            "n_total_mean": r["n_total_mean"],
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "delta_vs_raw.csv", index=False)
    print("\n=== Delta vs raw (mismo fold) ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(delta_df.round(4).to_string(index=False))

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, fold in zip(axes, ("test_clean", "test_hard")):
        for strat, marker in [("vad", "o"), ("both", "s")]:
            sub = summary[(summary["strategy"] == strat) & (summary["fold"] == fold)].sort_values("threshold")
            ax.plot(sub["threshold"], sub["conf_mean"], marker=marker, label=f"{strat} conf")
            ax.plot(sub["threshold"], sub["acc"], marker=marker, linestyle="--", label=f"{strat} acc", alpha=0.6)
        # Líneas horizontales raw / bandpass
        for strat, color in [("raw", "k"), ("bandpass", "tab:gray")]:
            r = summary[(summary["strategy"] == strat) & (summary["fold"] == fold)]
            if not r.empty:
                ax.axhline(r.iloc[0]["conf_mean"], color=color, linestyle=":", label=f"{strat} conf", alpha=0.7)
                ax.axhline(r.iloc[0]["acc"], color=color, linestyle="-.", label=f"{strat} acc", alpha=0.4)
        ax.set_title(fold)
        ax.set_xlabel("vad threshold")
        ax.set_ylabel("metric")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Cleaning sweep — {len(jobs)} archivos")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep.png", dpi=120)

    print(f"\nReportes en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
