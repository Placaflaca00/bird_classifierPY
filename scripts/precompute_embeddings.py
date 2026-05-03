"""Precomputa embeddings BirdNET (V2.4, 1024-dim) sobre todos los audios.

Pipeline por archivo:
    1. Cargar audio mono a 48 kHz (librosa).
    2. Trocear en ventanas de 3 s (144000 samples). La última se padea con ceros.
    3. Pasar cada ventana por BirdNET TFLite -> tensor index 545 (penúltima capa).
    4. Mean-pool sobre todas las ventanas -> 1 vector 1024-dim por audio.

Procesa en una sola pasada los splits 'train' (data/raw/) y 'test_hard'
(data/test_sets/xc_hard/), pero los marca con la columna `split` para que
el training NUNCA agarre test_hard.

Salida: data/processed/embeddings.parquet con columnas:
    filepath, species, split, embedding (lista de 1024 floats)

Uso:
    python scripts/precompute_embeddings.py             # full
    python scripts/precompute_embeddings.py --limit 20  # smoke test (20 audios)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TEST_DIR = ROOT / "data" / "test_sets" / "xc_hard"
OUT_DIR = ROOT / "data" / "processed"
OUT_PARQUET = OUT_DIR / "embeddings.parquet"

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 144_000  # 3 sec @ 48 kHz
EMBEDDING_DIM = 1024


def build_interpreter():
    """Crea el TFLite interpreter de BirdNET con preserve_all_tensors=True."""
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
    embedding_idx = classifier_out_idx - 1  # penúltima capa
    return interpreter, input_idx, embedding_idx


def chunk_audio(y: np.ndarray) -> np.ndarray:
    """Divide y en ventanas de WINDOW_SAMPLES. Última se padea con ceros."""
    n = len(y)
    if n < WINDOW_SAMPLES:
        padded = np.pad(y, (0, WINDOW_SAMPLES - n))
        return padded[np.newaxis, :]
    n_windows = (n + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    total = n_windows * WINDOW_SAMPLES
    if total > n:
        y = np.pad(y, (0, total - n))
    return y[: n_windows * WINDOW_SAMPLES].reshape(n_windows, WINDOW_SAMPLES)


def embed_audio(audio_path: Path, interpreter, input_idx, emb_idx) -> np.ndarray:
    """Devuelve un vector 1024-dim mean-pooleado sobre las ventanas."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError(f"audio vacío: {audio_path}")
    windows = chunk_audio(y).astype(np.float32)

    embs = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    for k, w in enumerate(windows):
        interpreter.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interpreter.invoke()
        embs[k] = interpreter.get_tensor(emb_idx)[0]
    return embs.mean(axis=0)


def collect_jobs() -> pd.DataFrame:
    """Une metadata de raw + test_hard, agrega columna 'split'."""
    raw = pd.read_parquet(RAW_DIR / "metadata.parquet").assign(split="train")
    test = pd.read_parquet(TEST_DIR / "metadata.parquet").assign(split="test_hard")

    # filepath en raw es relativo a data/raw/, en test_hard relativo a data/test_sets/xc_hard/
    raw["abs_path"] = raw["filepath"].apply(lambda p: str(RAW_DIR / p))
    test["abs_path"] = test["filepath"].apply(lambda p: str(TEST_DIR / p))

    cols = ["filepath", "species", "split", "abs_path"]
    return pd.concat([raw[cols], test[cols]], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo N audios (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltear filepaths que ya están en embeddings.parquet")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs()
    if args.limit:
        jobs = jobs.head(args.limit).copy()
    print(f"Audios a procesar: {len(jobs)}")
    print(f"  train:     {(jobs['split']=='train').sum()}")
    print(f"  test_hard: {(jobs['split']=='test_hard').sum()}")

    if args.resume and OUT_PARQUET.exists():
        done = set(pd.read_parquet(OUT_PARQUET)["filepath"])
        before = len(jobs)
        jobs = jobs[~jobs["filepath"].isin(done)].copy()
        print(f"Resume: ya hechos {before - len(jobs)}, faltan {len(jobs)}")

    if jobs.empty:
        print("Nada que procesar.")
        return 0

    print("\nCargando BirdNET TFLite...")
    interpreter, input_idx, emb_idx = build_interpreter()
    print("OK.\n")

    rows = []
    t0 = time.time()
    for i, job in enumerate(jobs.itertuples(index=False), 1):
        try:
            emb = embed_audio(Path(job.abs_path), interpreter, input_idx, emb_idx)
        except Exception as e:
            print(f"  ! [{i}/{len(jobs)}] {job.filepath}: {type(e).__name__}: {e}")
            continue
        rows.append({
            "filepath": job.filepath,
            "species": job.species,
            "split": job.split,
            "embedding": emb.tolist(),
        })
        if i % 50 == 0 or i == len(jobs):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(jobs)}] {rate:.1f} aud/s, ETA {eta/60:.1f} min")

    if not rows:
        print("! Ningún embedding extraído.")
        return 1

    df_new = pd.DataFrame(rows)

    if args.resume and OUT_PARQUET.exists():
        df_existing = pd.read_parquet(OUT_PARQUET)
        df = pd.concat([df_existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["filepath"], keep="last")
    else:
        df = df_new

    df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"\nEscrito {OUT_PARQUET} ({size_mb:.1f} MB, {len(df)} filas)")
    print(f"  train:     {(df['split']=='train').sum()}")
    print(f"  test_hard: {(df['split']=='test_hard').sum()}")
    print(f"  especies:  {df['species'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
