"""Precomputa embeddings BirdNET (V2.4, 1024-dim) sobre todos los audios.

Pipeline por archivo:
    1. Cargar audio mono a 48 kHz (librosa).
    2. (Opcional) Aplicar pipeline de audiomentations sobre el waveform completo
       para generar K copias aumentadas (sólo split=train).
    3. Trocear en ventanas de 3 s (144000 samples). La última se padea con ceros.
    4. Pasar cada ventana por BirdNET TFLite -> tensor index 545 (penúltima capa).
    5. Mean-pool sobre todas las ventanas -> 1 vector 1024-dim por (audio, aug_id).

Procesa en una sola pasada los splits 'train' (data/raw/) y 'test_hard'
(data/test_sets/xc_hard/), pero los marca con la columna `split` para que
el training NUNCA agarre test_hard. test_hard nunca se aumenta.

Salida: data/processed/embeddings.parquet con columnas:
    filepath, species, split, embedding, is_aug, aug_id

Cada audio aparece K+1 veces si está en train y --augment=K (aug_id=0 es el
original; 1..K son copias aumentadas). En test_hard aparece 1 vez (aug_id=0).

Uso:
    python scripts/precompute_embeddings.py                       # full, sin aug
    python scripts/precompute_embeddings.py --limit 20            # smoke test
    python scripts/precompute_embeddings.py --augment 2           # K=2 sobre train
    python scripts/precompute_embeddings.py --augment 2 --resume  # retomar parcial
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(ROOT))

from src.data.augmentations import (  # noqa: E402
    build_audio_pipeline_aggressive,
    build_audio_pipeline_conservative,
)

RAW_DIR = ROOT / "data" / "raw"
TEST_DIR = ROOT / "data" / "test_sets" / "xc_hard"
OUT_DIR = ROOT / "data" / "processed"
OUT_PARQUET = OUT_DIR / "embeddings.parquet"

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 144_000  # 3 sec @ 48 kHz
EMBEDDING_DIM = 1024

# Política tiered K por especie (post-cleanup label noise, 2026-05-09).
# Target: ~150-220 train rows post-aug por especie. Calculado contra N train
# de splits.parquet del dataset wa-drop3 limpio.
AUG_K_TIERED: dict[str, int] = {
    # N >= 150 -> K=0 (ya tienen suficiente)
    "Tringa flavipes": 0,
    "Tringa melanoleuca": 0,
    "Calidris canutus": 0,
    # N 80-149 -> K=1
    "Actitis macularius": 1,
    "Ara ararauna": 1,
    "Calidris minutilla": 1,
    "Ramphastos toco": 1,
    # N 50-79 -> K=2
    "Calidris pusilla": 2,
    "Chauna torquata": 2,
    "Ortalis canicollis": 2,
    "Calidris melanotos": 2,
    "Calidris bairdii": 2,
    "Pluvialis dominica": 2,
    # N < 50 -> K=3
    "Jabiru mycteria": 3,
    "Pluvialis squatarola": 3,
    "Pipile jacutinga": 3,
    "Columba livia": 3,
    "Calidris fuscicollis": 3,
    "Heliornis fulica": 3,
    "Rhea americana": 3,
}


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


def load_waveform(audio_path: Path) -> np.ndarray:
    """Carga mono 48 kHz, float32. Levanta ValueError si está vacío."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError(f"audio vacío: {audio_path}")
    return y.astype(np.float32)


def embed_waveform(y: np.ndarray, interpreter, input_idx, emb_idx) -> np.ndarray:
    """Mean-pool de embeddings BirdNET sobre las ventanas de un waveform."""
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

    raw["abs_path"] = raw["filepath"].apply(lambda p: str(RAW_DIR / p))
    test["abs_path"] = test["filepath"].apply(lambda p: str(TEST_DIR / p))

    cols = ["filepath", "species", "split", "abs_path"]
    return pd.concat([raw[cols], test[cols]], ignore_index=True)


def aug_seed(base_seed: int, filepath: str, aug_id: int) -> int:
    """Seed determinístico por (filepath, aug_id). Estable entre runs (md5)."""
    fp_hash = int.from_bytes(hashlib.md5(filepath.encode("utf-8")).digest()[:4], "big")
    return (base_seed * 1_000_003 + fp_hash * 17 + aug_id) & 0x7FFFFFFF


def load_done_keys(parquet_path: Path) -> set[tuple[str, int]]:
    """Lee parquet existente y devuelve set de (filepath, aug_id) ya hechos."""
    df = pd.read_parquet(parquet_path)
    if "aug_id" not in df.columns:
        df["aug_id"] = 0
    return set(zip(df["filepath"], df["aug_id"].astype(int)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo N audios (smoke test).")
    parser.add_argument("--resume", action="store_true",
                        help="Saltear (filepath, aug_id) ya presentes en el parquet.")
    parser.add_argument("--augment", type=int, default=0,
                        help="K copias aumentadas por audio del split=train (uniforme). Si --aug-policy=tiered, este valor se ignora salvo como fallback.")
    parser.add_argument("--aug-policy", default="uniform", choices=["uniform", "tiered"],
                        help="uniform: K igual para todos. tiered: K por especie según AUG_K_TIERED.")
    parser.add_argument("--aug-pipeline", default="conservative",
                        choices=["conservative", "aggressive"],
                        help="conservative: SNR+Shift+Gain (Fase 3). aggressive: + PitchShift + TimeStretch + LowPass (Fase 5 OOD-aware).")
    parser.add_argument("--augment-seed", type=int, default=42,
                        help="Base seed para aumentaciones (reproducibilidad).")
    args = parser.parse_args()

    if args.augment < 0:
        print("--augment debe ser >= 0", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs()
    if args.limit:
        jobs = jobs.head(args.limit).copy()

    def k_for(species: str) -> int:
        if args.aug_policy == "tiered":
            return AUG_K_TIERED.get(species, args.augment)
        return args.augment

    n_train = int((jobs["split"] == "train").sum())
    n_test = int((jobs["split"] == "test_hard").sum())
    train_jobs = jobs[jobs["split"] == "train"]
    rows_train_aug = sum(k_for(sp) + 1 for sp in train_jobs["species"]) if args.aug_policy == "tiered" \
                     else n_train * (args.augment + 1)
    rows_target = rows_train_aug + n_test

    print(f"Audios a procesar: {len(jobs)}")
    print(f"  train:     {n_train}  -> {rows_train_aug} embeddings (policy={args.aug_policy}, pipeline={args.aug_pipeline})")
    if args.aug_policy == "tiered":
        per_sp = train_jobs.groupby("species").size().rename("N").to_frame()
        per_sp["K"] = per_sp.index.map(lambda s: AUG_K_TIERED.get(s, args.augment))
        per_sp["post_aug"] = per_sp["N"] * (per_sp["K"] + 1)
        print("    K por especie (sorted by N descendente):")
        for sp, row in per_sp.sort_values("N", ascending=False).iterrows():
            print(f"      {sp:30s}  N={row['N']:>4}  K={row['K']}  post_aug={row['post_aug']:>4}")
    print(f"  test_hard: {n_test}")
    print(f"Embeddings totales esperados: {rows_target}")

    done_keys: set[tuple[str, int]] = set()
    if args.resume and OUT_PARQUET.exists():
        done_keys = load_done_keys(OUT_PARQUET)
        print(f"Resume: {len(done_keys)} (filepath, aug_id) ya en parquet.")

    needs_aug = (args.augment > 0) or (
        args.aug_policy == "tiered" and any(v > 0 for v in AUG_K_TIERED.values())
    )
    if needs_aug:
        aug_pipe = (build_audio_pipeline_aggressive() if args.aug_pipeline == "aggressive"
                    else build_audio_pipeline_conservative())
    else:
        aug_pipe = None

    print("\nCargando BirdNET TFLite...")
    interpreter, input_idx, emb_idx = build_interpreter()
    print("OK.\n")

    rows: list[dict] = []
    skipped_audios = 0
    failures = 0
    t0 = time.time()
    for i, job in enumerate(jobs.itertuples(index=False), 1):
        k = k_for(job.species) if job.split == "train" else 0
        needed_aug_ids = list(range(k + 1)) if job.split == "train" else [0]
        todo = [a for a in needed_aug_ids if (job.filepath, a) not in done_keys]
        if not todo:
            skipped_audios += 1
            continue

        try:
            y = load_waveform(Path(job.abs_path))
        except Exception as e:
            print(f"  ! [{i}/{len(jobs)}] {job.filepath} (load): {type(e).__name__}: {e}")
            failures += 1
            continue

        for aug_id in todo:
            try:
                if aug_id == 0:
                    emb = embed_waveform(y, interpreter, input_idx, emb_idx)
                else:
                    np.random.seed(aug_seed(args.augment_seed, job.filepath, aug_id))
                    y_aug = aug_pipe(samples=y.copy(), sample_rate=SAMPLE_RATE)
                    emb = embed_waveform(y_aug, interpreter, input_idx, emb_idx)
            except Exception as e:
                print(f"  ! [{i}/{len(jobs)}] {job.filepath} (aug_id={aug_id}): {type(e).__name__}: {e}")
                failures += 1
                continue

            rows.append({
                "filepath": job.filepath,
                "species": job.species,
                "split": job.split,
                "embedding": emb.tolist(),
                "is_aug": aug_id > 0,
                "aug_id": aug_id,
            })

        if i % 50 == 0 or i == len(jobs):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(jobs)}] {rate:.1f} aud/s, ETA {eta/60:.1f} min")

    print(f"\nResumen: {len(rows)} embeddings nuevos, {skipped_audios} audios skipped (resume), {failures} fallos.")
    if not rows and not (args.resume and OUT_PARQUET.exists()):
        print("! Ningún embedding extraído.")
        return 1

    df_new = pd.DataFrame(rows)

    if args.resume and OUT_PARQUET.exists():
        df_existing = pd.read_parquet(OUT_PARQUET)
        if "aug_id" not in df_existing.columns:
            df_existing["aug_id"] = 0
            df_existing["is_aug"] = False
        df = pd.concat([df_existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["filepath", "aug_id"], keep="last")
    else:
        df = df_new

    df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"\nEscrito {OUT_PARQUET} ({size_mb:.1f} MB, {len(df)} filas)")
    print(f"  train:        {(df['split']=='train').sum()}  ({df.loc[df['split']=='train', 'is_aug'].sum()} aumentados)")
    print(f"  test_hard:    {(df['split']=='test_hard').sum()}")
    print(f"  especies:     {df['species'].nunique()}")
    print(f"  audios únicos:{df['filepath'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
