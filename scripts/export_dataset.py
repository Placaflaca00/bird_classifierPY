"""Exporta el dataset combinado (original + annotator) para retrain de Fase 6.

Lee items ``review_status=approved`` de DynamoDB, descarga sus audios desde S3,
calcula embeddings BirdNET, y combina con el dataset original (los 2590 con
sus folds intactos del hold-out D20). Salida: un parquet self-contained que
``scripts/train.py`` (sub-fase 6.3, siguiente paso) consume sin merge.

DECISIONES (consensuadas 2026-05-25)
====================================
1. **Filter `is_seed_data` por default**. La tabla DDB es compartida prod+test
   (ver `scripts/generate_seed_data.py`). Sin filtro, seed contamina training.
   Flag `--include-seed` para testing del workflow E2E.

2. **Local SIEMPRE + W&B opcional**. El parquet output va siempre a
   `data/processed/embeddings_retrain_<ts>.parquet`. Si `--upload-wandb`,
   tambien sube como Artifact con metadata reproducible (git_sha, library
   versions, counts, BirdNET binary hash).

3. **filepath sintetico para items annotator**: `s3://<bucket>/<s3_key>`.
   Unico por construccion (UUID4 en s3_key), no colisiona con filepaths
   originales (formato `<species>/<file>.mp3` relativo).

4. **Re-generacion full cada run**, NO append incremental. Cada run regenera
   el parquet desde DDB+S3+originales. Razones: cero acumulacion de bugs
   historicos, sincronizado con DDB siempre, fail-loud si un s3_key esta
   roto, reproducible bajo reset humano.

5. **Embedding cache por SHA-256 del audio**. Los 2590 originales no se
   recalculan (ya tienen embedding en `embeddings.parquet`, lectura directa).
   Para items annotator, cache local indexado por SHA-256 del audio bytes
   evita recalcular en runs sucesivos (`.embedding_cache_v1.parquet`).
   En GH Actions el cache arranca vacio (disk efimero) — tolera, recalcula
   todo. El cache es optimizacion de debug local, no correctness.

6. **Fail-loud si s3_key falta**. Si un item DDB apunta a un s3_key que no
   existe en S3 (consent expirado y lifecycle borro, o bucket corrupto),
   abortar con mensaje claro citando `prediction_id`. Operador decide
   (corregir DDB, restaurar, o limpiar). NO skip silencioso.

7. **Audit trail en parquet**. Cada fila tiene:
     - source: "original" | "annotator"
     - ddb_pk: NULL para originales, "PRED#<uuid>" para nuevos
     - generated_at: ISO timestamp del run
     - git_sha: commit del export script
   Sirve para debugging y compliance.

Pipeline
========
1. Cargar originales: `data/processed/embeddings.parquet` (2590 filas).
   Merge con `data/processed/splits.parquet` para obtener `fold` autoritativo.
2. Cargar cache SHA-256 local (si existe).
3. Query DDB scan: `review_status=approved AND attribute_exists(reviewed_label)
   AND reviewed_label IN <20>` (filtra `is_seed_data != True` por default).
4. Para cada item:
     - Download S3 audio bytes (fail-loud si missing).
     - SHA-256 del audio.
     - Cache hit -> reusar embedding. Miss -> calcular + agregar al cache.
     - Construir fila con audit columns.
5. Concatenar originales + nuevos. Agregar `generated_at`, `git_sha`.
6. Persistir parquet local. Subir a W&B (si --upload-wandb).
7. Reporte: counts por fold/source/especie, hard vs easy, cache hit ratio.

Usage
=====
    python scripts/export_dataset.py                       # dry-run, sin DDB
    python scripts/export_dataset.py --apply               # solo annotator real
    python scripts/export_dataset.py --apply --include-seed # incluye seed (testing)
    python scripts/export_dataset.py --apply --upload-wandb # + sube a W&B
    python scripts/export_dataset.py --apply --no-cache    # bypass cache
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reusar funciones de precompute_embeddings.py garantiza paridad bit-a-bit
# en el calculo de embeddings con los 2590 originales.
from scripts.precompute_embeddings import (  # noqa: E402
    build_interpreter,
    chunk_audio,
)

EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"
OUT_DIR = ROOT / "data" / "processed"
CACHE_PATH = OUT_DIR / ".embedding_cache_v1.parquet"

DDB_TABLE_NAME = "bird-classifier-py-data"
S3_BUCKET = "conocetuave-py-uploads"
AWS_REGION = "us-east-1"

SAMPLE_RATE = 48000
EMBEDDING_DIM = 1024
VALID_FOLDS = {"train", "val", "test_clean", "test_hard"}

# Las 20 especies del clasificador wa-drop3-v1. Filtro de seguridad por si
# algun item DDB tiene reviewed_label fuera del scope.
VALID_SPECIES_20: list[str] = []  # lazy-loaded desde embeddings.parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exporta dataset combinado original+annotator para retrain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Ejecuta (default: dry-run, lee DDB pero no calcula embeddings "
        "ni escribe output).",
    )
    p.add_argument(
        "--include-seed",
        action="store_true",
        help="Incluye items con is_seed_data=True. Solo para testing E2E. "
        "En produccion (workflow GH Actions con data real), no usar.",
    )
    p.add_argument(
        "--embeddings-artifact",
        type=str,
        default="embeddings:wa-drop3",
        help="W&B artifact name:alias del embeddings de los originales. "
        "Default 'embeddings:wa-drop3' matchea wa-drop3-v1 en prod "
        "(politica D20 inmutabilidad).",
    )
    p.add_argument(
        "--splits-artifact",
        type=str,
        default="splits:wa-drop3",
        help="W&B artifact name:alias del splits.parquet. "
        "Default 'splits:wa-drop3' idem.",
    )
    p.add_argument(
        "--upload-wandb",
        action="store_true",
        help="Despues de escribir local, sube como W&B Artifact con metadata "
        "reproducible (git_sha, library versions, counts).",
    )
    p.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Alias custom del artifact W&B (default: retrain-<timestamp>).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass del cache SHA-256. Siempre recalcula embeddings de items "
        "annotator. Util para validacion de reproducibilidad.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita procesamiento a N items annotator (smoke test).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def git_sha() -> str:
    """Hash del commit actual. 'unknown' si no estamos en git o hay error."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def library_versions() -> dict[str, str]:
    """Snapshot de versiones de libs criticas para reproducibilidad del W&B
    Artifact metadata. Sin esto, "embeddings:retrain-<ts>" no es reproducible.
    """
    import librosa
    import scipy
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import tensorflow as tf
        versions["tensorflow"] = tf.__version__
    except ImportError:
        try:
            import tflite_runtime
            versions["tflite_runtime"] = tflite_runtime.__version__
        except (ImportError, AttributeError):
            versions["tflite"] = "unknown"
    return versions


def birdnet_binary_sha256() -> str:
    """SHA-256 del .tflite de BirdNET. Cambia si rotamos a otra version del
    modelo BirdNET (e.g. V2.4 -> V3.0), lo cual invalidaria embeddings.

    El path lo resuelve birdnetlib dinamicamente (mismo mecanismo que usa
    `precompute_embeddings.build_interpreter`).
    """
    try:
        from birdnetlib.analyzer import Analyzer
        model_path = Path(Analyzer().model_path)
    except Exception:
        return "<analyzer init failed>"
    if not model_path.exists():
        return "<model file missing>"
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Original dataset loading
# ---------------------------------------------------------------------------

def load_originals(
    embeddings_artifact: str, splits_artifact: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Descarga embeddings + splits desde W&B Artifacts (NO disco local).

    Por que W&B y no disco:
      - CI (GH Actions runner) arranca con disco limpio — no existen los
        parquets locales.
      - Lineage automatico: el output artifact del export queda vinculado
        a las versiones EXACTAS de inputs (politica D20).
      - Cache transparente: ``~/.cache/wandb/artifacts/<hash>/`` post-primer
        download. Runs subsequentes son instantaneos en dev local.

    Si querés evaluar/entrenar SIN W&B (modo offline), no es path soportado.
    El proyecto declara W&B Artifacts como source-of-truth del dataset
    versionado (D20).

    Args:
        embeddings_artifact: nombre:alias del artifact embeddings.
        splits_artifact: idem splits.

    Returns:
        (merged_df, resolved_versions) donde resolved_versions es dict
        {"embeddings": "embeddings:v2", "splits": "splits:v1"} con la
        version concreta a la que el alias resolvio. Usar para metadata
        del output artifact + audit.
    """
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not entity or not project:
        raise RuntimeError(
            "WANDB_ENTITY o WANDB_PROJECT faltan en env. Cargar .env o "
            "exportar como secret en CI."
        )

    api = wandb.Api()
    resolved: dict[str, str] = {}

    print(f"  Descargando {embeddings_artifact} desde W&B...")
    art_emb = api.artifact(f"{entity}/{project}/{embeddings_artifact}", type="dataset")
    resolved["embeddings"] = art_emb.name   # e.g. "embeddings:v2"
    emb_dir = Path(art_emb.download())
    emb_file = emb_dir / "embeddings.parquet"
    if not emb_file.exists():
        raise RuntimeError(
            f"embeddings.parquet no esta en {emb_dir}. "
            f"Files: {[f.name for f in emb_dir.iterdir()]}"
        )

    print(f"  Descargando {splits_artifact} desde W&B...")
    art_splits = api.artifact(f"{entity}/{project}/{splits_artifact}", type="dataset")
    resolved["splits"] = art_splits.name
    splits_dir = Path(art_splits.download())
    splits_file = splits_dir / "splits.parquet"
    if not splits_file.exists():
        raise RuntimeError(
            f"splits.parquet no esta en {splits_dir}. "
            f"Files: {[f.name for f in splits_dir.iterdir()]}"
        )

    print(f"  Resolved: {resolved}")
    emb = pd.read_parquet(emb_file)
    splits = pd.read_parquet(splits_file)
    merged = emb.merge(splits, on="filepath", how="inner")

    # Sanidad: post-merge no deberian aparecer folds invalidos.
    invalid = set(merged["fold"].unique()) - VALID_FOLDS
    if invalid:
        raise RuntimeError(f"splits.parquet tiene folds invalidos: {invalid}")

    merged["source"] = "original"
    merged["ddb_pk"] = pd.NA

    # Especies validas a partir de los originales (fuente de verdad de
    # las 20 clases del clasificador).
    global VALID_SPECIES_20
    VALID_SPECIES_20 = sorted(merged["species"].unique())
    if len(VALID_SPECIES_20) != 20:
        print(f"! WARNING: embeddings tiene {len(VALID_SPECIES_20)} especies, "
              "no 20. Verificar artifact version.", file=sys.stderr)

    return merged, resolved


# ---------------------------------------------------------------------------
# DDB query
# ---------------------------------------------------------------------------

def query_approved_items(table, include_seed: bool) -> list[dict[str, Any]]:
    """Scan paginado de items review_status=approved con reviewed_label
    valido. Filtra is_seed_data por default.
    """
    from boto3.dynamodb.conditions import Attr

    base_filter = (
        Attr("review_status").eq("approved")
        & Attr("reviewed_label").exists()
        & Attr("reviewed_label").is_in(VALID_SPECIES_20)
    )
    if not include_seed:
        # Tolera items que NO tienen el atributo (items reales de prod):
        # is_seed_data ausente == False conceptualmente.
        base_filter = base_filter & (
            Attr("is_seed_data").ne(True) | Attr("is_seed_data").not_exists()
        )

    items: list[dict[str, Any]] = []
    last_key = None
    while True:
        kwargs: dict[str, Any] = {
            "FilterExpression": base_filter,
            "ProjectionExpression": (
                "prediction_id, pk, s3_key, reviewed_label, top1_species, "
                "top1_confidence, is_seed_data"
            ),
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

    return items


# ---------------------------------------------------------------------------
# Embedding cache (SHA-256 -> 1024-d vector)
# ---------------------------------------------------------------------------

def load_cache() -> dict[str, np.ndarray]:
    """Lee cache local. Si no existe, dict vacio."""
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_parquet(CACHE_PATH)
    return {row["sha256"]: row["embedding"] for _, row in df.iterrows()}


def save_cache(cache: dict[str, np.ndarray]) -> None:
    """Persiste el cache a parquet local."""
    if not cache:
        return
    rows = [{"sha256": k, "embedding": v} for k, v in cache.items()]
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH, index=False)


# ---------------------------------------------------------------------------
# S3 download + embedding compute
# ---------------------------------------------------------------------------

def download_audio_bytes(s3, s3_key: str, prediction_id: str) -> bytes:
    """Descarga audio de S3. Fail-loud con mensaje claro si missing."""
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return resp["Body"].read()
    except s3.exceptions.NoSuchKey:
        raise RuntimeError(
            f"S3 NoSuchKey: prediction_id={prediction_id}, s3_key={s3_key}. "
            f"El audio fue borrado por lifecycle o nunca se subio. "
            f"Opciones: corregir item DDB (cambiar s3_key o borrar item), "
            f"restaurar desde backup, o limpiar manualmente y re-correr."
        )
    except Exception as e:
        raise RuntimeError(
            f"S3 error inesperado en prediction_id={prediction_id}, "
            f"s3_key={s3_key}: {type(e).__name__}: {e}"
        ) from e


def compute_embedding_from_bytes(
    audio_bytes: bytes, interpreter, input_idx, emb_idx
) -> np.ndarray:
    """Carga audio desde bytes (sin escribir a disk) y calcula embedding
    1024-dim mean-pooled (mismo pipeline que precompute_embeddings).
    """
    import librosa
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError("audio decodificado tiene 0 samples")
    y = y.astype(np.float32)

    windows = chunk_audio(y).astype(np.float32)
    embs = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    for k, w in enumerate(windows):
        interpreter.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interpreter.invoke()
        embs[k] = interpreter.get_tensor(emb_idx)[0]
    return embs.mean(axis=0)


def process_annotator_items(
    items: list[dict[str, Any]],
    s3,
    cache: dict[str, np.ndarray],
    use_cache: bool,
) -> tuple[pd.DataFrame, int, int]:
    """Procesa items DDB. Returns (df_annotator, cache_hits, cache_misses)."""
    interpreter, input_idx, emb_idx = build_interpreter()

    rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    t0 = time.time()
    for i, item in enumerate(items, 1):
        prediction_id = item.get("prediction_id", "<no-prediction_id>")
        s3_key = item.get("s3_key")
        if not s3_key:
            raise RuntimeError(
                f"item DDB sin s3_key: prediction_id={prediction_id}. "
                f"Items approved sin audio no deberian existir. Investigar."
            )

        audio_bytes = download_audio_bytes(s3, s3_key, prediction_id)
        sha = hashlib.sha256(audio_bytes).hexdigest()

        if use_cache and sha in cache:
            embedding = cache[sha]
            cache_hits += 1
            tag = "HIT "
        else:
            embedding = compute_embedding_from_bytes(
                audio_bytes, interpreter, input_idx, emb_idx
            )
            cache[sha] = embedding
            cache_misses += 1
            tag = "MISS"

        rows.append({
            "filepath": f"s3://{S3_BUCKET}/{s3_key}",
            "species": item["reviewed_label"],
            "split": "train",
            "fold": "train",
            "embedding": embedding,
            "is_aug": False,
            "aug_id": 0,
            "source": "annotator",
            "ddb_pk": item["pk"],
            "_top1_species": item.get("top1_species"),  # tmp para report
        })

        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(items) - i) if i > 0 else 0
        print(f"  [{i:3d}/{len(items)}] {tag}  {item['reviewed_label']:<30s} "
              f"sha={sha[:10]}...  ETA {eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    return df, cache_hits, cache_misses


# ---------------------------------------------------------------------------
# Final assembly + audit columns
# ---------------------------------------------------------------------------

def add_audit_columns(df: pd.DataFrame, sha: str) -> pd.DataFrame:
    """Agrega generated_at + git_sha. Mutates en place + returns."""
    df["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    df["git_sha"] = sha
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_final(df: pd.DataFrame, cache_hits: int, cache_misses: int) -> None:
    print()
    print("=" * 70)
    print("RESUMEN DEL DATASET COMBINADO")
    print("=" * 70)

    print(f"Total filas: {len(df)}")
    print()

    print("Por fold:")
    fold_counts = df["fold"].value_counts().to_dict()
    for fold in ("train", "val", "test_clean", "test_hard"):
        print(f"  {fold:<12s} {fold_counts.get(fold, 0)}")
    print()

    print("Por source (en fold=train):")
    train_only = df[df["fold"] == "train"]
    src_counts = train_only["source"].value_counts().to_dict()
    print(f"  original    {src_counts.get('original', 0)}")
    print(f"  annotator   {src_counts.get('annotator', 0)}")
    print()

    annot = df[df["source"] == "annotator"]
    if len(annot) > 0:
        print(f"Items annotator: {len(annot)}")
        print("  Distribucion por especie:")
        for sp, count in annot["species"].value_counts().sort_index().items():
            print(f"    {sp:<30s} {count}")

        if "_top1_species" in annot.columns:
            hard = annot[annot["_top1_species"] != annot["species"]]
            easy = len(annot) - len(hard)
            print(f"  Easy (top1 == reviewed):     {easy}")
            print(f"  Hard (top1 != reviewed):     {len(hard)}")

    print()
    total_processed = cache_hits + cache_misses
    if total_processed > 0:
        hit_rate = cache_hits / total_processed * 100
        print(f"Cache embedding: hits={cache_hits}, misses={cache_misses} "
              f"({hit_rate:.0f}% hit rate)")


# ---------------------------------------------------------------------------
# W&B upload
# ---------------------------------------------------------------------------

def upload_to_wandb(
    parquet_path: Path,
    metadata: dict[str, Any],
    alias: str,
    input_artifacts: dict[str, str],
) -> None:
    """Sube el parquet como W&B Artifact con metadata reproducible.

    input_artifacts: dict {"embeddings": "embeddings:v2", "splits": "splits:v1"}
    de versiones resueltas que load_originals() descargo. Se declaran como
    use_artifact() para lineage automatico en W&B UI.
    """
    import wandb
    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="export-dataset",
        name=f"export-{alias}",
        notes="Dataset combinado (original + annotator) para retrain Fase 6.",
    )
    # Lineage explicito: el output queda vinculado a las versiones EXACTAS
    # que el export consumio (no aliases — la resolucion de alias->version se
    # fijo en load_originals).
    for input_name in input_artifacts.values():
        run.use_artifact(input_name)

    artifact = wandb.Artifact(
        name="embeddings_retrain",
        type="dataset",
        description=(
            "Dataset combinado original + annotator items con review_status="
            "approved. Reproducible via git_sha + library versions en "
            "metadata. Politica D20 respetada (hold-out test_clean+test_hard "
            "inmutable, items nuevos solo a fold=train)."
        ),
        metadata=metadata,
    )
    artifact.add_file(str(parquet_path))
    run.log_artifact(artifact, aliases=[alias])
    run.finish()
    print(f"  W&B Artifact subido: embeddings_retrain:{alias}")


# ---------------------------------------------------------------------------
# AWS clients
# ---------------------------------------------------------------------------

def _boto3_clients():
    import boto3
    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError(
            "boto3 no encontro credenciales AWS. Configurar via "
            "`aws configure` o cargar .env con AWS_ACCESS_KEY_ID."
        )
    ddb = session.resource("dynamodb", region_name=AWS_REGION)
    s3 = session.client("s3", region_name=AWS_REGION)
    table = ddb.Table(DDB_TABLE_NAME)
    return table, s3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    print("=== Export dataset (Fase 6.3) ===", flush=True)
    sha = git_sha()
    print(f"git_sha:           {sha}", flush=True)
    print(f"include_seed:      {args.include_seed}", flush=True)
    print(f"use_cache:         {not args.no_cache}", flush=True)
    print(f"limit:             {args.limit or 'no limit'}", flush=True)
    print(f"embeddings input:  {args.embeddings_artifact}", flush=True)
    print(f"splits input:      {args.splits_artifact}", flush=True)
    print(flush=True)

    # 1. Cargar originales (2590 con folds intactos) desde W&B Artifacts
    print("Cargando originales desde W&B Artifacts...", flush=True)
    df_orig, input_artifacts = load_originals(
        args.embeddings_artifact, args.splits_artifact,
    )
    print(f"  filas originales: {len(df_orig)}")
    print(f"  especies (sera el set valido): {len(VALID_SPECIES_20)}")
    print()

    # 2. Query DDB
    table, s3 = _boto3_clients()
    print(f"Scan DDB ({'with' if args.include_seed else 'WITHOUT'} seed)...",
          flush=True)
    items = query_approved_items(table, args.include_seed)
    print(f"  items approved encontrados: {len(items)}")
    if args.limit and len(items) > args.limit:
        items = items[:args.limit]
        print(f"  truncado a --limit={args.limit}")
    print()

    # --- DRY-RUN: no procesa embeddings ni escribe output ---
    if not args.apply:
        print("DRY-RUN: no calculo embeddings ni escribo output.", flush=True)
        print("Para ejecutar real: agregar --apply", flush=True)
        print()
        # Construir preview liviano para report (sin embeddings)
        if items:
            preview_annot = pd.DataFrame({
                "fold": "train",
                "source": "annotator",
                "species": [i["reviewed_label"] for i in items],
                "_top1_species": [i.get("top1_species") for i in items],
            })
            df_preview = pd.concat([
                df_orig[["fold", "source", "species"]].assign(_top1_species=None),
                preview_annot,
            ], ignore_index=True)
        else:
            df_preview = df_orig[["fold", "source", "species"]].assign(_top1_species=None)
        report_final(df_preview, 0, 0)
        return 0

    # --- APPLY: procesar embeddings + escribir parquet ---
    hits = 0
    misses = 0
    if items:
        cache = {} if args.no_cache else load_cache()
        print(f"Cache loaded: {len(cache)} entries", flush=True)
        print()
        print(f"Procesando {len(items)} items annotator...", flush=True)
        df_annot, hits, misses = process_annotator_items(
            items, s3, cache, use_cache=not args.no_cache
        )
        if not args.no_cache:
            save_cache(cache)
            print(f"\nCache persistido: {CACHE_PATH} ({len(cache)} entries)")

        df_combined = pd.concat([df_orig, df_annot], ignore_index=True, sort=False)
    else:
        print("! No hay items approved, output sera solo dataset original.")
        df_combined = df_orig.copy()

    df_final = add_audit_columns(df_combined, sha)

    # Reporte (incluyendo _top1_species si esta) y luego drop
    report_final(df_final, hits, misses)
    if "_top1_species" in df_final.columns:
        df_final = df_final.drop(columns=["_top1_species"])

    # Normalizar dtype de embedding a float32. Los originales en
    # embeddings.parquet estan en float64; los nuevos vienen de BirdNET en
    # float32. pyarrow no permite mezclar dtypes en column-of-arrays. float32
    # es el dtype nativo de BirdNET y lo que torch.from_numpy(...).float()
    # convierte downstream — sin perdida util de precision.
    df_final["embedding"] = df_final["embedding"].apply(
        lambda x: x.astype(np.float32) if x.dtype != np.float32 else x
    )

    # Persistir
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"embeddings_retrain_{ts}.parquet"
    df_final.to_parquet(out_path, index=False)
    print(f"\nEscrito: {out_path}")
    print(f"  size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 5. W&B upload opcional
    if args.upload_wandb:
        alias = args.alias or f"retrain-{ts}"
        metadata = {
            "git_sha": sha,
            "library_versions": library_versions(),
            "birdnet_sha256": birdnet_binary_sha256(),
            "n_total": int(len(df_final)),
            "n_original": int((df_final["source"] == "original").sum()),
            "n_annotator": int((df_final["source"] == "annotator").sum()),
            "include_seed": bool(args.include_seed),
            "cache_hits": int(locals().get("hits", 0)),
            "cache_misses": int(locals().get("misses", 0)),
            "fold_counts": {
                k: int(v) for k, v in df_final["fold"].value_counts().items()
            },
            "species_in_train": {
                k: int(v)
                for k, v in df_final[df_final["fold"] == "train"]["species"]
                .value_counts().items()
            },
        }
        print(f"\nSubiendo a W&B como artifact 'embeddings_retrain:{alias}'...",
              flush=True)
        # Pasar input_artifacts para que el run upload declare use_artifact()
        # sobre los inputs concretos (lineage automatico).
        metadata["input_artifacts"] = input_artifacts
        upload_to_wandb(out_path, metadata, alias, input_artifacts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
