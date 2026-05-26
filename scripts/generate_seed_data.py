"""Genera items review_status=approved fake en DynamoDB para Fase 6 testing.

Crea N items PREDICTION sinteticos con audios reales del dataset principal
(fold=train solamente, jamas del hold-out) subidos a S3, marcados con
``is_seed_data=True``. Sirven como input contra el que debuggear
``scripts/export_dataset.py`` (sub-fase 6.3) y el workflow de retraining
(sub-fase 6.4) antes de que el annotator real (Fase 5) acumule volumen
suficiente.

WARNING — tabla y bucket COMPARTIDOS prod+test
==============================================
``bird-classifier-py-data`` (DDB) y ``conocetuave-py-uploads`` (S3) sirven a
produccion. Con seed data presente, TODA query del proyecto debe filtrar
``is_seed_data != True`` (o ``attribute_not_exists(is_seed_data)``) por
default, sino el dashboard de Fase 5 contara seed como trafico real y el
export de Fase 6 entrenara con audios sinteticos.

Componentes que deben actualizarse al filtrar ``is_seed_data``:
    - ``bird-annotator-py/dashboard/*.py`` (todas las queries que arman
      metricas en el tab admin).
    - ``scripts/export_dataset.py`` (Fase 6.3, todavia no escrito; al
      escribir, filtrar).
    - Cualquier futuro scanner / exportador que lea PREDICTION items.

Modelo conceptual del seed
==========================
El seed simula items **ya revisados por el ornitologo** (review_status=
approved). NO simula items pending — esos los genera el handler real
cuando un usuario hace /predict. El ground truth para retraining es
``review_status=approved AND reviewed_label IS NOT NULL``, NO
``feedback_status`` (que es del usuario, no del experto).

Para realismo visual del dashboard, ``feedback_status`` se setea
``confirmed`` para easy examples y ``corrected`` para hard examples, pero
``export_dataset.py`` no lo filtra. Es decoracion, no senal.

Reproducibilidad
================
Todas las decisiones random (sampling de audios, easy vs hard, valores de
confidence) usan ``random.Random(--random-seed)`` con default 42. Misma
seed -> mismos items. Cambiar la seed para generar conjuntos
independientes (si por algun motivo queres dos seeds en paralelo, pasa
distintos ``--seed-run-id``).

Idempotencia
============
``--apply`` aborta si ya hay items con ``is_seed_data=True`` en la tabla,
sugiriendo correr ``--cleanup`` primero. Flag opcional ``--append``
bypasea con warning (util si queres acumular varios runs).

Usage
=====
    # Inspeccion sin tocar AWS:
    python scripts/generate_seed_data.py --count 5

    # Smoke test (escribe 5 items reales):
    python scripts/generate_seed_data.py --count 5 --apply

    # Cleanup (borra todos los items con is_seed_data=True + objetos S3):
    python scripts/generate_seed_data.py --cleanup

    # Full set para Fase 6.3:
    python scripts/generate_seed_data.py --count 100 --apply

    # Ajustar mix de hard examples:
    python scripts/generate_seed_data.py --count 100 --hard-rate 0.4 --apply
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
METADATA_PATH = RAW_DIR / "metadata.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"

# Mismos nombres que produccion (lambda/handler.py:76,87). Compartidos prod+test.
DDB_TABLE_NAME = "bird-classifier-py-data"
S3_BUCKET = "conocetuave-py-uploads"
S3_PREFIX = "uploads/seed/"
AWS_REGION = "us-east-1"

# Coherente con prod (lambda/handler.py model_version constant).
MODEL_VERSION = "wa-drop3-v1"

# TTL identico a items reales (30 dias). Aplica aunque seed-data no expire
# semanticamente; consistencia de schema > exactitud.
PREDICTION_TTL_DAYS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera items review_status=approved fake en DDB para Fase 6.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g_action = p.add_mutually_exclusive_group()
    g_action.add_argument(
        "--apply",
        action="store_true",
        help="Ejecuta (default: dry-run, solo reporta).",
    )
    g_action.add_argument(
        "--cleanup",
        action="store_true",
        help="Modo destructivo: borra items con is_seed_data=True + objetos S3 "
        "con prefix uploads/seed/.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=100,
        help="Cantidad de items a generar (default: 100). Ignorado con --cleanup.",
    )
    p.add_argument(
        "--hard-rate",
        type=float,
        default=0.3,
        help="Fraccion de hard examples (top1_species != reviewed_label). "
        "Default 0.3. Range valido: 0.0-1.0.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para todas las decisiones random (sampling, easy/hard, "
        "confidence values). Default 42 para reproducibilidad.",
    )
    p.add_argument(
        "--seed-run-id",
        type=str,
        default=None,
        help="Override del run_id (default: uuid4 generado). Util para agrupar "
        "items o forzar re-run con mismo ID.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Permite --apply aunque ya haya items seed en la tabla. Solo si "
        "sabes lo que haces.",
    )
    args = p.parse_args()
    if not 0.0 <= args.hard_rate <= 1.0:
        p.error(f"--hard-rate fuera de range: {args.hard_rate}")
    if args.count < 1 and not args.cleanup:
        p.error(f"--count debe ser >= 1, got {args.count}")
    return args


# ---------------------------------------------------------------------------
# Audio selection
# ---------------------------------------------------------------------------

def load_eligible_audios() -> pd.DataFrame:
    """Carga audios elegibles para seed: fold=train solamente.

    JAMAS de test_clean ni test_hard (politica D20: el hold-out no se
    contamina con datos de testing del retrain loop). Si por alguna razon
    el merge metadata+splits no encuentra fold para una fila, se descarta
    silenciosamente (no falla).
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Falta {METADATA_PATH}. Corre primero: "
            "python scripts/build_metadata.py --apply"
        )
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {SPLITS_PATH}. Corre primero: "
            "python scripts/build_splits.py --apply"
        )
    meta = pd.read_parquet(METADATA_PATH)
    splits = pd.read_parquet(SPLITS_PATH)
    merged = meta.merge(splits, on="filepath", how="inner")
    eligible = merged[merged["fold"] == "train"].copy()
    return eligible


def sample_audios(
    df: pd.DataFrame, count: int, rng: random.Random
) -> pd.DataFrame:
    """Stratified sampling: ~count/N_species por especie.

    Si una especie tiene menos audios que el cupo proporcional, toma lo
    que hay. Si sobran cupos despues del round (por especies con N
    chico), rellena con sampling random del pool restante para alcanzar
    exactamente ``count``.
    """
    species_list = sorted(df["species"].unique())
    n_species = len(species_list)
    target_per_species = max(1, count // n_species)

    picked: list[pd.DataFrame] = []
    for sp in species_list:
        pool = df[df["species"] == sp]
        n_to_take = min(target_per_species, len(pool))
        # Sampling deterministico via .sample con seed reproducible
        indices = sorted(pool.index.tolist())
        rng.shuffle(indices)
        chosen = pool.loc[indices[:n_to_take]]
        picked.append(chosen)

    result = pd.concat(picked, ignore_index=True)

    # Si quedaron menos que count (especies con pocos audios), rellenar
    # del pool restante sin reemplazo.
    if len(result) < count:
        already_paths = set(result["filepath"])
        remaining = df[~df["filepath"].isin(already_paths)]
        if len(remaining) > 0:
            n_extra = min(count - len(result), len(remaining))
            indices = sorted(remaining.index.tolist())
            rng.shuffle(indices)
            extra = remaining.loc[indices[:n_extra]]
            result = pd.concat([result, extra], ignore_index=True)

    # Si quedaron mas que count (por target_per_species redondeado),
    # truncar deterministicamente.
    if len(result) > count:
        all_indices = sorted(result.index.tolist())
        rng.shuffle(all_indices)
        result = result.loc[all_indices[:count]].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Fake prediction generation
# ---------------------------------------------------------------------------

def generate_fake_prediction(
    true_species: str,
    all_species: list[str],
    is_hard: bool,
    rng: random.Random,
) -> dict[str, Any]:
    """Genera top1/top3 fake coherentes para un audio.

    Easy: top1 = true_species, confidence 0.75-0.95.
    Hard: top1 = otra especie random, confidence 0.45-0.75. reviewed_label
          queda como true_species (el "ornitologo corrigio").

    Returns dict con keys: top1_species, top1_confidence, top3_predictions,
    max_birdnet_confidence, n_windows.
    """
    if is_hard:
        others = [s for s in all_species if s != true_species]
        top1 = rng.choice(others)
        top1_conf = rng.uniform(0.45, 0.75)
    else:
        top1 = true_species
        top1_conf = rng.uniform(0.75, 0.95)

    # top3: top1 + 2 random distintas con confidence decreciente
    pool = [s for s in all_species if s != top1]
    rng.shuffle(pool)
    top2_sp, top3_sp = pool[0], pool[1]
    top2_conf = top1_conf * rng.uniform(0.4, 0.7)
    top3_conf = top2_conf * rng.uniform(0.3, 0.6)

    top3 = [
        {"species": top1, "common_name": None, "confidence": round(top1_conf, 4)},
        {"species": top2_sp, "common_name": None, "confidence": round(top2_conf, 4)},
        {"species": top3_sp, "common_name": None, "confidence": round(top3_conf, 4)},
    ]

    # max_birdnet_confidence: el gate de la Capa 2 (BirdNET) tipico esta
    # en ~0.1-0.5 para audios reales. Lo mantengo en ese rango.
    max_birdnet = rng.uniform(0.10, 0.50)

    # n_windows: BirdNET ventanea audio en chunks de 3s. Audios cortos
    # (<3s) -> 1 ventana. Tipicos del dataset (10-30s) -> 3-10 ventanas.
    n_windows = rng.randint(2, 8)

    return {
        "top1_species": top1,
        "top1_confidence": top1_conf,
        "top3_predictions": top3,
        "max_birdnet_confidence": max_birdnet,
        "n_windows": n_windows,
    }


def build_ddb_item(
    *,
    prediction_id: str,
    audio_path: Path,
    audio_duration_s: float,
    audio_size_bytes: int,
    true_species: str,
    pred: dict[str, Any],
    is_hard: bool,
    seed_run_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Construye el item DDB completo para un seed PREDICTION."""
    now = datetime.now(timezone.utc)
    timestamp_iso = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    ttl = int(now.timestamp()) + PREDICTION_TTL_DAYS * 24 * 3600

    # Fingerprint con prefijo distintivo "seed_" para que sea facil
    # filtrarlos en queries por fingerprint si alguien las construye.
    fingerprint = f"seed_{rng.getrandbits(32):08x}"

    item: dict[str, Any] = {
        "pk": f"PRED#{prediction_id}",
        "sk": "META",
        "item_type": "PREDICTION",
        "prediction_id": prediction_id,
        "timestamp": timestamp_iso,
        "fingerprint": fingerprint,
        "result_status": "detected",
        "inference_time_ms": Decimal(str(round(rng.uniform(100.0, 200.0), 2))),
        "audio_duration_s": Decimal(str(round(audio_duration_s, 2))),
        "audio_size_bytes": int(audio_size_bytes),
        "model_version": MODEL_VERSION,
        "training_consent": True,
        # feedback_status: decoracion para que el dashboard se vea coherente;
        # export_dataset.py NO filtra por este campo.
        "feedback_status": "corrected" if is_hard else "confirmed",
        # review_status + reviewed_label: lo que importa para Fase 6.
        "review_status": "approved",
        "reviewed_label": true_species,
        "reviewed_by": "seed_generator",
        "reviewed_at": timestamp_iso,
        # Seed markers (CRITICOS para cleanup + filtrado en queries).
        "is_seed_data": True,
        "seed_run_id": seed_run_id,
        # TTL para consistencia de schema con items reales.
        "ttl": ttl,
        # Campos de inference fake.
        "s3_key": f"{S3_PREFIX}{prediction_id}.mp3",
        "top1_species": pred["top1_species"],
        "top1_confidence": Decimal(str(round(pred["top1_confidence"], 4))),
        "max_birdnet_confidence": Decimal(
            str(round(pred["max_birdnet_confidence"], 4))
        ),
        "n_windows": int(pred["n_windows"]),
        "top3_predictions": [
            {
                "species": p["species"],
                "common_name": p["common_name"],
                "confidence": Decimal(str(round(p["confidence"], 4))),
            }
            for p in pred["top3_predictions"]
        ],
    }
    return item


# ---------------------------------------------------------------------------
# AWS interaction
# ---------------------------------------------------------------------------

def _boto3_clients():
    """Lazy import + clients setup. Verifica que boto3 puede resolver
    credenciales por su default chain (env vars / ~/.aws/credentials /
    IAM role / etc.). Falla loud si no hay nada.
    """
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 no instalado. pip install boto3") from e

    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError(
            "boto3 no encontro credenciales AWS por ningun mecanismo. "
            "Configurar via: aws configure  /  exportar AWS_ACCESS_KEY_ID "
            "y AWS_SECRET_ACCESS_KEY  /  cargar .env."
        )

    ddb = session.resource("dynamodb", region_name=AWS_REGION)
    s3 = session.client("s3", region_name=AWS_REGION)
    table = ddb.Table(DDB_TABLE_NAME)
    return table, s3


def count_existing_seed_items(table) -> int:
    """Cuenta items con is_seed_data=True via Scan paginado. Bajo cost
    porque la tabla del proyecto tiene <1000 items.
    """
    from boto3.dynamodb.conditions import Attr
    total = 0
    last_key = None
    while True:
        kwargs: dict[str, Any] = {
            "FilterExpression": Attr("is_seed_data").eq(True),
            "Select": "COUNT",
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**kwargs)
        total += resp.get("Count", 0)
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return total


def upload_audio_to_s3(s3, bucket: str, key: str, local_path: Path) -> None:
    """Upload con tagging coherente con Fase 5B retention.

    retain=true: sobrevive al lifecycle de 30 dias (training_consent=True).
    seed_data=true: marker S3 paralelo al de DDB. Permite cleanup por
                    prefix + verificacion cruzada.
    """
    with open(local_path, "rb") as f:
        body = f.read()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="audio/mpeg",
        Tagging="retain=true&seed_data=true",
    )


def write_item_to_ddb(table, item: dict[str, Any]) -> None:
    """PutItem con ConditionExpression para idempotencia (atomic insert).

    Si por algun caso de borde el pk ya existe (uuid colision o re-run
    sin cleanup), levanta ConditionalCheckFailedException.
    """
    table.put_item(
        Item=item,
        ConditionExpression="attribute_not_exists(pk)",
    )


def cleanup_seed_data(table, s3) -> tuple[int, int]:
    """Borra items DDB con is_seed_data=True + objetos S3 con prefix seed/.

    Returns (n_ddb_deleted, n_s3_deleted).
    """
    from boto3.dynamodb.conditions import Attr

    # 1. Scan DDB y batch delete
    keys_to_delete: list[dict[str, str]] = []
    last_key = None
    while True:
        kwargs: dict[str, Any] = {
            "FilterExpression": Attr("is_seed_data").eq(True),
            "ProjectionExpression": "pk, sk",
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**kwargs)
        for item in resp.get("Items", []):
            keys_to_delete.append({"pk": item["pk"], "sk": item["sk"]})
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

    n_ddb = 0
    with table.batch_writer() as batch:
        for key in keys_to_delete:
            batch.delete_item(Key=key)
            n_ddb += 1

    # 2. List + DeleteObjects en S3 con prefix
    n_s3 = 0
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    batch_keys: list[dict[str, str]] = []
    for page in pages:
        for obj in page.get("Contents", []):
            batch_keys.append({"Key": obj["Key"]})
            if len(batch_keys) == 1000:  # max batch size for DeleteObjects
                s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": batch_keys})
                n_s3 += len(batch_keys)
                batch_keys = []
    if batch_keys:
        s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": batch_keys})
        n_s3 += len(batch_keys)

    return n_ddb, n_s3


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_dry_run(
    sampled: pd.DataFrame,
    hard_rate: float,
    seed_run_id: str,
    rng_seed: int,
) -> None:
    print("=" * 70)
    print("DRY-RUN (no se escribe a AWS)")
    print("=" * 70)
    print(f"  seed_run_id:   {seed_run_id}")
    print(f"  random-seed:   {rng_seed}")
    print(f"  hard-rate:     {hard_rate:.2f}")
    print(f"  total items:   {len(sampled)}")

    n_hard = int(round(len(sampled) * hard_rate))
    n_easy = len(sampled) - n_hard
    print(f"  -> easy:       {n_easy} (feedback_status=confirmed)")
    print(f"  -> hard:       {n_hard} (feedback_status=corrected)")

    print()
    print("Distribucion por especie:")
    counts = Counter(sampled["species"])
    for sp in sorted(counts.keys()):
        bar = "#" * counts[sp]
        print(f"  {sp:<30s} {counts[sp]:3d}  {bar}")

    total_bytes = sampled["filepath"].apply(
        lambda fp: (RAW_DIR / fp).stat().st_size
        if (RAW_DIR / fp).exists() else 0
    ).sum()
    print()
    print(f"Total a subir a S3: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Costo DDB writes:   {len(sampled)} WCU (~$0.00 dentro de free tier)")

    print()
    print("Re-ejecutar con --apply para escribir.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    # --- Cleanup mode -------------------------------------------------------
    if args.cleanup:
        table, s3 = _boto3_clients()
        print("Cleanup mode: borrando items DDB con is_seed_data=True + "
              f"objetos S3 con prefix {S3_PREFIX}...", flush=True)
        n_ddb, n_s3 = cleanup_seed_data(table, s3)
        print(f"  DDB items borrados:  {n_ddb}")
        print(f"  S3 objetos borrados: {n_s3}")
        if n_ddb == 0 and n_s3 == 0:
            print("  (no habia nada que borrar)")
        return 0

    # --- Generate mode ------------------------------------------------------
    rng = random.Random(args.random_seed)
    seed_run_id = args.seed_run_id or str(uuid.uuid4())

    print("Cargando audios elegibles (fold=train)...", flush=True)
    eligible = load_eligible_audios()
    print(f"  audios elegibles: {len(eligible)}")
    print(f"  especies:         {eligible['species'].nunique()}")
    print()

    print(f"Sampling {args.count} audios stratified por especie...", flush=True)
    sampled = sample_audios(eligible, args.count, rng)
    print(f"  audios elegidos: {len(sampled)}")
    print()

    if not args.apply:
        report_dry_run(sampled, args.hard_rate, seed_run_id, args.random_seed)
        return 0

    # --- Apply mode: idempotency guard --------------------------------------
    table, s3 = _boto3_clients()
    existing = count_existing_seed_items(table)
    if existing > 0 and not args.append:
        print(f"! ABORTANDO: ya hay {existing} items con is_seed_data=True "
              "en la tabla.", file=sys.stderr)
        print(f"! Corre primero: python {Path(__file__).name} --cleanup",
              file=sys.stderr)
        print(f"! O bypass con --append (NO RECOMENDADO si vas a comparar "
              "runs).", file=sys.stderr)
        return 1
    if existing > 0 and args.append:
        print(f"! WARNING: ya hay {existing} items seed en la tabla, "
              "agregando MAS encima (--append).", file=sys.stderr)
        print(flush=True)

    # --- Pre-compute hard/easy assignments (reproducible) -------------------
    n_hard = int(round(len(sampled) * args.hard_rate))
    indices = list(range(len(sampled)))
    rng.shuffle(indices)
    hard_set = set(indices[:n_hard])

    all_species = sorted(eligible["species"].unique())

    # --- Write loop ---------------------------------------------------------
    print(f"=== APPLY ({len(sampled)} items, seed_run_id={seed_run_id}) ===",
          flush=True)
    t_start = time.time()
    n_ok = 0
    n_fail = 0
    for i, row in sampled.reset_index(drop=True).iterrows():
        prediction_id = str(uuid.uuid4())
        local_path = RAW_DIR / row["filepath"]
        if not local_path.exists():
            print(f"  [{i+1:3d}/{len(sampled)}] SKIP (file missing): {row['filepath']}",
                  flush=True)
            n_fail += 1
            continue

        is_hard = i in hard_set
        true_species = row["species"]
        pred = generate_fake_prediction(true_species, all_species, is_hard, rng)
        item = build_ddb_item(
            prediction_id=prediction_id,
            audio_path=local_path,
            audio_duration_s=float(row.get("duration_s", 10.0)),
            audio_size_bytes=local_path.stat().st_size,
            true_species=true_species,
            pred=pred,
            is_hard=is_hard,
            seed_run_id=seed_run_id,
            rng=rng,
        )

        s3_key = item["s3_key"]
        try:
            upload_audio_to_s3(s3, S3_BUCKET, s3_key, local_path)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(sampled)}] S3 FAIL: {type(e).__name__}: {e}",
                  flush=True)
            n_fail += 1
            continue

        try:
            write_item_to_ddb(table, item)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(sampled)}] DDB FAIL: {type(e).__name__}: {e}",
                  flush=True)
            # Rollback S3: sino queda audio huerfano sin item en DDB.
            try:
                s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
            except Exception:
                pass
            n_fail += 1
            continue

        tag = "HARD" if is_hard else "easy"
        print(f"  [{i+1:3d}/{len(sampled)}] {tag:4s}  {true_species:<30s}  "
              f"pred={pred['top1_species']}  conf={pred['top1_confidence']:.3f}",
              flush=True)
        n_ok += 1

    elapsed = time.time() - t_start
    print()
    print(f"=== Done en {elapsed:.1f}s: ok={n_ok}, fail={n_fail} ===")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
