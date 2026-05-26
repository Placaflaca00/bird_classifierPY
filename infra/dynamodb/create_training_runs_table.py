"""Crea la tabla DDB ``bird-classifier-py-training-runs`` para sub-fase 6.4.

Schema diseñado para sub-fase 6.4 (workflow GitHub Actions de retrain) y
sub-fase 6.5 (UI tab Training en bird-annotator-py). Cada item es un run
completo del workflow, con campos llenados secuencialmente por cada job
(export -> train -> evaluate -> promote).

Single-table design siguiendo el patrón de ``bird-classifier-py-data``:
``pk=RUN#<uuid>``, ``sk="META"``. El sk constante deja la partición abierta
a sub-items futuros (e.g. ``AUDIT#...``) sin migracion.

GSI 1 — ``by_triggered_at`` (projection ALL)
============================================
``run_partition`` (constante "RUN") como HASH + ``triggered_at`` como
RANGE permite query time-ordered de TODOS los runs en O(log N), sin Scan.
Pattern de "single-partition GSI" para listas time-ordered moderadas
(<1000 items para nuestra escala — <1 run/semana esperado).

GSI 2 — ``by_status_and_time`` (projection INCLUDE keys + 5 atributos)
======================================================================
HASH=``status`` + RANGE=``triggered_at`` permite "all runs en estado X
ordenados por tiempo" sin Scan. Util para UI dashboard:
  - "runs running ahora" -> Query status="running"
  - "ultimos 10 promoted" -> Query status="promoted", ScanIndexForward=False, limit=10
  - "runs failed esta semana" -> Query status="failed", triggered_at >= "<iso>"

Projection INCLUDE en vez de ALL para reducir costo a la mitad (storage
GSI ~= half de KEYS_ONLY+INCLUDE vs ALL). Atributos elegidos cubren lo
que el dashboard de Fase 5 / UI tab Training de 6.5 necesita listar.

TTL: ``expires_at``
===================
Items se auto-borran 365 dias post-trigger. DynamoDB TTL es best-effort
(borra dentro de 48h del expire), suficiente para retencion de auditoria.
365 dias = razonable para "tener historia visible en CV interview" sin
crecer indefinidamente.

Usage
=====
    python infra/dynamodb/create_training_runs_table.py             # dry-run
    python infra/dynamodb/create_training_runs_table.py --apply     # crea
    python infra/dynamodb/create_training_runs_table.py --smoke     # smoke test
                                                                     # contra tabla ya creada
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = Path(__file__).parent / "training_runs_table.json"

TABLE_NAME = "bird-classifier-py-training-runs"
AWS_REGION = "us-east-1"
TTL_ATTRIBUTE = "expires_at"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="Crea la tabla + configura TTL. Idempotente: si la tabla "
                        "ya existe, reporta y sale.")
    g.add_argument("--smoke", action="store_true",
                   help="Smoke test contra tabla existente: PutItem + Query GSI.")
    g.add_argument("--describe", action="store_true",
                   help="Describe la tabla existente (status + GSIs + TTL).")
    return p.parse_args()


def _boto3_client():
    import boto3
    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError(
            "boto3 no encontro credenciales AWS. Configurar con `aws configure`."
        )
    return session.client("dynamodb", region_name=AWS_REGION)


def report_plan(schema: dict) -> None:
    print("=" * 70)
    print("PLAN — crear tabla DynamoDB")
    print("=" * 70)
    print(f"  TableName:    {schema['TableName']}")
    print(f"  BillingMode:  {schema['BillingMode']}")
    print(f"  KeySchema:    {[k['AttributeName'] for k in schema['KeySchema']]}")
    print()
    print("  Atributos indexados (key + GSI):")
    for ad in schema["AttributeDefinitions"]:
        print(f"    - {ad['AttributeName']:<16s}  {ad['AttributeType']}")
    print()
    print(f"  GSIs ({len(schema['GlobalSecondaryIndexes'])}):")
    for gsi in schema["GlobalSecondaryIndexes"]:
        keys = " + ".join(f"{k['AttributeName']}({k['KeyType']})" for k in gsi["KeySchema"])
        proj = gsi["Projection"]["ProjectionType"]
        print(f"    - {gsi['IndexName']:<24s} {keys:<50s}  projection={proj}")
    print()
    print(f"  TTL attribute: {TTL_ATTRIBUTE} (365 dias post-trigger)")
    print()
    print(f"  Tags:")
    for tag in schema["Tags"]:
        print(f"    - {tag['Key']}={tag['Value']}")
    print()
    print("Re-ejecutar con --apply para crearla.")


def create_table(client, schema: dict) -> None:
    # Idempotencia: si ya existe, exit sin tocar.
    try:
        info = client.describe_table(TableName=TABLE_NAME)
        status = info["Table"]["TableStatus"]
        print(f"Tabla ya existe (status={status}). NO se recrea.")
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    print(f"Creando tabla {TABLE_NAME}...")
    client.create_table(**schema)

    # Wait for ACTIVE
    print("Esperando status=ACTIVE...")
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=TABLE_NAME)

    # Espera adicional a que los GSIs queden ACTIVE
    print("Esperando GSIs ACTIVE...")
    t0 = time.time()
    while True:
        info = client.describe_table(TableName=TABLE_NAME)
        gsis = info["Table"].get("GlobalSecondaryIndexes", [])
        all_active = all(g["IndexStatus"] == "ACTIVE" for g in gsis)
        if all_active:
            break
        if time.time() - t0 > 300:
            raise RuntimeError("GSIs no llegaron a ACTIVE en 5 min")
        time.sleep(5)

    # Configurar TTL
    print(f"Configurando TTL en atributo '{TTL_ATTRIBUTE}'...")
    client.update_time_to_live(
        TableName=TABLE_NAME,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": TTL_ATTRIBUTE},
    )

    print()
    print(f"=== OK: tabla {TABLE_NAME} creada y configurada ===")


def describe_table(client) -> None:
    try:
        info = client.describe_table(TableName=TABLE_NAME)
    except client.exceptions.ResourceNotFoundException:
        print(f"Tabla {TABLE_NAME} NO existe. Correr --apply primero.")
        sys.exit(1)

    t = info["Table"]
    print(f"Tabla:        {t['TableName']}")
    print(f"Status:       {t['TableStatus']}")
    print(f"ItemCount:    {t['ItemCount']} (eventually consistent)")
    print(f"SizeBytes:    {t['TableSizeBytes']}")
    print(f"Created:      {t['CreationDateTime']}")
    print()
    print("GSIs:")
    for gsi in t.get("GlobalSecondaryIndexes", []):
        print(f"  - {gsi['IndexName']:<26s}  status={gsi['IndexStatus']}  "
              f"items={gsi.get('ItemCount', '?')}")
    print()

    try:
        ttl = client.describe_time_to_live(TableName=TABLE_NAME)
        ttl_desc = ttl["TimeToLiveDescription"]
        print(f"TTL:          {ttl_desc.get('TimeToLiveStatus', 'DISABLED')}  "
              f"attr={ttl_desc.get('AttributeName', 'n/a')}")
    except Exception as e:
        print(f"TTL describe fallo: {e}")


def smoke_test(client) -> None:
    """End-to-end: PutItem -> Query GSI by_triggered_at -> Query GSI by_status
    -> UpdateItem (idempotency pattern) -> DeleteItem.
    """
    print("=== Smoke test ===")
    run_id = str(uuid.uuid4())
    pk = f"RUN#{run_id}"
    now = datetime.now(timezone.utc)
    triggered_at = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    expires_at = int((now + timedelta(days=365)).timestamp())

    item = {
        "pk": {"S": pk},
        "sk": {"S": "META"},
        "run_id": {"S": run_id},
        "run_partition": {"S": "RUN"},  # constante para GSI by_triggered_at
        "status": {"S": "running"},
        "current_job": {"S": "export"},
        "triggered_at": {"S": triggered_at},
        "triggered_by": {"S": "smoke_test"},
        "triggered_via": {"S": "smoke_test"},
        "include_seed": {"BOOL": True},
        "github_run_id": {"S": "smoke-000"},
        "github_run_url": {"S": "https://github.com/Placaflaca00/bird_classifierPY/actions/runs/smoke"},
        "git_sha": {"S": "smoke" + "0" * 35},
        "expires_at": {"N": str(expires_at)},
    }

    print(f"  1. PutItem (pk={pk})...")
    client.put_item(TableName=TABLE_NAME, Item=item)
    print(f"     OK")

    print(f"  2. Query GSI by_triggered_at (run_partition=RUN, last 24h)...")
    yesterday = (now - timedelta(days=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    resp = client.query(
        TableName=TABLE_NAME,
        IndexName="by_triggered_at",
        KeyConditionExpression="run_partition = :rp AND triggered_at >= :ts",
        ExpressionAttributeValues={":rp": {"S": "RUN"}, ":ts": {"S": yesterday}},
    )
    count = resp.get("Count", 0)
    print(f"     OK items={count} (esperado >=1)")
    assert count >= 1, f"by_triggered_at devolvio {count} items, esperado >=1"

    print(f"  3. Query GSI by_status_and_time (status=running)...")
    resp = client.query(
        TableName=TABLE_NAME,
        IndexName="by_status_and_time",
        KeyConditionExpression="#s = :st",
        ExpressionAttributeNames={"#s": "status"},  # status es reserved
        ExpressionAttributeValues={":st": {"S": "running"}},
    )
    count = resp.get("Count", 0)
    print(f"     OK items={count} (esperado >=1)")
    assert count >= 1, f"by_status_and_time devolvio {count} items, esperado >=1"

    print(f"  4. UpdateItem idempotente (status running->succeeded)...")
    # Idempotency pattern: ConditionExpression verifica current_job=export.
    # Si esta carrera ya completo, ConditionalCheckFailedException -> skip silencioso.
    try:
        client.update_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": pk}, "sk": {"S": "META"}},
            UpdateExpression="SET #s = :new_status, current_job = :job",
            ConditionExpression="current_job = :expected_job",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":new_status": {"S": "succeeded"},
                ":job": {"S": "done"},
                ":expected_job": {"S": "export"},
            },
        )
        print(f"     OK (status -> succeeded)")
    except client.exceptions.ConditionalCheckFailedException:
        print(f"     SKIP silencioso (current_job != export, idempotent OK)")

    print(f"  5. DeleteItem (cleanup smoke)...")
    client.delete_item(TableName=TABLE_NAME, Key={"pk": {"S": pk}, "sk": {"S": "META"}})
    print(f"     OK")

    print()
    print("=== Smoke 5/5 PASS ===")


def main() -> int:
    args = parse_args()

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    if args.describe:
        client = _boto3_client()
        describe_table(client)
        return 0

    if args.smoke:
        client = _boto3_client()
        smoke_test(client)
        return 0

    if args.apply:
        client = _boto3_client()
        create_table(client, schema)
        describe_table(client)
        return 0

    # Dry-run default
    report_plan(schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
