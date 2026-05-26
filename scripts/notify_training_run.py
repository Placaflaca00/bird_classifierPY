"""Actualiza el item DDB ``RUN#<run_id>`` en ``bird-classifier-py-training-runs``.

Helper standalone que cada job del workflow GitHub Actions de Fase 6 invoca
para reportar su estado. Pattern: ``UpdateItem`` con ``ConditionExpression``
para idempotency — si GH Actions reintenta un job o si dos jobs llegan
fuera de orden, no corrompe el item.

Idempotency via job_order monotonic
====================================
Cada job tiene un ``job_order`` entero monotonic (start=0 < export=1 <
train=2 < evaluate=3 < promote=4 < finalize=5). El ``UpdateItem`` solo
aplica si ``:new_order >= :current_order`` en el item. Esto previene:

  - Re-ejecucion de jobs anteriores (un retry de "export" no sobreescribe
    un "train" ya completo).
  - Race conditions teoricas (los jobs corren secuenciales en GH Actions,
    pero defensa en profundidad).
  - Estados terminales (status=failed se setea con job_order=99 para que
    nada lo sobreescriba accidentalmente).

Actions
=======
``--action create``: PutItem inicial. ConditionExpression
``attribute_not_exists(pk)`` previene duplicados si el trigger se
reintenta. Si ya existe, skip silencioso (idempotente).

``--action update``: UpdateItem. Si el job actual del item es posterior
al pasado, skip silencioso. Sino, actualiza todos los campos pasados.

``--action finalize``: UpdateItem final (status terminal + completed_at).
Setea job_order=99 (failed_critical) o job_order=5 (succeeded/promoted)
segun ``--status``.

Usage
=====
    # Job start (inicial):
    python scripts/notify_training_run.py \\
        --run-id "$RUN_ID" --action create \\
        --status running --current-job start \\
        --field triggered_by="github-actions" \\
        --field triggered_via="workflow_dispatch" \\
        --field github_run_id="${{ github.run_id }}" \\
        --field github_run_url="https://github.com/$REPO/actions/runs/${{ github.run_id }}" \\
        --field git_sha="${{ github.sha }}" \\
        --field include_seed=true

    # Job intermedio (update):
    python scripts/notify_training_run.py \\
        --run-id "$RUN_ID" --action update \\
        --current-job train \\
        --field best_val_macro_f1=0.909 \\
        --field state_dict_sha256="$SHA" \\
        --field training_current_epoch=22

    # Job final (finalize):
    python scripts/notify_training_run.py \\
        --run-id "$RUN_ID" --action finalize \\
        --status succeeded
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

AWS_REGION = "us-east-1"
TABLE_NAME = "bird-classifier-py-training-runs"
TTL_DAYS = 365

# Job order monotonic. Nuevo job solo puede ser >= al actual (excepto
# estados terminales que se setean al final).
JOB_ORDER: dict[str, int] = {
    "start": 0,
    "export": 1,
    "train": 2,
    "evaluate": 3,
    "promote": 4,
    "finalize": 5,
    "done": 5,            # alias de finalize
    # Terminal: cualquier valor superior es sticky (no se sobreescribe)
    "failed_critical": 99,
}

# Status terminales (no se sobreescriben una vez seteados)
TERMINAL_STATUSES = {"succeeded", "promoted", "rejected", "failed", "failed_critical"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", required=True, help="UUID4 del run.")
    p.add_argument("--action", required=True,
                   choices=["create", "update", "finalize"],
                   help="create=PutItem inicial, update=UpdateItem intermedio, "
                        "finalize=UpdateItem terminal.")
    p.add_argument("--status", default=None,
                   help="running | succeeded | promoted | rejected | failed | "
                        "failed_critical")
    p.add_argument("--current-job", default=None,
                   help="start | export | train | evaluate | promote | finalize")
    p.add_argument("--field", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Campo a setear. Repetible. Valor auto-detecta tipo: "
                        "'true'/'false'->bool, parseable como int/float->number, "
                        "sino string.")
    p.add_argument("--failure-reason", default=None,
                   help="Solo aplica si --status=failed_critical. Texto libre "
                        "que la UI muestra en banner rojo.")
    return p.parse_args()


def _parse_value(raw: str) -> Any:
    """Auto-detecta tipo del valor pasado en --field key=value."""
    lower = raw.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if raw == "null" or raw == "":
        return None
    # Probar int primero
    try:
        return int(raw)
    except ValueError:
        pass
    # Probar float -> Decimal (DDB no acepta float)
    try:
        return Decimal(raw)
    except Exception:
        pass
    return raw


def _parse_fields(field_args: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in field_args:
        if "=" not in f:
            print(f"! Field '{f}' no tiene '='. Skip.", file=sys.stderr)
            continue
        k, v = f.split("=", 1)
        parsed = _parse_value(v)
        if parsed is None:
            continue  # no setear None
        out[k.strip()] = parsed
    return out


def _boto3_table():
    import boto3
    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError("boto3 no encontro credenciales AWS.")
    return session.resource("dynamodb", region_name=AWS_REGION).Table(TABLE_NAME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _now_plus_ttl() -> int:
    return int((datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).timestamp())


def do_create(table, run_id: str, status: str, current_job: str,
              fields: dict[str, Any]) -> int:
    """PutItem inicial. Idempotente: si pk ya existe, skip."""
    item = {
        "pk": f"RUN#{run_id}",
        "sk": "META",
        "run_id": run_id,
        "run_partition": "RUN",  # constante para GSI by_triggered_at
        "status": status or "running",
        "current_job": current_job or "start",
        "job_order_int": JOB_ORDER.get(current_job or "start", 0),
        "triggered_at": _now_iso(),
        "expires_at": _now_plus_ttl(),
    }
    item.update(fields)
    try:
        table.put_item(Item=item, ConditionExpression="attribute_not_exists(pk)")
        print(f"OK create: RUN#{run_id} (status={item['status']}, "
              f"current_job={item['current_job']})")
        return 0
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        print(f"SKIP create: RUN#{run_id} ya existe (idempotency OK)")
        return 0


def do_update(table, run_id: str, status: str | None, current_job: str | None,
              fields: dict[str, Any]) -> int:
    """UpdateItem intermedio. Solo aplica si el nuevo job_order >= actual."""
    new_order: int | None = None
    if current_job is not None:
        new_order = JOB_ORDER.get(current_job)
        if new_order is None:
            print(f"! current_job '{current_job}' no esta en JOB_ORDER. Abort.",
                  file=sys.stderr)
            return 1

    # Construir UpdateExpression con SET dinamico
    set_parts: list[str] = []
    names: dict[str, str] = {}
    values: dict[str, Any] = {}

    if status is not None:
        set_parts.append("#st = :st")
        names["#st"] = "status"
        values[":st"] = status
    if current_job is not None:
        set_parts.append("current_job = :cj")
        values[":cj"] = current_job
        set_parts.append("job_order_int = :jo")
        values[":jo"] = new_order

    # Campos genericos pasados via --field. Usar attribute names aliasados
    # por si conflictuan con reserved keywords.
    for i, (k, v) in enumerate(fields.items()):
        ph_name = f"#f{i}"
        ph_val = f":f{i}"
        names[ph_name] = k
        values[ph_val] = v
        set_parts.append(f"{ph_name} = {ph_val}")

    if not set_parts:
        print("! Nada que actualizar (sin --status, --current-job, ni --field). Abort.",
              file=sys.stderr)
        return 1

    # ConditionExpression idempotency: solo si nuevo orden >= actual
    # (o si el item aun no tiene job_order_int — primer update post-create).
    condition = "attribute_exists(pk)"
    if new_order is not None:
        condition += " AND (attribute_not_exists(job_order_int) OR job_order_int <= :jo_max)"
        values[":jo_max"] = new_order

    update_expr = "SET " + ", ".join(set_parts)

    # DDB rechaza ExpressionAttributeNames={} (vacio) con ValidationException.
    # Solo pasarlo si tiene al menos una entry. Status (#st) y --field key=value
    # son los unicos que agregan a names; si ninguno de los dos esta, el dict
    # queda vacio y hay que omitirlo del call.
    kwargs: dict[str, Any] = {
        "Key": {"pk": f"RUN#{run_id}", "sk": "META"},
        "UpdateExpression": update_expr,
        "ConditionExpression": condition,
        "ExpressionAttributeValues": values,
    }
    if names:
        kwargs["ExpressionAttributeNames"] = names

    try:
        table.update_item(**kwargs)
        print(f"OK update: RUN#{run_id} "
              f"(current_job={current_job or '<unchanged>'}, "
              f"status={status or '<unchanged>'}, fields={len(fields)})")
        return 0
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        print(f"SKIP update: RUN#{run_id} ya esta en job posterior "
              "(idempotency OK)", file=sys.stderr)
        return 0


def do_finalize(table, run_id: str, status: str, fields: dict[str, Any],
                failure_reason: str | None) -> int:
    """UpdateItem terminal. Setea completed_at + status + job_order final
    (5 para succeeded/promoted/rejected, 99 para failed/failed_critical
    para que nada lo sobreescriba accidentalmente). NO usa ConditionExpression
    de job_order (queremos que terminal-state gane siempre).
    """
    if status not in TERMINAL_STATUSES:
        print(f"! finalize requiere --status terminal "
              f"({sorted(TERMINAL_STATUSES)}). Got: {status}", file=sys.stderr)
        return 1

    final_order = 99 if "failed" in status else 5

    set_parts = ["#st = :st", "completed_at = :ct", "job_order_int = :jo",
                 "current_job = :cj"]
    names = {"#st": "status"}
    values = {
        ":st": status,
        ":ct": _now_iso(),
        ":jo": final_order,
        ":cj": "done" if "failed" not in status else "failed",
    }
    if failure_reason:
        set_parts.append("failure_reason = :fr")
        values[":fr"] = failure_reason
    for i, (k, v) in enumerate(fields.items()):
        ph_name = f"#f{i}"
        ph_val = f":f{i}"
        names[ph_name] = k
        values[ph_val] = v
        set_parts.append(f"{ph_name} = {ph_val}")

    try:
        table.update_item(
            Key={"pk": f"RUN#{run_id}", "sk": "META"},
            UpdateExpression="SET " + ", ".join(set_parts),
            ConditionExpression="attribute_exists(pk)",
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
        print(f"OK finalize: RUN#{run_id} status={status}")
        return 0
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        print(f"! ABORT finalize: RUN#{run_id} no existe en la tabla.",
              file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    fields = _parse_fields(args.field)

    table = _boto3_table()

    if args.action == "create":
        return do_create(table, args.run_id, args.status or "running",
                         args.current_job or "start", fields)
    if args.action == "update":
        return do_update(table, args.run_id, args.status, args.current_job, fields)
    if args.action == "finalize":
        return do_finalize(table, args.run_id, args.status or "failed",
                           fields, args.failure_reason)
    print(f"! action desconocida: {args.action}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
