"""Confirmacion PROD del audit local (Fase 6.6 diagnostico).

Lee el ultimo CSV de diag_filter_audit, selecciona el sample del borde
(2 decode + 8 DSP + 17 BirdNET 0.05-0.10 + 10 passes baja confianza),
invoca Lambda directo via boto3 con delay 2s, compara local_outcome vs
prod_outcome. NO toca thresholds, NO modifica nada.

Por que invoke directo y no API Gateway:
    Misma logica de filter chain — el rate limit se chequea en handler.py
    no en API GW. La diferencia API GW es solo headers + edge CloudFront
    cache. Para confirmar comportamiento de filter chain alcanza con boto3.

Rate limit:
    Usamos 2 fingerprints rotando para mantenernos bajo 30/dia por fp.
    Si llega un 429 en respuesta -> log como confound, no como reject.

Uso:
    python scripts/diag_filter_audit_prod.py
    python scripts/diag_filter_audit_prod.py --csv reports/diagnostics/test_hard_audit_<ts>.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent

LAMBDA_FUNCTION_NAME = "bird-classifier-py-inference"
AWS_REGION = "us-east-1"
DELAY_S = 2.0
FINGERPRINTS = [
    f"fp_{hashlib.md5(b'diag_audit_prod_fp_a').hexdigest()[:32]}",
    f"fp_{hashlib.md5(b'diag_audit_prod_fp_b').hexdigest()[:32]}",
]


def _latest_csv() -> Path:
    csvs = sorted((ROOT / "reports" / "diagnostics").glob("test_hard_audit_*.csv"))
    if not csvs:
        sys.exit("! No hay CSV en reports/diagnostics/. Correr diag_filter_audit primero.")
    return csvs[-1]


def _select_sample(rows: list[dict]) -> list[dict]:
    """Sample del borde: 2 decode + 8 DSP + 17 BirdNET 0.05-0.10 + 10 passes."""
    decode_fail = [r for r in rows if r["outcome"] == "decode_fail"]
    white_noise = [r for r in rows if r["layer_reject"] == "white_noise"]
    pure_tone = [r for r in rows if r["layer_reject"] == "pure_tone"]

    # BirdNET banda 0.05-0.10 (no sub-0.05 — no puede flipear)
    birdnet = []
    for r in rows:
        if r["layer_reject"] == "not_a_bird":
            mc = float(r["max_birdnet_confidence"])
            if mc >= 0.05:
                birdnet.append(r)
    # Top de banda primero (mas cerca de threshold)
    birdnet.sort(key=lambda r: float(r["max_birdnet_confidence"]), reverse=True)

    # Passes de baja confianza (mas potencialmente flippable)
    passed = [r for r in rows if r["outcome"] == "passed" and r["top1_confidence"]]
    passed.sort(key=lambda r: float(r["top1_confidence"]))
    passed_low = passed[:10]

    sample = decode_fail + white_noise + pure_tone + birdnet + passed_low
    return sample


def _classify_prod_response(http_status: int, body: dict) -> tuple[str, str]:
    """Mapea respuesta de Lambda a (outcome, layer_reject) coherente con
    el formato local del CSV."""
    if http_status == 400:
        return "decode_fail", "decode"
    if http_status >= 500:
        return "server_error", f"http_{http_status}"
    if http_status == 429:
        return "rate_limited", "throttle"
    if http_status != 200:
        return f"http_{http_status}", "unknown"
    # 200 OK — distinguir reject (detected=False) de pass (tiene predictions)
    if body.get("detected") is False:
        reason = body.get("reason") or "unknown_reject"
        return "rejected", reason
    if body.get("predictions"):
        return "passed", ""
    return "unknown", "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV del audit local. Default: ultimo en reports/diagnostics/")
    parser.add_argument("--delay-s", type=float, default=DELAY_S,
                        help=f"Delay entre invokes (default {DELAY_S}s)")
    args = parser.parse_args()

    csv_path = args.csv or _latest_csv()
    print(f"[audit-prod] Leyendo: {csv_path.name}")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = _select_sample(rows)
    print(f"[audit-prod] Sample: {len(sample)} audios")
    by_local = Counter(
        f"{r['outcome']}/{r['layer_reject']}" for r in sample
    )
    for k, v in sorted(by_local.items()):
        print(f"  local {k}: {v}")
    print(f"[audit-prod] Delay {args.delay_s}s, fingerprints rotando ({len(FINGERPRINTS)})")

    lambda_client = boto3.client("lambda", region_name=AWS_REGION)

    results = []
    matches = 0
    confounded = 0  # rate-limited / server-error
    fp_counts = Counter()

    for i, local_row in enumerate(sample):
        audio_path = ROOT / local_row["rel_path"]
        local_outcome = local_row["outcome"]
        local_layer = local_row["layer_reject"]

        # Rotacion fingerprint: el 1er fp toma 30 max, despues rota
        fp = FINGERPRINTS[fp_counts.total() // 30 % len(FINGERPRINTS)]
        fp_counts[fp] += 1

        audio_bytes = audio_path.read_bytes()
        body = {
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "fingerprint": fp,
            "top_k": 1,
            "training_consent": False,
        }
        event = {"rawPath": "/predict", "body": json.dumps(body)}

        t0 = time.time()
        try:
            resp = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                Payload=json.dumps(event).encode("utf-8"),
            )
            payload = json.loads(resp["Payload"].read())
            latency = time.time() - t0
            http_status = int(payload.get("statusCode", 0))
            body_str = payload.get("body", "{}")
            body_dict = json.loads(body_str) if body_str else {}
            prod_outcome, prod_layer = _classify_prod_response(http_status, body_dict)
            error = None
        except Exception as e:  # noqa: BLE001
            latency = time.time() - t0
            http_status = -1
            prod_outcome = "invoke_error"
            prod_layer = type(e).__name__
            error = str(e)

        match = (local_outcome == prod_outcome) and (
            (local_layer == prod_layer) or
            (local_outcome == "passed" and prod_outcome == "passed")
        )
        if match:
            matches += 1
        if prod_outcome in ("rate_limited", "server_error", "invoke_error"):
            confounded += 1

        results.append({
            "rel_path": local_row["rel_path"],
            "local_outcome": local_outcome,
            "local_layer": local_layer,
            "prod_outcome": prod_outcome,
            "prod_layer": prod_layer,
            "match": match,
            "http_status": http_status,
            "local_max_birdnet_conf": local_row.get("max_birdnet_confidence", ""),
            "local_top1_conf": local_row.get("top1_confidence", ""),
            "latency_s": round(latency, 2),
            "fingerprint_short": fp[3:11],
            "error": error,
        })

        flag = "OK " if match else "DIFF"
        print(f"  [{i+1:2d}/{len(sample)}] {flag}  "
              f"local={local_outcome}/{local_layer:<12s}  "
              f"prod={prod_outcome}/{prod_layer:<12s}  "
              f"{audio_path.parent.name}/{audio_path.name}")

        if i < len(sample) - 1:
            time.sleep(args.delay_s)

    # CSV output
    out_dir = ROOT / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"test_hard_audit_prod_{ts}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Resumen
    print()
    print("=" * 80)
    print(f"RESUMEN PROD vs LOCAL — sample {len(sample)} audios")
    print(f"  matches local==prod:           {matches}/{len(sample)}  ({100*matches/len(sample):.1f}%)")
    print(f"  mismatches:                    {len(sample) - matches - confounded}")
    print(f"  confounded (429/500/error):    {confounded}")
    print()
    mismatches = [r for r in results if not r["match"] and r["prod_outcome"] not in
                  ("rate_limited", "server_error", "invoke_error")]
    if mismatches:
        print("MISMATCHES (local != prod, no confound):")
        for m in mismatches:
            print(f"  {m['rel_path']}")
            print(f"    local = {m['local_outcome']}/{m['local_layer']}")
            print(f"    prod  = {m['prod_outcome']}/{m['prod_layer']}  http={m['http_status']}")
            if m.get("local_max_birdnet_conf"):
                print(f"    local_max_birdnet_conf={m['local_max_birdnet_conf']}")
    else:
        print("ZERO MISMATCHES — local == prod en el borde testeado.")
    print()
    print(f"CSV: {out_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
