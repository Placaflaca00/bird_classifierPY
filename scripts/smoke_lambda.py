"""Smoke test standalone contra Lambda en prod (Fase 6.6).

Extraido de ``promote.py:stage4_smoke_test`` para que el workflow
``rollback.yml`` (y eventualmente otros — canary, health check) lo invoquen
sin tener que correr todo el flujo de promote.

Que hace:
    Invoca Lambda con N audios fijos definidos en
    ``scripts/smoke_baseline.json`` (5 audios con sus predicciones esperadas
    persistidas la ultima vez que paso un smoke). Compara top1 + confianza +
    latencia.

    PASS si:
        - top1 species matchea baseline en >= SMOKE_TOP1_THRESHOLD (default 80%)
        - confianza dentro de SMOKE_CONFIDENCE_TOLERANCE_PP del baseline (15pp)
        - latencia p95 <= SMOKE_LATENCY_P95_MAX_S (2s)

Por que esto vive separado de promote.py:
    - rollback.yml necesita correr SOLO el smoke despues de re-deployar
      Lambda al digest anterior; no quiere el resto de promote.
    - Tests futuros (canary, healthcheck periodico) lo pueden reusar.
    - Separation of concerns: promote = full pipeline; este = smoke aislado.

promote.py sigue importando ``smoke_test`` desde aca (re-export) para no
duplicar logica. La firma se mantiene 100% compatible con
``stage4_smoke_test`` original.

CLI:
    python scripts/smoke_lambda.py
    python scripts/smoke_lambda.py --baseline scripts/smoke_baseline.json
    python scripts/smoke_lambda.py --output-json result.json
    python scripts/smoke_lambda.py --function-name bird-classifier-py-inference

Exit codes:
    0 — smoke PASSED
    1 — smoke FAILED (cualquier criterio)
    2 — error de configuracion (baseline missing, audios missing, AWS error)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np

# Repo root: este script vive en scripts/, asi que el root es parent.parent.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constantes — duplicadas a proposito de promote.py para que este modulo no
# dependa de el. Si promote cambia thresholds, sincronizar manualmente o
# mover ambos a un config.json compartido.
# ---------------------------------------------------------------------------
LAMBDA_FUNCTION_NAME = "bird-classifier-py-inference"
AWS_REGION = "us-east-1"

SMOKE_BASELINE_PATH = ROOT / "scripts" / "smoke_baseline.json"

SMOKE_TOP1_THRESHOLD = 0.80          # 4/5 audios deben matchear top1
SMOKE_CONFIDENCE_TOLERANCE_PP = 15.0  # |delta confianza| <= 15pp

# Latency tracking en el smoke (Fase 6.6.a paso 5):
#
#   - SMOKE_LATENCY_MEDIAN_MAX_S: mediana razonable. Con N=4 audios (post
#     warmup-first-skip) p95 era ruido — cualquier outlier setteaba p95.
#     Mediana = mas robusta a outliers de un audio pesado puntual.
#
#   - SMOKE_LATENCY_HARD_CAP_S: cap absoluto por audio. ESTO ES PARA CACHAR
#     CUELGUES (timeout efectivo), no para enforcing de SLO. Si un audio
#     individual demora >5s estamos en problemas reales (cold start grave,
#     red rota, infra degradada). El SLO real de latencia se trackea en
#     CloudWatch (Lambda Insights + custom metrics), donde hay N>>4 muestras
#     y percentiles tienen significado estadistico.
SMOKE_LATENCY_MEDIAN_MAX_S = 1.5
SMOKE_LATENCY_HARD_CAP_S = 5.0
# DEPRECATED — alias compat por si alguien externo lo lee. NO usar nuevo codigo.
SMOKE_LATENCY_P95_MAX_S = SMOKE_LATENCY_HARD_CAP_S

SMOKE_LAMBDA_WARMUP_INVOKES = 2       # invokes pre-medicion para warm Lambda


# ---------------------------------------------------------------------------
# Core helpers (mismos que promote.py)
# ---------------------------------------------------------------------------
def _invoke_lambda_with_audio(
    lambda_client, audio_path: Path, fingerprint: str,
    function_name: str = LAMBDA_FUNCTION_NAME,
) -> tuple[dict, float]:
    """Invoke Lambda directo con audio_b64 (schema v1, sin S3).

    Returns:
        (response_body_dict, latency_seconds)
    """
    audio_bytes = audio_path.read_bytes()
    body = {
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        "fingerprint": fingerprint,
        "top_k": 3,
        "training_consent": False,
    }
    event = {"rawPath": "/predict", "body": json.dumps(body)}
    t0 = time.time()
    resp = lambda_client.invoke(
        FunctionName=function_name,
        Payload=json.dumps(event).encode("utf-8"),
    )
    latency = time.time() - t0
    payload = json.loads(resp["Payload"].read())
    body_str = payload.get("body", "{}")
    return json.loads(body_str), latency


def smoke_test(
    lambda_client=None,
    *,
    baseline_path: Path | None = None,
    function_name: str = LAMBDA_FUNCTION_NAME,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compara invocacion Lambda contra baseline persistido.

    Args:
        lambda_client: boto3 lambda client. Si None, se crea uno default
            sobre ``AWS_REGION``.
        baseline_path: ruta al smoke_baseline.json. Default
            scripts/smoke_baseline.json relativo al repo root.
        function_name: nombre del Lambda function. Permite override para
            testing contra otra version.
        root: repo root para resolver audio_path relativo del baseline.
            Default = parent.parent de este archivo.

    Returns:
        Dict con keys ``passed`` (bool), ``top1_rate``, ``p95_latency_s``,
        ``confidence_violations``, ``per_audio``.

    Raises:
        FileNotFoundError si baseline o audios faltan.
    """
    lambda_client = lambda_client or boto3.client("lambda", region_name=AWS_REGION)
    baseline_path = baseline_path or SMOKE_BASELINE_PATH
    root = root or ROOT

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} no existe. "
            "Correr `python scripts/promote.py --generate-baseline` primero."
        )
    baseline = json.loads(baseline_path.read_text())
    entries = baseline["entries"]

    # Fingerprint estable identificable como smoke (32 hex post fp_).
    # Match el que usa promote.py para que las rate-limit y los logs
    # reconozcan a este caller como sintetico.
    fp_hash = hashlib.md5(b"smoke_test_promote").hexdigest()[:32]
    fingerprint = f"fp_{fp_hash}"

    # Warmup: mitigar cold start antes de medir p95
    print(f"[smoke] Warmup ({SMOKE_LAMBDA_WARMUP_INVOKES} invokes)...")
    for _ in range(SMOKE_LAMBDA_WARMUP_INVOKES):
        try:
            lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps({"warm": True}).encode("utf-8"),
            )
        except Exception:
            pass

    results = []
    latencies = []
    matches = 0
    confidence_violations = []

    print(f"[smoke] Smoke contra {len(entries)} audios baseline...")
    for i, entry in enumerate(entries):
        audio_path = root / entry["audio_path"]
        if not audio_path.exists():
            raise FileNotFoundError(f"baseline audio missing: {audio_path}")
        body, latency = _invoke_lambda_with_audio(
            lambda_client, audio_path, fingerprint,
            function_name=function_name,
        )
        # Excluir primer invoke de la p95 (cold path edge case)
        if i > 0:
            latencies.append(latency)

        predictions = body.get("predictions") or []
        if predictions:
            actual_top1 = predictions[0].get("species")
            actual_conf = float(predictions[0].get("confidence", 0.0))
        else:
            actual_top1 = body.get("reason") or "<no predictions>"
            actual_conf = 0.0
        expected_top1 = entry["expected_top1_species"]
        expected_conf = entry["expected_top1_confidence"]
        top1_match = (actual_top1 == expected_top1)
        conf_diff_pp = abs((actual_conf - expected_conf) * 100)
        conf_ok = conf_diff_pp <= SMOKE_CONFIDENCE_TOLERANCE_PP
        if top1_match:
            matches += 1
        if not conf_ok:
            confidence_violations.append({
                "audio": entry["audio_path"],
                "expected": expected_conf,
                "actual": actual_conf,
                "diff_pp": conf_diff_pp,
            })
        results.append({
            "audio": entry["audio_path"],
            "expected_top1": expected_top1,
            "actual_top1": actual_top1,
            "top1_match": top1_match,
            "expected_conf": expected_conf,
            "actual_conf": actual_conf,
            "conf_diff_pp": conf_diff_pp,
            "latency_s": latency,
        })
        tag = "OK" if top1_match else "MISS"
        print(
            f"  [{tag}] {entry['species_dir']:<25s} top1={actual_top1} "
            f"conf={actual_conf:.4f} (exp {expected_conf:.4f}) "
            f"lat={latency:.2f}s"
        )

    top1_rate = matches / len(entries)
    median_latency = float(np.median(latencies)) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    top1_ok = top1_rate >= SMOKE_TOP1_THRESHOLD
    median_ok = median_latency <= SMOKE_LATENCY_MEDIAN_MAX_S
    cap_ok = max_latency <= SMOKE_LATENCY_HARD_CAP_S
    conf_ok = len(confidence_violations) == 0

    passed = top1_ok and median_ok and cap_ok and conf_ok
    print()
    print(f"[smoke] Resultado:")
    print(f"  top1_rate={top1_rate:.0%} (>= {SMOKE_TOP1_THRESHOLD:.0%}?  {top1_ok})")
    print(f"  median_latency={median_latency:.2f}s (<= {SMOKE_LATENCY_MEDIAN_MAX_S}s?  {median_ok})")
    print(f"  max_latency={max_latency:.2f}s (<= {SMOKE_LATENCY_HARD_CAP_S}s hard cap?  {cap_ok})")
    print(f"  confidence_violations={len(confidence_violations)} (0?  {conf_ok})")
    print(f"  PASSED = {passed}")

    return {
        "passed": passed,
        "top1_rate": top1_rate,
        "median_latency_s": median_latency,
        "max_latency_s": max_latency,
        # DEPRECATED — el p95 con N=4 es ruido. Mantenido para compat con
        # callers viejos que lo leen. Tracking real va a CloudWatch.
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "confidence_violations": confidence_violations,
        "per_audio": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=SMOKE_BASELINE_PATH,
        help="Path al baseline JSON (default: scripts/smoke_baseline.json)",
    )
    parser.add_argument(
        "--function-name", default=LAMBDA_FUNCTION_NAME,
        help=f"Lambda function name (default: {LAMBDA_FUNCTION_NAME})",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Si se pasa, escribe el dict de resultado al path. Util para CI.",
    )
    args = parser.parse_args()

    try:
        result = smoke_test(
            baseline_path=args.baseline,
            function_name=args.function_name,
        )
    except FileNotFoundError as e:
        print(f"! Config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — entrypoint del CLI, queremos exit limpio
        print(f"! Error inesperado: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"\n[smoke] Resultado escrito a {args.output_json}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
