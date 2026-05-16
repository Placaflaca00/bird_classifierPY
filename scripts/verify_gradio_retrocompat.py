"""Verifica retrocompat del client.py + app.py PARCHEADOS contra:

    A) Handler NUEVO (Fase 2 Nivel 2, local): 4 casos
       - bird real -> predicciones
       - voz humana -> reject_reason=not_a_bird
       - white noise sintetico -> reject_reason=white_noise
       - tono 440 Hz -> reject_reason=pure_tone

    B) Handler VIEJO Fase 1 (mock, simulando shape pre-Nivel-2): 2 casos
       - response valida con predictions: list, sin "detected" key -> ok=True
         con detected=True por default (retrocompat transparente).
       - response con shape malformado (sin predictions) -> bad_response.

    C) Lambda PRODUCCION (real, network): 1 caso
       - bird real al endpoint vivito (Fase 1). Confirma que el hotfix
         mantiene retrocompat con la Lambda en prod hoy.

Output: tabla 7 casos con verdict. Si todos OK -> seguro pushear Gradio
antes que Lambda.

Uso:
    python -u scripts/verify_gradio_retrocompat.py
    python -u scripts/verify_gradio_retrocompat.py --skip-prod  # sin caso C
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import client as cli  # type: ignore  # noqa: E402

# Handler nuevo local
spec = importlib.util.spec_from_file_location("h", ROOT / "lambda" / "handler.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


PROD_API_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"


# ---------------------------------------------------------------------------
# Replica del branching user-facing de app.classify (sin instanciar Gradio).
# Acompasa con app/app.py despues del hotfix.
# ---------------------------------------------------------------------------
_ERROR_MESSAGES = {
    "throttled": "...saturado...",
    "timeout": "...tardo demasiado...",
    "network": "...no se pudo conectar...",
    "server": "...error interno del servidor...",
    "bad_response": "El servidor respondio algo inesperado. Proba de nuevo en un rato.",
    "config": "...mal configurado...",
}
_REJECT_MESSAGES = {
    "not_a_bird": "No detecte un ave en el audio.",
    "white_noise": "El audio parece ruido sintetico.",
    "pure_tone": "El audio es un tono puro.",
}


def simulate_ui(result: cli.PredictResult) -> str:
    """Mensaje que veria el usuario en la UI, segun la logica de app.classify."""
    if not result.ok:
        if result.error_kind == "bad_request":
            return f"[ERROR_BOX] No se pudo procesar el audio: {result.error_message}"
        return f"[ERROR_BOX] {_ERROR_MESSAGES.get(result.error_kind, 'Ocurrio un error inesperado.')}"
    if not result.detected:
        msg = _REJECT_MESSAGES.get(result.reject_reason or "not_a_bird", _REJECT_MESSAGES["not_a_bird"])
        return f"[ERROR_BOX reject:{result.reject_reason}] {msg}"
    if not result.predictions:
        return "[ERROR_BOX] El servidor no devolvio predicciones."
    top = result.predictions[0]
    return f"[RESULT] top-1={top.get('species')} ({top.get('confidence'):.4f})"


class MockResponse:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body
    def json(self) -> dict:
        return self._body


def call_local_handler(audio_path: Path) -> dict:
    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    ev = {"body": json.dumps({"audio_b64": b64, "top_k": 3})}
    r = h.handler(ev, None)
    return {"status": r["statusCode"], "body": json.loads(r["body"])}


def fake_phase1_response_bird() -> dict:
    """Shape EXACTO de la Lambda Fase 1 vieja (sin keys de Nivel 2)."""
    return {
        "status": 200,
        "body": {
            "predictions": [
                {"species": "Calidris canutus", "common_name": "playero rojizo", "common_name_en": "Red Knot", "confidence": 0.9972},
                {"species": "Calidris alpina", "common_name": None, "common_name_en": None, "confidence": 0.0014},
                {"species": "Calidris pusilla", "common_name": None, "common_name_en": None, "confidence": 0.0009},
            ],
            "model_version": "classifier_v1",
            "n_windows": 5,
            "inference_time_ms": 128.0,
        },
    }


def fake_phase1_response_malformed() -> dict:
    """Shape malformado del legacy: response 200 sin predictions list."""
    return {
        "status": 200,
        "body": {"model_version": "classifier_v1", "n_windows": 5},
    }


def report_case(case_id: str, label: str, resp: dict, expected_kind: str) -> bool:
    print(f"\n--- [{case_id}] {label} ---", flush=True)
    print(f"    handler status={resp['status']}  keys={list(resp['body'].keys())}", flush=True)
    if "reason" in resp["body"]:
        print(f"        reason={resp['body']['reason']!r}", flush=True)

    mock = MockResponse(resp["status"], resp["body"])
    result = cli._map_response(mock)
    print(f"    client._map_response -> ok={result.ok}  detected={result.detected}  reject_reason={result.reject_reason!r}", flush=True)
    if result.ok and result.predictions:
        print(f"        predictions={len(result.predictions)} top-1={result.predictions[0]['species']}", flush=True)
    elif not result.ok:
        print(f"        error_kind={result.error_kind!r}  msg={result.error_message!r}", flush=True)

    ui = simulate_ui(result)
    print(f"    UI -> {ui}", flush=True)

    # verdict matching
    if expected_kind == "bird":
        ok = result.ok and result.detected and len(result.predictions) > 0
    elif expected_kind == "reject_not_a_bird":
        ok = result.ok and not result.detected and result.reject_reason == "not_a_bird"
    elif expected_kind == "reject_white_noise":
        ok = result.ok and not result.detected and result.reject_reason == "white_noise"
    elif expected_kind == "reject_pure_tone":
        ok = result.ok and not result.detected and result.reject_reason == "pure_tone"
    elif expected_kind == "phase1_bird":
        ok = result.ok and result.detected and len(result.predictions) > 0
    elif expected_kind == "phase1_malformed":
        ok = (not result.ok) and result.error_kind == "bad_response"
    else:
        ok = False
    verdict = "OK" if ok else "FAIL"
    print(f"    >>> {verdict} (esperado: {expected_kind})", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prod", action="store_true", help="No llamar Lambda prod (caso C).")
    args = ap.parse_args()

    print(f"\n{'='*100}", flush=True)
    print("Verificacion retrocompat — client.py + app.py PARCHEADOS", flush=True)
    print(f"{'='*100}", flush=True)

    results: list[tuple[str, bool]] = []

    # ----- A) Handler nuevo local (4 casos) -----
    print(f"\n### A) Handler NUEVO (Fase 2 Nivel 2, local) ###", flush=True)
    cases_local = [
        ("A1", "bird real (Calidris)", ROOT/"data/raw/calidris_canutus/GBIF1975781150.mp3", "bird"),
        ("A2", "voz humana (regression)", ROOT/"tests/fixtures/voz_humana_6s.wav", "reject_not_a_bird"),
        ("A3", "white noise", ROOT/"tests/fixtures/white_noise_5s.wav", "reject_white_noise"),
        ("A4", "tono 440 Hz", ROOT/"tests/fixtures/sine_440hz_5s.wav", "reject_pure_tone"),
    ]
    for cid, label, path, expected in cases_local:
        resp = call_local_handler(path)
        ok = report_case(cid, label, resp, expected)
        results.append((cid, ok))

    # ----- B) Handler viejo Fase 1 mock (2 casos) -----
    print(f"\n### B) Handler VIEJO Fase 1 mock (sin keys de Nivel 2) ###", flush=True)
    results.append(("B1", report_case(
        "B1", "Fase 1 bird mock (predictions sin 'detected')",
        fake_phase1_response_bird(), "phase1_bird",
    )))
    results.append(("B2", report_case(
        "B2", "Fase 1 malformed mock (sin predictions)",
        fake_phase1_response_malformed(), "phase1_malformed",
    )))

    # ----- C) Lambda PRODUCCION (1 caso real network) -----
    if not args.skip_prod:
        print(f"\n### C) Lambda PROD real (Fase 1 vivita) ###", flush=True)
        print(f"    endpoint: {PROD_API_URL}", flush=True)
        print(f"    NOTA: cold start puede tardar ~26 s la primera vez.", flush=True)
        audio_path = ROOT / "data/raw/calidris_canutus/GBIF1975781150.mp3"
        t0 = time.time()
        result = cli.predict(audio_path, top_k=3, api_url=PROD_API_URL)
        latency_s = time.time() - t0
        print(f"    e2e latency: {latency_s:.1f} s", flush=True)
        print(f"    PredictResult -> ok={result.ok}  detected={result.detected}", flush=True)
        if result.ok and result.predictions:
            print(f"        top-1={result.predictions[0]['species']} ({result.predictions[0]['confidence']:.4f})", flush=True)
            print(f"        model_version={result.model_version}  inf_ms={result.inference_time_ms}", flush=True)
        elif not result.ok:
            print(f"        error_kind={result.error_kind!r}  msg={result.error_message!r}", flush=True)
        ui = simulate_ui(result)
        print(f"    UI -> {ui}", flush=True)
        ok = result.ok and result.detected and len(result.predictions) > 0
        verdict = "OK" if ok else "FAIL"
        print(f"    >>> {verdict}", flush=True)
        results.append(("C1", ok))

    # ----- Tabla resumen -----
    print(f"\n{'='*100}", flush=True)
    print("RESUMEN", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'caso':<6} {'verdict':<6}", flush=True)
    for cid, ok in results:
        print(f"{cid:<6} {'OK' if ok else 'FAIL':<6}", flush=True)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} PASS", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
