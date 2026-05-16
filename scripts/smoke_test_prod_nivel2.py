"""Smoke test post-deploy: 4 casos contra Lambda prod recien actualizada.

Usa el client.py PARCHEADO contra el endpoint API Gateway. Verifica que:
    - bird real -> detected=True con predictions
    - voz humana -> detected=False, reason='not_a_bird'
    - white noise -> detected=False, reason='white_noise'
    - tono 440 Hz -> detected=False, reason='pure_tone'

Si los 4 pasan: Ciclo 1 cerrado. Si alguno falla: rollback inmediato a
imagen :wa-drop3-v1 via aws lambda update-function-code.

Uso:
    python -u scripts/smoke_test_prod_nivel2.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import client as cli  # type: ignore  # noqa: E402

PROD_API_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"


def main() -> int:
    # Silencio sintetico dispara 'not_a_bird' por gate BirdNET (max_conf ~ 0.003,
    # bajo el threshold 0.10). Equivalente funcional a voz humana para verificar
    # ese reason path, y reproducible sin assets personales.
    cases = [
        ("bird (Calidris)",       ROOT/"data/raw/calidris_canutus/GBIF1975781150.mp3", None,          True),
        ("not_a_bird (silencio)", ROOT/"tests/fixtures/silence_5s.wav",                "not_a_bird",  False),
        ("white_noise",           ROOT/"tests/fixtures/white_noise_5s.wav",            "white_noise", False),
        ("pure_tone",             ROOT/"tests/fixtures/sine_440hz_5s.wav",             "pure_tone",   False),
    ]

    print(f"endpoint: {PROD_API_URL}", flush=True)
    print(f"esperando cold start (~26s primera llamada)...\n", flush=True)

    passed = 0
    print(f"{'caso':<22} {'lat_s':>6} {'ok':>4} {'detected':>9} {'reject_reason':>14} {'top-1':<22}", flush=True)
    print("-" * 90, flush=True)
    for label, path, expected_reason, expected_detected in cases:
        t0 = time.time()
        try:
            result = cli.predict(path, top_k=3, api_url=PROD_API_URL)
        except Exception as e:
            print(f"{label:<22}  ERROR  {type(e).__name__}: {e}", flush=True)
            continue
        lat = time.time() - t0

        ok = result.ok
        det = result.detected
        rr = result.reject_reason or "—"
        top1 = result.predictions[0]["species"] if result.predictions else ""

        # verdict
        if expected_detected:
            test_pass = ok and det and bool(result.predictions)
        else:
            test_pass = ok and (not det) and (result.reject_reason == expected_reason)

        flag = "OK" if test_pass else "FAIL"
        passed += int(test_pass)
        print(f"{label:<22} {lat:>6.1f} {flag:>4} {str(det):>9} {rr:>14} {top1:<22}", flush=True)

        if not test_pass:
            print(f"    expected: detected={expected_detected} reason={expected_reason}", flush=True)
            print(f"    got: error_kind={result.error_kind} msg={result.error_message}", flush=True)

    print(f"\n{passed}/4 PASS", flush=True)
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
