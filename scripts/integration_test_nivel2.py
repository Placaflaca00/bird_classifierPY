"""Integration test Fase 2 - Nivel 2: gating BirdNET + threshold 0.10 + bird-only filter.

Corre el handler completo (audio_b64 -> JSON response) sobre 3+4 audios:
    3 bird (esperamos detected=True, predictions presentes):
        - easy:   Calidris canutus de test_clean (baseline ~0.997)
        - medium: Chauna torquata de xc_hard (baseline ~0.205)
        - border: Actitis macularius de xc_hard (baseline ~0.337)
    4 non-bird (esperamos detected=False):
        - silencio 5s @ 48 kHz (sintetico)
        - white noise 5s (sintetico, low amplitude)
        - tono puro 440 Hz 5s (sintetico)
        - voz humana real (regression case del bug original)

Bonus opcional (no cuenta en el pass/fail):
    - audio de ambiente real

Genera los fixtures sinteticos en tests/fixtures/ si no existen (idempotente).
NO destructivo: ningun overwrite a archivos pre-existentes con el mismo nombre.

Uso:
    python -u scripts/integration_test_nivel2.py
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
RAW_DIR = ROOT / "data" / "raw"
TEST_HARD_DIR = ROOT / "data" / "test_sets" / "xc_hard"

SR = 48_000


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    """Escribe WAV mono PCM16 sin dependencia externa.

    Sintetiza el header WAV a mano para evitar agregar soundfile/scipy a deps.
    samples: float32 en [-1, 1].
    """
    import struct

    samples_i16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    data_bytes = samples_i16.tobytes()
    n_samples = len(samples_i16)
    byte_rate = sr * 1 * 2  # mono, 16-bit
    block_align = 1 * 2

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data_bytes)))  # chunk size
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # subchunk1 size
        f.write(struct.pack("<H", 1))            # PCM
        f.write(struct.pack("<H", 1))            # mono
        f.write(struct.pack("<I", sr))           # sample rate
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 16))           # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)


def ensure_synthetic_fixtures() -> dict[str, Path]:
    """Genera los 3 sinteticos en tests/fixtures/ si no existen. Idempotente."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    duration_s = 5.0
    n = int(duration_s * SR)
    out: dict[str, Path] = {}

    # 1. Silencio absoluto
    silence_path = FIXTURES / "silence_5s.wav"
    if not silence_path.exists():
        write_wav(silence_path, np.zeros(n, dtype=np.float32))
        print(f"  generado: {silence_path.name}", flush=True)
    out["silence_5s"] = silence_path

    # 2. White noise (amplitud baja para no parecer pajaro)
    noise_path = FIXTURES / "white_noise_5s.wav"
    if not noise_path.exists():
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, 0.05, n).astype(np.float32)  # ~ -26 dBFS
        write_wav(noise_path, noise)
        print(f"  generado: {noise_path.name}", flush=True)
    out["white_noise_5s"] = noise_path

    # 3. Tono puro 440 Hz (no aviar)
    sine_path = FIXTURES / "sine_440hz_5s.wav"
    if not sine_path.exists():
        t = np.arange(n) / SR
        sine = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        write_wav(sine_path, sine)
        print(f"  generado: {sine_path.name}", flush=True)
    out["sine_440hz_5s"] = sine_path

    return out


def load_handler():
    spec = importlib.util.spec_from_file_location("lambda_handler", ROOT / "lambda" / "handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_handler(handler_mod, audio_path: Path, top_k: int = 3) -> dict:
    """Llama handler.handler() con event format API Gateway."""
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    event = {"body": json.dumps({"audio_b64": audio_b64, "top_k": top_k})}
    resp = handler_mod.handler(event, None)
    body = json.loads(resp["body"])
    body["_status"] = resp["statusCode"]
    return body


def fmt_pred(predictions: list[dict] | None) -> str:
    if not predictions:
        return "—"
    return ", ".join(f"{p['species']}({p['confidence']:.2f})" for p in predictions[:2])


def main() -> int:
    print(f"[1/3] Generando fixtures sinteticos en {FIXTURES.relative_to(ROOT)}...", flush=True)
    synth = ensure_synthetic_fixtures()
    print(f"  {len(synth)} sinteticos listos.\n", flush=True)

    print("[2/3] Cargando handler (BirdNET + ONNX, ~5s)...", flush=True)
    h = load_handler()
    print(f"  threshold={h.BIRDNET_DETECTION_THRESHOLD}, non-bird filtered={len(h._NON_BIRD_INDICES)}\n", flush=True)

    cases = [
        # (label, abs_path, kind, expected_detected)
        ("bird-easy: Calidris canutus (clean)", RAW_DIR / "calidris_canutus" / "GBIF1975781150.mp3", "bird", True),
        ("bird-medium: Chauna torquata (hard)", TEST_HARD_DIR / "chauna_torquata" / "XC844833.mp3", "bird", True),
        ("bird-border: Actitis macularius (hard)", TEST_HARD_DIR / "actitis_macularius" / "XC667193.mp3", "bird", True),
        ("non-bird: silencio 5s",                   synth["silence_5s"],     "non-bird", False),
        ("non-bird: white noise 5s",                synth["white_noise_5s"], "non-bird", False),
        ("non-bird: tono 440 Hz 5s",                synth["sine_440hz_5s"],  "non-bird", False),
        ("non-bird: voz humana 6s (regression)",    FIXTURES / "voz_humana_6s.wav", "non-bird", False),
        # Bonus (no cuenta en pass/fail)
        ("non-bird BONUS: ambiente (regression)",   FIXTURES / "ambiente.wav", "bonus", False),
    ]

    print("[3/3] Ejecutando handler sobre cada caso...\n", flush=True)
    results = []
    for label, path, kind, expected_detected in cases:
        if not path.exists():
            print(f"  SKIP  {label}: {path} no existe", flush=True)
            continue
        try:
            resp = call_handler(h, path, top_k=3)
        except Exception as e:
            print(f"  ERROR {label}: {type(e).__name__}: {e}", flush=True)
            continue
        detected = resp.get("detected", True) if "detected" in resp else (resp.get("predictions") is not None)
        max_conf = resp.get("max_birdnet_confidence")
        preds = resp.get("predictions")
        pass_fail = (detected == expected_detected)
        results.append({
            "label": label, "kind": kind, "detected": detected,
            "expected": expected_detected, "max_conf": max_conf,
            "predictions": preds, "pass": pass_fail, "status": resp.get("_status"),
            "inference_ms": resp.get("inference_time_ms"),
            "n_windows": resp.get("n_windows"),
        })

    # Tabla
    print("\n" + "=" * 100, flush=True)
    print(f"{'caso':<48} {'kind':<10} {'detected':<9} {'max_conf':<10} {'top-1 pred':<28} {'PASS':<5}", flush=True)
    print("-" * 100, flush=True)
    for r in results:
        det = "True" if r["detected"] else "False"
        conf = f"{r['max_conf']:.4f}" if r["max_conf"] is not None else "—"
        pred = fmt_pred(r["predictions"])
        ok = "OK" if r["pass"] else "FAIL"
        if r["kind"] == "bonus":
            ok = ok + "*"
        print(f"{r['label']:<48} {r['kind']:<10} {det:<9} {conf:<10} {pred:<28} {ok:<5}", flush=True)
    print("=" * 100, flush=True)

    # Summary excluyendo bonus
    core = [r for r in results if r["kind"] != "bonus"]
    passed = sum(1 for r in core if r["pass"])
    total = len(core)
    print(f"\nCore: {passed}/{total} PASS  (bonus marked with *, no cuenta)", flush=True)

    # Latencia
    times = [r["inference_ms"] for r in results if r["inference_ms"] is not None]
    if times:
        print(f"Inference time: min={min(times):.0f}ms  median={np.median(times):.0f}ms  max={max(times):.0f}ms", flush=True)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
