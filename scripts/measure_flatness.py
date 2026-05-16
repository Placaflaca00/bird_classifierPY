"""Mide spectral flatness + spectral bandwidth sobre los audios del integration test.

Objetivo: derivar empiricamente thresholds para el filtro pre-BirdNET que rechaza
adversarial sintetico (white noise, tono puro) sin tocar pajaros reales.

Spectral flatness (Wiener entropy): ratio geometric mean / arithmetic mean del
espectro. Cercano a 1 = ruido blanco (espectro plano). Cercano a 0 = tono puro
o senal armonica concentrada. Pajaros reales caen en medio con variacion.

Spectral bandwidth: ancho efectivo del espectro alrededor del centroide.
Tonos puros: bandwidth muy chico. Senales con ruido o multi-banda: alto.

NO inventa thresholds. Solo reporta numeros — la decision viene despues.

Uso:
    python -u scripts/measure_flatness.py
"""
from __future__ import annotations

import os
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

CASES = [
    # (label, kind, path)
    ("bird-easy:   Calidris canutus (clean)",   "bird",     RAW_DIR / "calidris_canutus" / "GBIF1975781150.mp3"),
    ("bird-medium: Chauna torquata  (hard)",    "bird",     TEST_HARD_DIR / "chauna_torquata" / "XC844833.mp3"),
    ("bird-border: Actitis macularius (hard)",  "bird",     TEST_HARD_DIR / "actitis_macularius" / "XC667193.mp3"),
    ("non-bird:    silencio 5s",                 "non-bird", FIXTURES / "silence_5s.wav"),
    ("non-bird:    white noise 5s",              "non-bird", FIXTURES / "white_noise_5s.wav"),
    ("non-bird:    tono 440 Hz 5s",              "non-bird", FIXTURES / "sine_440hz_5s.wav"),
    ("non-bird:    voz humana 6s",               "non-bird", FIXTURES / "voz_humana_6s.wav"),
    ("non-bird:    ambiente",                    "non-bird", FIXTURES / "ambiente.wav"),
]


def measure(audio_path: Path) -> dict:
    """Carga audio mono 48 kHz y mide flatness + bandwidth + RMS."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    if len(y) == 0:
        return {"error": "audio vacio"}

    # Spectral flatness: salida shape (1, n_frames). Valores en [0, 1].
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    # Spectral bandwidth: shape (1, n_frames). Hz.
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    # RMS energy: shape (1, n_frames). Para distinguir silencio puro.
    rms = librosa.feature.rms(y=y)[0]

    return {
        "duration_s": len(y) / SR,
        "n_frames": len(flatness),
        "flatness_mean": float(np.mean(flatness)),
        "flatness_max": float(np.max(flatness)),
        "flatness_min": float(np.min(flatness)),
        "flatness_std": float(np.std(flatness)),
        "flatness_p95": float(np.percentile(flatness, 95)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_p5": float(np.percentile(bandwidth, 5)),
        "bandwidth_p95": float(np.percentile(bandwidth, 95)),
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
    }


def main() -> int:
    print("Cargando librosa (esto demora ~3s)...\n", flush=True)

    rows = []
    for label, kind, path in CASES:
        if not path.exists():
            print(f"  SKIP {label}: {path} no existe", flush=True)
            continue
        try:
            m = measure(path)
            rows.append((label, kind, m))
        except Exception as e:
            print(f"  ERROR {label}: {type(e).__name__}: {e}", flush=True)

    if not rows:
        return 1

    # Tabla principal: flatness stats
    print("=" * 130, flush=True)
    print(f"{'caso':<44} {'kind':<10} {'flat_mean':>10} {'flat_max':>10} {'flat_p95':>10} {'flat_std':>10} {'bw_mean(Hz)':>12} {'rms_mean':>10}", flush=True)
    print("-" * 130, flush=True)
    for label, kind, m in rows:
        print(
            f"{label:<44} {kind:<10} "
            f"{m['flatness_mean']:>10.4f} {m['flatness_max']:>10.4f} {m['flatness_p95']:>10.4f} "
            f"{m['flatness_std']:>10.4f} {m['bandwidth_mean']:>12.1f} {m['rms_mean']:>10.4f}",
            flush=True,
        )
    print("=" * 130, flush=True)

    # Resumen por kind: rangos
    print("\nRangos por categoria:\n", flush=True)
    for kind in ["bird", "non-bird"]:
        sub = [m for _, k, m in rows if k == kind]
        if not sub:
            continue
        fl_means = [m["flatness_mean"] for m in sub]
        fl_maxes = [m["flatness_max"] for m in sub]
        bws = [m["bandwidth_mean"] for m in sub]
        print(f"{kind}: n={len(sub)}", flush=True)
        print(f"  flatness_mean: range [{min(fl_means):.4f}, {max(fl_means):.4f}]", flush=True)
        print(f"  flatness_max:  range [{min(fl_maxes):.4f}, {max(fl_maxes):.4f}]", flush=True)
        print(f"  bandwidth_mean: range [{min(bws):.1f}, {max(bws):.1f}] Hz", flush=True)

    # Diagnostico de separabilidad
    print("\nSeparabilidad (bird vs non-bird) por feature:", flush=True)
    bird_rows = [m for _, k, m in rows if k == "bird"]
    nb_rows = [m for _, k, m in rows if k == "non-bird"]
    for feat in ["flatness_mean", "flatness_max", "flatness_p95", "bandwidth_mean"]:
        bird_max = max(m[feat] for m in bird_rows)
        bird_min = min(m[feat] for m in bird_rows)
        nb_max = max(m[feat] for m in nb_rows)
        nb_min = min(m[feat] for m in nb_rows)
        overlap = not (bird_max < nb_min or nb_max < bird_min)
        print(f"  {feat:<18} bird [{bird_min:.4f}, {bird_max:.4f}]  non-bird [{nb_min:.4f}, {nb_max:.4f}]  {'OVERLAP' if overlap else 'SEPARABLE'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
