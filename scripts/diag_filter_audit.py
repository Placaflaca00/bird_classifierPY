"""Diagnostico Fase 6.6 (paralelo a rollback.yml): cuantos audios del set
test_hard rebotan en el filter anti-troll y por que capa, con scores
intermedios. NO toca thresholds, NO modifica el filtro.

Pipeline:
    1. Para cada audio del set, decodifica con audio_io.decode_audio_bytes
       (mismo decoder que prod). Si falla -> decode_ok=No.
    2. Re-computa inline los scores espectrales (spectral_flatness_p95,
       spectral_flatness_mean, rms_mean, bandwidth_mean) que usa
       _classify_synthetic — necesario para verbose output, no se puede
       extraer de la funcion as-is que devuelve solo el verdict.
    3. Llama _classify_synthetic(y) directamente para confirmar verdict.
    4. Si pasa Capa 1 -> _embed(y) -> _evaluate_detection ->
       max_birdnet_confidence + detected.
    5. Si detected -> _classify(embedding, 1) -> top1 + confidence.

Salida:
    - CSV per-audio: reports/diagnostics/test_hard_audit_<ts>.csv
    - Summary stdout: totales + breakdown por capa.

Uso:
    python scripts/diag_filter_audit.py
    python scripts/diag_filter_audit.py --set-dir data/test_sets/xc_hard
    python scripts/diag_filter_audit.py --limit 20    # quick smoke

NO calls to prod Lambda. Pure local. Para comparar con prod end-to-end ver
diag_filter_audit_prod.py (a escribir despues, paceado por rate limit).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lambda"))

# Imports as-is desde handler.py. NO monkey-patch. Si handler explota al
# importar (modelos faltantes en dev), eso fail-fast y avisa.
import librosa  # noqa: E402

import handler as h  # noqa: E402
from audio_io import decode_audio_bytes  # noqa: E402


def _compute_synth_scores(y: np.ndarray) -> dict:
    """Replica el computo de scores que hace _classify_synthetic, exponiendo
    los intermedios. Mismas llamadas a librosa.feature.* + misma constante
    SAMPLE_RATE de handler. NO toca thresholds.
    """
    flat = librosa.feature.spectral_flatness(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]
    sf_p95 = float(np.percentile(flat, 95))
    sf_mean = float(flat.mean())
    rms_mean = float(rms.mean())
    # bandwidth solo se computa cuando entra al branch pure_tone (optimizacion
    # de handler). Para diagnostico, calculamos siempre para tener la columna.
    bw = librosa.feature.spectral_bandwidth(y=y, sr=h.SAMPLE_RATE)[0]
    bw_mean = float(bw.mean())
    return {
        "sf_p95": sf_p95,
        "sf_mean": sf_mean,
        "rms_mean": rms_mean,
        "bw_mean": bw_mean,
    }


def audit_one(audio_path: Path) -> dict:
    """Pipeline un audio. Devuelve dict con todas las metricas/decisions."""
    rel = audio_path.relative_to(ROOT).as_posix()
    species_dir = audio_path.parent.name
    audio_id = audio_path.stem
    row = {
        "audio_id": audio_id,
        "species_dir": species_dir,
        "rel_path": rel,
        "decode_ok": False,
        "outcome": "decode_fail",
        "layer_reject": "decode",
        "sf_p95": None, "sf_mean": None,
        "rms_mean": None, "bw_mean": None,
        "max_birdnet_confidence": None,
        "n_windows": None,
        "top1_species": None,
        "top1_confidence": None,
        "latency_ms": None,
        "error": None,
    }

    t0 = time.perf_counter()
    try:
        y = decode_audio_bytes(audio_path.read_bytes(), sr=h.SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001 — capturamos todo y reportamos
        row["error"] = f"{type(e).__name__}: {e}"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    row["decode_ok"] = True

    # Capa 1: scores sinteticos + verdict
    try:
        scores = _compute_synth_scores(y)
        row.update(scores)
    except Exception as e:  # noqa: BLE001
        row["outcome"] = "filter_error"
        row["layer_reject"] = "synth_scores"
        row["error"] = f"{type(e).__name__}: {e}"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    synth_verdict = h._classify_synthetic(y)
    if synth_verdict is not None:
        # white_noise o pure_tone
        row["outcome"] = "rejected"
        row["layer_reject"] = synth_verdict
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    # Capa 2: BirdNET gate
    try:
        embedding, max_conf_per_window, n_windows = h._embed(y)
        max_birdnet_conf, detected = h._evaluate_detection(max_conf_per_window)
        row["max_birdnet_confidence"] = round(max_birdnet_conf, 4)
        row["n_windows"] = int(n_windows)
    except Exception as e:  # noqa: BLE001
        row["outcome"] = "embed_error"
        row["layer_reject"] = "birdnet"
        row["error"] = f"{type(e).__name__}: {e}"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    if not detected:
        row["outcome"] = "rejected"
        row["layer_reject"] = "not_a_bird"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    # Capa 3: classifier
    try:
        preds = h._classify(embedding, top_k=1)
        if preds:
            row["top1_species"] = preds[0]["species"]
            row["top1_confidence"] = round(float(preds[0]["confidence"]), 4)
    except Exception as e:  # noqa: BLE001
        row["outcome"] = "classify_error"
        row["error"] = f"{type(e).__name__}: {e}"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return row

    row["outcome"] = "passed"
    row["layer_reject"] = ""
    row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set-dir", type=Path,
        default=ROOT / "data" / "test_sets" / "xc_hard",
        help="Dir raiz con subdirs por especie (default: data/test_sets/xc_hard)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Procesa solo N audios (smoke). Por default todos.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "reports" / "diagnostics",
        help="Dir destino del CSV (default: reports/diagnostics/)",
    )
    args = parser.parse_args()

    if not args.set_dir.exists():
        print(f"! Set dir no existe: {args.set_dir}", file=sys.stderr)
        return 2

    # Glob de audios (mp3/wav/ogg/flac) en TODOS los subdirs de especie
    exts = ("*.mp3", "*.wav", "*.ogg", "*.flac")
    audios = []
    for ext in exts:
        audios.extend(args.set_dir.glob(f"*/{ext}"))
    audios.sort()
    if args.limit:
        audios = audios[: args.limit]
    if not audios:
        print(f"! No hay audios en {args.set_dir}", file=sys.stderr)
        return 2

    print(f"[audit] Procesando {len(audios)} audios de {args.set_dir}...")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = args.out_dir / f"test_hard_audit_{ts}.csv"

    rows = []
    progress_every = max(1, len(audios) // 20)
    for i, ap in enumerate(audios):
        row = audit_one(ap)
        rows.append(row)
        if (i + 1) % progress_every == 0 or i == len(audios) - 1:
            done = i + 1
            print(f"  [{done}/{len(audios)}] {ap.parent.name}/{ap.name} "
                  f"-> {row['outcome']}/{row['layer_reject']}")

    # Escribir CSV
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[audit] CSV escrito: {csv_path}")

    # Resumen
    total = len(rows)
    by_outcome = Counter(r["outcome"] for r in rows)
    by_layer = Counter(r["layer_reject"] for r in rows if r["outcome"] == "rejected")

    def pct(n):
        return f"{100 * n / total:.1f}%" if total else "0%"

    print()
    print("=" * 70)
    print(f"RESUMEN — set: {args.set_dir.relative_to(ROOT)}")
    print(f"  total audios:                {total}")
    print(f"  decode_fail:                 {by_outcome.get('decode_fail', 0):3d}  ({pct(by_outcome.get('decode_fail', 0))})")
    rejected_total = by_outcome.get("rejected", 0)
    print(f"  filter_reject TOTAL:         {rejected_total:3d}  ({pct(rejected_total)})")
    print(f"    └ white_noise:             {by_layer.get('white_noise', 0):3d}  ({pct(by_layer.get('white_noise', 0))})")
    print(f"    └ pure_tone:               {by_layer.get('pure_tone', 0):3d}  ({pct(by_layer.get('pure_tone', 0))})")
    print(f"    └ not_a_bird (BirdNET):    {by_layer.get('not_a_bird', 0):3d}  ({pct(by_layer.get('not_a_bird', 0))})")
    print(f"  passed (llego al MLP):       {by_outcome.get('passed', 0):3d}  ({pct(by_outcome.get('passed', 0))})")
    other = total - sum([
        by_outcome.get("decode_fail", 0),
        by_outcome.get("rejected", 0),
        by_outcome.get("passed", 0),
    ])
    if other:
        print(f"  errores (embed/classify):    {other:3d}  ({pct(other)})")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
