"""Entry point de la función AWS Lambda — clasificación de aves por audio.

Pipeline por request:
    1. Decodificar audio (base64 en body, o presigned S3 key en v2 — bifurcado).
    2. Cargar mono 48 kHz (librosa).
    3. Trocear en ventanas de 3 s (144000 samples), padear última con ceros.
    4. BirdNET V2.4 TFLite -> embedding 1024-dim por ventana -> mean-pool.
    5. ONNX classifier -> logits -> softmax -> top-K.
    6. Devolver JSON con predictions, model_version, n_windows, inference_time_ms.

Cold-start: BirdNET y ONNX se cargan a módulo (fuera del handler), se reutilizan
entre invocaciones de la misma instancia Lambda.

Test local:
    python lambda/handler.py data/test_sets/xc_hard/ortalis_canicollis/XC343334.mp3
    python lambda/handler.py <audio.mp3> --top-k 5
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constantes y rutas
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"  # bakeado en la imagen Docker (/var/task/lambda/models)
CLASSIFIER_ONNX = MODELS_DIR / "classifier.onnx"
CLASSIFIER_META = MODELS_DIR / "classifier.json"
BIRDNET_TFLITE = MODELS_DIR / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
SPECIES_META = HERE / "species_metadata.json"

# Para testing local: si los archivos no están en lambda/models/, se buscan en
# las rutas de desarrollo (repo root y birdnetlib).
ROOT = HERE.parent
DEV_CLASSIFIER_ONNX = ROOT / "models" / "classifier.onnx"
DEV_CLASSIFIER_META = ROOT / "models" / "classifier.json"

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 144_000  # 3 s @ 48 kHz
EMBEDDING_DIM = 1024
MODEL_VERSION = "classifier_v1"  # bump en cada retrain; documentar en ADR

DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Resolución de paths (Lambda image vs dev local)
# ---------------------------------------------------------------------------
def _resolve_classifier_paths() -> tuple[Path, Path]:
    """Devuelve (onnx_path, meta_path). Prioriza /var/task/lambda/models, fallback dev."""
    if CLASSIFIER_ONNX.exists() and CLASSIFIER_META.exists():
        return CLASSIFIER_ONNX, CLASSIFIER_META
    if DEV_CLASSIFIER_ONNX.exists() and DEV_CLASSIFIER_META.exists():
        return DEV_CLASSIFIER_ONNX, DEV_CLASSIFIER_META
    raise FileNotFoundError(
        f"classifier.onnx/.json no encontrado en {MODELS_DIR} ni en {ROOT/'models'}"
    )


def _resolve_birdnet_path() -> Path:
    """BirdNET TFLite: lambda/models/ en imagen, o el de birdnetlib en dev."""
    if BIRDNET_TFLITE.exists():
        return BIRDNET_TFLITE
    try:
        from birdnetlib.analyzer import Analyzer

        return Path(Analyzer().model_path)
    except ImportError as e:
        raise FileNotFoundError(
            f"BirdNET TFLite no está en {BIRDNET_TFLITE} y birdnetlib no está instalado."
        ) from e


# ---------------------------------------------------------------------------
# Carga module-level (cold start)
# ---------------------------------------------------------------------------
def _load_birdnet():
    """Crea TFLite interpreter con `experimental_preserve_all_tensors=True` para
    poder leer la penúltima capa (embedding 1024-d). Mismo patrón que
    scripts/precompute_embeddings.py.
    """
    # Prioridad: LiteRT (producción, GA desde TF 2.21) -> tflite_runtime (legacy)
    # -> tensorflow.lite (fallback dev local vía birdnetlib).
    try:
        from ai_edge_litert.interpreter import Interpreter  # producción Lambda
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # runtime legacy
        except ImportError:
            import tensorflow as tf  # fallback dev local
            Interpreter = tf.lite.Interpreter

    path = _resolve_birdnet_path()
    interp = Interpreter(
        model_path=str(path),
        experimental_preserve_all_tensors=True,
    )
    interp.allocate_tensors()
    input_idx = interp.get_input_details()[0]["index"]
    classifier_out_idx = interp.get_output_details()[0]["index"]
    embedding_idx = classifier_out_idx - 1  # penúltima capa
    return interp, input_idx, embedding_idx


def _load_classifier():
    import onnxruntime as ort

    onnx_path, meta_path = _resolve_classifier_paths()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    idx_to_species = {int(k): v for k, v in meta["idx_to_species"].items()}
    return sess, idx_to_species, meta


def _load_species_metadata() -> dict[str, dict[str, str]]:
    """Devuelve {species: {"es": ..., "en": ...}}. Vacío si el archivo no existe."""
    if not SPECIES_META.exists():
        return {}
    with open(SPECIES_META, encoding="utf-8") as f:
        return json.load(f).get("species", {})


# Inicialización module-level. En Lambda corre durante init phase (antes del
# primer invoke). En tests locales corre al importar el módulo. Si falla,
# preferimos crash temprano antes que fallar en runtime.
_BIRDNET = _load_birdnet()
_CLASSIFIER_SESS, _IDX_TO_SPECIES, _CLASSIFIER_META = _load_classifier()
_SPECIES_META = _load_species_metadata()
_NUM_CLASSES = len(_IDX_TO_SPECIES)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
def _chunk_audio(y: np.ndarray) -> np.ndarray:
    """Divide y en ventanas de WINDOW_SAMPLES. Última se padea con ceros."""
    n = len(y)
    if n < WINDOW_SAMPLES:
        return np.pad(y, (0, WINDOW_SAMPLES - n))[np.newaxis, :]
    n_windows = (n + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    total = n_windows * WINDOW_SAMPLES
    if total > n:
        y = np.pad(y, (0, total - n))
    return y[: n_windows * WINDOW_SAMPLES].reshape(n_windows, WINDOW_SAMPLES)


def _embed(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Embedding 1024-d (mean-pool sobre ventanas) y n_windows usadas."""
    interp, input_idx, emb_idx = _BIRDNET
    windows = _chunk_audio(y).astype(np.float32)
    embs = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    for k, w in enumerate(windows):
        interp.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interp.invoke()
        embs[k] = interp.get_tensor(emb_idx)[0]
    return embs.mean(axis=0), len(windows)


def _classify(embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    """Embedding 1024-d -> top-K predicciones con softmax."""
    x = embedding.astype(np.float32).reshape(1, EMBEDDING_DIM)
    logits = _CLASSIFIER_SESS.run(None, {"embedding": x})[0][0]
    # softmax estable
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top_idx = np.argsort(probs)[::-1][:top_k]
    out = []
    for i in top_idx:
        species = _IDX_TO_SPECIES[int(i)]
        names = _SPECIES_META.get(species, {})
        out.append({
            "species": species,
            "common_name": names.get("es"),
            "common_name_en": names.get("en"),
            "confidence": float(probs[int(i)]),
        })
    return out


def _load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Bytes -> waveform mono 48 kHz."""
    import librosa

    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError("audio vacío después de decodificar")
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def _parse_body(event: dict) -> dict:
    """Soporta API Gateway (body string) y test directo (dict)."""
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"body no es JSON válido: {e}") from e
    if not isinstance(body, dict):
        raise ValueError("body debe ser un objeto JSON")
    return body


def _response(status: int, payload: dict) -> dict:
    """Respuesta en formato API Gateway."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def handler(event: dict, context: Any = None) -> dict:
    """AWS Lambda entry point.

    Body schema v1 (testing + primer deploy):
        {"audio_b64": "<base64>", "top_k": 3}

    Body schema v2 (cuando los audios pasen 4-5 MB) — bifurcación lista, no implementada:
        {"s3_key": "uploads/abc.mp3", "top_k": 3}

    Top-k clip a [1, num_classes]. Default 3.
    """
    try:
        body = _parse_body(event)
    except ValueError as e:
        return _response(400, {"error": str(e)})

    top_k_raw = body.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = max(1, min(int(top_k_raw), _NUM_CLASSES))
    except (TypeError, ValueError):
        return _response(400, {"error": f"top_k inválido: {top_k_raw!r}"})

    # Bifurcación audio_b64 / s3_key
    if "audio_b64" in body:
        try:
            audio_bytes = base64.b64decode(body["audio_b64"], validate=True)
        except (ValueError, base64.binascii.Error) as e:
            return _response(400, {"error": f"audio_b64 inválido: {e}"})
    elif "s3_key" in body:
        # v2: presigned S3 download. Cuando se implemente: boto3 + IAM s3:GetObject
        # sobre el bucket S3_FEEDBACK_BUCKET / S3_UPLOADS_BUCKET (a definir).
        return _response(501, {
            "error": "s3_key path no implementado todavía. Usá audio_b64 en v1.",
        })
    else:
        return _response(400, {"error": "body debe tener 'audio_b64' o 's3_key'"})

    try:
        y = _load_audio_bytes(audio_bytes)
    except Exception as e:
        return _response(400, {"error": f"no se pudo decodificar audio: {e}"})

    t0 = time.perf_counter()
    try:
        embedding, n_windows = _embed(y)
        predictions = _classify(embedding, top_k)
    except Exception as e:
        return _response(500, {"error": f"inferencia falló: {type(e).__name__}: {e}"})
    inference_ms = (time.perf_counter() - t0) * 1000.0

    return _response(200, {
        "predictions": predictions,
        "model_version": MODEL_VERSION,
        "n_windows": n_windows,
        "inference_time_ms": round(inference_ms, 2),
    })


# ---------------------------------------------------------------------------
# Test local (CLI)
# ---------------------------------------------------------------------------
def _local_main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test local del handler de Lambda.")
    parser.add_argument("audio", type=Path, help="Path a un mp3/wav.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"audio no existe: {args.audio}")
        return 2

    audio_bytes = args.audio.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    event = {"body": json.dumps({"audio_b64": audio_b64, "top_k": args.top_k})}

    t0 = time.perf_counter()
    resp = handler(event, None)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nstatus={resp['statusCode']}  total_ms={total_ms:.1f}")
    print(json.dumps(json.loads(resp["body"]), indent=2, ensure_ascii=False))
    return 0 if resp["statusCode"] == 200 else 1


if __name__ == "__main__":
    raise SystemExit(_local_main())
