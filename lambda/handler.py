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
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore.client import Config

# ---------------------------------------------------------------------------
# Logger module-level
# ---------------------------------------------------------------------------
# Level desde env var permite ajustar verbosidad en produccion sin rebuild
# Docker. Setear LAMBDA_LOG_LEVEL=DEBUG via update-function-configuration para
# debugging temporal; vuelve a INFO con otro update-function-configuration.
# JSON log format se configura en la function config (LogFormat=JSON), no aca.
logger = logging.getLogger()
log_level = getenv("LAMBDA_LOG_LEVEL", "INFO")
logger.setLevel(logging.getLevelName(log_level))


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

# Fase 4: DynamoDB para logging predictions + rate limit por fingerprint.
# Reset diario del rate limit viene del sk "DAY#<today>" (key cambia cada
# dia UTC), NO del TTL — TTL es solo cleanup y AWS no garantiza inmediato
# (puede tardar hasta 48h en borrar items expirados).
DYNAMODB_TABLE_NAME = "bird-classifier-py-data"
DAILY_REQUEST_LIMIT = 30  # permite EXACTAMENTE 30 requests/dia (29<30 OK; 30<30 falla)
RATE_LIMIT_TTL_SECONDS = 24 * 3600
PREDICTION_TTL_SECONDS = 90 * 24 * 3600
# Fingerprint pattern estricto: fp_ + 32 hex chars exactos (match uuid4().hex).
# UX-grade rate limiting, no security boundary — atacante motivado rota
# localStorage. Documentar en README.
FINGERPRINT_PATTERN = re.compile(r"^fp_[0-9a-f]{32}$")

# Fase 3: S3 presigned URLs para audios largos (cap 30s del API GW desaparece
# cuando el browser sube directo a S3 y manda solo el s3_key al /predict).
S3_UPLOADS_BUCKET = "conocetuave-py-uploads"
S3_UPLOADS_PREFIX = "uploads/"
PRESIGNED_URL_EXPIRES_S = 600  # 10 min — alcanza para subir, no tanto para abusar

# MIME types soportados. Whitelist contra inputs malformados ("exe", "../").
# Subset estricto de validation.py:ALLOWED_FORMATS — m4a NO esta porque el
# frontend Gradio convierte a wav del lado servidor antes de mandar (Safari
# iOS verificado 2026-05-17, no necesitamos soportar m4a aca).
_CONTENT_TYPE_BY_EXT = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}

# Patron estricto para s3_key. Solo aceptamos keys que MATCHEEN lo que
# nosotros generamos (UUID4 + extension whitelisted). Defense in depth:
# rechaza path traversal, prefix wrong, formatos no soportados ANTES de
# tocar S3. UUID4 con hex lowercase (uuid.uuid4() siempre lo es).
_S3_KEY_PATTERN = re.compile(
    r"^uploads/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(mp3|wav|ogg|flac)$"
)

# Fase 2 - Nivel 2: gating OOD via clasificador nativo de BirdNET.
# Sigmoid puro (no flat_sigmoid con sensitivity). Para gating binario
# "pajaro/no pajaro" son equivalentes — divergen en ranking ENTRE especies,
# que no nos importa (usamos ONNX para ese ranking).
#
# Threshold 0.10 (no 0.5 del paper original): Wood & Kahl 2024 (J. Ornithol.)
# recomienda 0.5-0.7 para detection en monitoreo pasivo (audios largos, fauna
# diversa, FP costosos). Nuestro caso es opuesto: audios cortos enviados por
# usuarios humanos que YA decidieron que es un ave; el costo de FN ("no
# detecte ave" sobre un ave real, peor UX) supera al de FP. Tuneo empirico
# sobre el test set indica 0.10 como sweet spot (ver ADR D11).
BIRDNET_DETECTION_THRESHOLD = 0.10

# Fase 2 - Nivel 2 (capa pre-BirdNET): filtro espectral para rechazar audio
# sinetico (white noise, tono puro) que BirdNET interpreta como ave generica
# y pasa el gate de 0.10. Thresholds derivados empiricamente — ver
# scripts/measure_flatness.py y ADR D11. Margenes amplios contra el peor
# pajaro medido (~7x para flatness, ~2.7x para bandwidth).
FLATNESS_WHITE_NOISE_P95 = 0.30  # p95 white noise = 0.59, max bird = 0.084
WHITE_NOISE_MIN_RMS = 0.001       # excluye silencio puro (rms=0)
FLATNESS_PURE_TONE_MEAN = 0.01    # tono puro = 0; aves tonales tambien ~0 pero las separa bw
BANDWIDTH_PURE_TONE_HZ = 1000.0   # tono = 325 Hz; min bird bw = 2706 Hz (Actitis)

# Filtro non-bird: clases de BirdNET V2.4 que NO son aves. Aplicado dentro de
# `_embed()` antes del max-pool — sin esto, audios sin ave (ruido, voz humana,
# rana) pueden disparar max_sigmoid alto sobre una meta-clase y pasar el
# gate. Lista generada por inspeccion de las 6522 labels de V2.4 (script
# scripts/benchmark_baseline_vs_finetuned.py:build_non_bird_mask). 18 idxs:
# 12 meta-clases (Engine, Noise, Dog, Human-*, etc) + 6 ranas (generos Acris,
# Eleutherodactylus, Hyliola, Lithobates). Si BirdNET actualiza el catalogo,
# regenerar esta lista — los indices son posicionales del .txt de labels.
_NON_BIRD_INDICES = frozenset([
       50,  # Acris crepitans_Northern Cricket Frog
       51,  # Acris gryllus_Southern Cricket Frog
     1949,  # Dog_Dog
     2080,  # Eleutherodactylus planirostris_Greenhouse Frog
     2143,  # Engine_Engine
     2152,  # Environmental_Environmental
     2325,  # Fireworks_Fireworks
     2818,  # Human non-vocal_Human non-vocal
     2819,  # Human vocal_Human vocal
     2820,  # Human whistle_Human whistle
     2847,  # Hyliola regilla_Pacific Chorus Frog
     3240,  # Lithobates catesbeianus_American Bullfrog
     3241,  # Lithobates clamitans_Green Frog
     3242,  # Lithobates palustris_Pickerel Frog
     3243,  # Lithobates sylvaticus_Wood Frog
     3927,  # Noise_Noise
     4862,  # Power tools_Power tools
     5560,  # Siren_Siren
])


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
    # Retenemos classifier_out_idx para Nivel 2 (gating OOD). Hasta Fase 1
    # lo descartabamos porque solo nos importaba el embedding.
    return interp, input_idx, embedding_idx, classifier_out_idx


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

# boto3 S3 client module-level: reuse entre invocations (cold start cost
# solo la primera). Sigv4 + region explicita: sin esto, S3 puede devolver
# 307 Temporary Redirect que el browser no sigue en PUTs presigned
# (caso conocido AWS re:Post).
_S3_CLIENT = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estable para logits de BirdNET.

    BirdNET es multi-label: cada logit es independiente y necesita sigmoid
    individual para que el threshold 0.5 sea interpretable como
    'probabilidad de presencia' (Wood & Kahl 2024, J. Ornithol.).

    Implementacion: casteo a float64 + clip a +-700 antes de exp(). El clip
    es no-op para el rango real de BirdNET (~[-30, +15]) y garantiza que
    no haya overflow para inputs sinteticos extremos (los limites de
    np.exp() en float64 estan en ~+-709). Saturacion semanticamente
    correcta: sigmoid(-1000) = 0, sigmoid(1000) = 1.
    """
    x = np.clip(x.astype(np.float64), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-x))


def _evaluate_detection(max_bird_conf_per_window: np.ndarray) -> tuple[float, bool]:
    """Agrega confidences per-window y decide si hay deteccion.

    Input: vector per-window de max-sigmoid YA filtrado por _NON_BIRD_INDICES
    (las meta-clases y ranas fueron excluidas dentro de ``_embed``).

    Returns (max_birdnet_confidence, detected).

    Agregacion: MAX (no MEAN). BirdNET fue disenado para deteccion en 3 s.
    Un audio "30 s con pajaro 5 s" + MEAN diluiria a ~0.10 (falso negativo);
    MAX captura el peak. Mirrora la naturaleza puntual del modelo.
    """
    max_confidence = float(max_bird_conf_per_window.max())
    return max_confidence, max_confidence >= BIRDNET_DETECTION_THRESHOLD


def _embed(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Devuelve (embedding mean-pool, max_bird_conf_per_window, n_windows).

    Por ventana lee dos tensores ya computados por ``invoke()``:
      - ``emb_idx``: embedding 1024-d (penultima capa).
      - ``cls_idx``: logits del clasificador nativo (~6522 especies BirdNET).

    Aplica filtro bird-only sobre los logits antes del max-pool: las clases
    en ``_NON_BIRD_INDICES`` (ruido, voz humana, ranas, etc.) se setean a
    -inf, asi NUNCA pueden disparar la deteccion. Esto se hace ACA y no en
    ``_evaluate_detection`` porque solo aca tenemos los logits raw — pasar
    el vector completo per-window al evaluador seria duplicar memoria y
    contaminar la signature del decisor.

    Costo extra vs Nivel 1: lectura de ~6522 floats + sigmoid + mask + max
    por ventana. El ``invoke()`` ya computa ambos tensores — no hay forward
    adicional.
    """
    interp, input_idx, emb_idx, cls_idx = _BIRDNET
    windows = _chunk_audio(y).astype(np.float32)
    embs = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    max_bird_conf_per_window = np.empty(len(windows), dtype=np.float64)
    non_bird_idx_arr = np.fromiter(_NON_BIRD_INDICES, dtype=np.int64)
    for k, w in enumerate(windows):
        interp.set_tensor(input_idx, np.expand_dims(w, axis=0))
        interp.invoke()
        embs[k] = interp.get_tensor(emb_idx)[0]
        logits = interp.get_tensor(cls_idx)[0]  # shape: (~6522,)
        probs = _sigmoid(logits)
        probs[non_bird_idx_arr] = -np.inf  # excluir non-bird del max
        max_bird_conf_per_window[k] = probs.max()
    return embs.mean(axis=0), max_bird_conf_per_window, len(windows)


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


def _classify_synthetic(y: np.ndarray) -> str | None:
    """Pre-filter espectral: detecta audio sintetico que BirdNET interpreta
    como ave (white noise plano, tono puro).

    Devuelve:
        "white_noise" si spectral flatness p95 > 0.30 y rms > 0.001
        "pure_tone"   si spectral flatness mean < 0.01 y bandwidth mean < 1000 Hz
        None          si parece audio natural — sigue al gate de BirdNET

    Logica:
      - Flatness (Wiener entropy): ratio gmean/amean del power spectrum.
        Cercano a 1 = ruido blanco. Cercano a 0 = senal armonica
        concentrada (tono puro, vocalizacion clara).
      - Bandwidth: ancho efectivo alrededor del centroide espectral.
        Tonos puros tienen bandwidth chico (~300 Hz); aves tonales
        igual tienen flat bajo pero su bandwidth es 1 orden de magnitud
        mayor por harmonics + transients.
      - RMS energy: para no confundir silencio puro (flatness alta por
        division por cero) con white noise.

    Costo: ~5-10 ms para audio de 5 s. Ahorra ~120-150 ms de BirdNET cuando
    rechaza. Net positivo si los rejects son frecuentes.
    """
    import librosa

    flat = librosa.feature.spectral_flatness(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]

    if float(np.percentile(flat, 95)) > FLATNESS_WHITE_NOISE_P95 \
            and float(rms.mean()) > WHITE_NOISE_MIN_RMS:
        return "white_noise"

    flat_mean = float(flat.mean())
    if flat_mean < FLATNESS_PURE_TONE_MEAN:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)[0]
        if float(bw.mean()) < BANDWIDTH_PURE_TONE_HZ:
            return "pure_tone"

    return None


# ---------------------------------------------------------------------------
# Fase 3: presigned uploads + S3 download
# ---------------------------------------------------------------------------
def _generate_upload_url(ext: str) -> dict:
    """Devuelve presigned PUT URL + metadata para upload directo a S3.

    Args:
        ext: extension del archivo (sin punto). Whitelist estricta:
             mp3, wav, ogg, flac. Normalizamos a lowercase + strip antes
             de validar (rechaza "WAV", " mp3 ", "MP3").

    Returns:
        {"upload_url": str, "s3_key": str, "expires_in": int, "content_type": str}

    Raises:
        ValueError: ext no soportada.

    ContentType se pasa EXPLICITO en Params. S3 rechaza el PUT si el browser
    manda un MIME distinto al firmado — defense in depth contra uploads
    no-deseados (AWS docs recomienda esto explicitamente).
    """
    ext = ext.lower().strip()
    if ext not in _CONTENT_TYPE_BY_EXT:
        raise ValueError(
            f"extension no soportada: {ext!r}. "
            f"Permitidas: {sorted(_CONTENT_TYPE_BY_EXT)}"
        )
    content_type = _CONTENT_TYPE_BY_EXT[ext]
    s3_key = f"{S3_UPLOADS_PREFIX}{uuid.uuid4()}.{ext}"

    upload_url = _S3_CLIENT.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": S3_UPLOADS_BUCKET,
            "Key": s3_key,
            "ContentType": content_type,
        },
        ExpiresIn=PRESIGNED_URL_EXPIRES_S,
        HttpMethod="PUT",
    )

    return {
        "upload_url": upload_url,
        "s3_key": s3_key,
        "expires_in": PRESIGNED_URL_EXPIRES_S,
        "content_type": content_type,
    }


def _read_audio_from_s3(s3_key: str) -> bytes:
    """Lee bytes de un object S3. Valida formato del key ANTES de tocar S3.

    Rechaza:
        - path traversal: "../etc/passwd", "/absolute/path", etc.
        - prefix wrong: "other/file.mp3", "models/x.bin"
        - formato wrong: "uploads/random_name.mp3" (no es UUID4)

    Defense in depth: solo deserializamos keys que MATCHEEN lo que el propio
    Lambda generó (regex en _S3_KEY_PATTERN). Si alguien intenta leer un
    object que por casualidad existe pero no fue generado por nosotros, se
    rechaza antes del S3 call (ahorra request + cierra ataque de
    enumeration).

    Raises:
        ValueError: s3_key no matchea el patron esperado.
        _S3_CLIENT.exceptions.NoSuchKey: object no existe en S3 (expirado o nunca subido).
    """
    if not _S3_KEY_PATTERN.match(s3_key):
        raise ValueError("s3_key inválido: formato no reconocido")
    obj = _S3_CLIENT.get_object(Bucket=S3_UPLOADS_BUCKET, Key=s3_key)
    return obj["Body"].read()


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
    """AWS Lambda entry point — routea por path.

    Rutas (API Gateway HTTP API v2 envia `rawPath`):
        POST /upload-url   -> presigned PUT URL para subir audio a S3
        POST /predict      -> clasifica audio (default si rawPath ausente, para
                              preservar compat con tests directos sin event API GW)

    El refactor a routing (Fase 3) en vez de un solo handler mantiene la
    misma Lambda function ARN — API Gateway routea 2 paths al mismo Lambda
    y nosotros despachamos internamente. Mas barato que 2 Lambdas separadas
    (sin cold start duplicado).
    """
    raw_path = event.get("rawPath") or event.get("path") or ""
    if raw_path.endswith("/upload-url"):
        return _handle_upload_url(event)
    return _handle_predict(event)


def _handle_upload_url(event: dict) -> dict:
    """POST /upload-url -> genera presigned PUT URL para S3.

    Body schema: {"ext": "mp3"}  (mp3 | wav | ogg | flac)
    Response 200: {"upload_url", "s3_key", "expires_in", "content_type"}
    Response 400: ext faltante, no string, o no whitelisted.
    Response 500: generate_presigned_url fallo (IAM, network, etc).
    """
    try:
        body = _parse_body(event)
    except ValueError as e:
        return _response(400, {"error": str(e)})

    ext = body.get("ext")
    if not isinstance(ext, str) or not ext.strip():
        return _response(400, {"error": "body debe tener 'ext' (string: mp3/wav/ogg/flac)"})

    try:
        result = _generate_upload_url(ext)
    except ValueError as e:
        return _response(400, {"error": str(e)})
    except Exception as e:  # noqa: BLE001 — boto3 puede lanzar varias clases
        return _response(500, {"error": f"presigned URL falló: {type(e).__name__}: {e}"})

    return _response(200, result)


def _handle_predict(event: dict) -> dict:
    """POST /predict -> clasifica audio.

    Body schema v1 (audio chico, base64 inline):
        {"audio_b64": "<base64>", "top_k": 3}

    Body schema v2 (Fase 3 — audios largos via S3):
        {"s3_key": "uploads/<uuid4>.<ext>", "top_k": 3}

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

    # Bifurcación audio_b64 / s3_key. Retrocompat: audio_b64 sigue funcionando
    # para clientes que no quieran el round-trip extra de presigned URL.
    if "audio_b64" in body:
        try:
            audio_bytes = base64.b64decode(body["audio_b64"], validate=True)
        except (ValueError, base64.binascii.Error) as e:
            return _response(400, {"error": f"audio_b64 inválido: {e}"})
    elif "s3_key" in body:
        try:
            audio_bytes = _read_audio_from_s3(body["s3_key"])
        except ValueError as e:
            # Key invalido (path traversal, prefix wrong, formato wrong) —
            # NO tocamos S3, defense in depth.
            return _response(400, {"error": str(e)})
        except _S3_CLIENT.exceptions.NoSuchKey:
            return _response(404, {"error": "s3_key no existe o expiró"})
        except Exception as e:  # noqa: BLE001
            return _response(500, {"error": f"S3 read falló: {type(e).__name__}: {e}"})
    else:
        return _response(400, {"error": "body debe tener 'audio_b64' o 's3_key'"})

    try:
        y = _load_audio_bytes(audio_bytes)
    except Exception as e:
        return _response(400, {"error": f"no se pudo decodificar audio: {e}"})

    t0 = time.perf_counter()

    # Capa 1 (pre-BirdNET): rechazo de audio sintetico. Ahorra invoke a BirdNET.
    try:
        synthetic_reason = _classify_synthetic(y)
    except Exception as e:
        return _response(500, {"error": f"pre-filter falló: {type(e).__name__}: {e}"})
    if synthetic_reason is not None:
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return _response(200, {
            "detected": False,
            "reason": synthetic_reason,
            "model_version": MODEL_VERSION,
            "inference_time_ms": round(inference_ms, 2),
        })

    # Capa 2 (BirdNET native gate) + Capa 3 (ONNX classifier).
    try:
        embedding, max_conf_per_window, n_windows = _embed(y)
        max_birdnet_confidence, detected = _evaluate_detection(max_conf_per_window)
        predictions = _classify(embedding, top_k) if detected else None
    except Exception as e:
        return _response(500, {"error": f"inferencia falló: {type(e).__name__}: {e}"})
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # Fase 2 - Nivel 2: gating OOD. Si BirdNET nativo no detecto ave,
    # devolvemos status 200 con detected=false en vez de predictions.
    # max_birdnet_confidence se loguea SIEMPRE (gated y happy path) para
    # tener distribucion empirica y poder tunear el threshold con datos.
    if not detected:
        return _response(200, {
            "detected": False,
            "reason": "not_a_bird",
            "max_birdnet_confidence": round(max_birdnet_confidence, 4),
            "model_version": MODEL_VERSION,
            "n_windows": n_windows,
            "inference_time_ms": round(inference_ms, 2),
        })

    return _response(200, {
        "predictions": predictions,
        "model_version": MODEL_VERSION,
        "n_windows": n_windows,
        "inference_time_ms": round(inference_ms, 2),
        "max_birdnet_confidence": round(max_birdnet_confidence, 4),
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
