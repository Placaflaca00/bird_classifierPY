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
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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


# ---------------------------------------------------------------------------
# Fase 5A — warm pipeline (anti cold start)
# ---------------------------------------------------------------------------
# Generamos un WAV dummy una vez al cargar el modulo: 1 s de "noise blanco
# suave" @ 22050 Hz mono PCM16. Lo usa `_warm_pipeline()` cuando el handler
# recibe {"warm": true} (EventBridge cada 7 min + demo.load del frontend).
#
# Sample rate 22050 (NO 48000) a proposito: librosa.load(sr=48000) resamplea
# del dummy a la SR target, lo que dispara la JIT compilation de resampy/soxr
# (la ruta resample-numba es de las mas caras del cold start efectivo). Si
# usaramos 48000 ya, librosa skipearia el resample y dejariamos esa ruta cold.
#
# Amplitude baja (~2% del rango int16): suficiente para que librosa.load
# devuelva un vector no-trivial; no nos importa que el filtro espectral lo
# rechazaria como "white_noise" porque en el warm path llamamos las funciones
# de librosa.feature.* DIRECTAMENTE (no via _classify_synthetic), ignorando
# el output. Lo que warmamos es el JIT, no la decision.
def _make_dummy_wav_bytes() -> bytes:
    """1s noise @ 22050Hz mono PCM16 WAV para warmup. Determinista (seed=42)."""
    import struct as _struct  # local: solo se usa al cargar el modulo

    n_samples = 22050
    rng = np.random.default_rng(seed=42)
    samples_int16 = (rng.standard_normal(n_samples) * 800).astype(np.int16)
    audio = samples_int16.tobytes()
    header = (
        b"RIFF" + _struct.pack("<I", 36 + len(audio))
        + b"WAVE" + b"fmt " + _struct.pack("<I", 16)
        + _struct.pack("<HHIIHH", 1, 1, 22050, 22050 * 2, 2, 16)
        + b"data" + _struct.pack("<I", len(audio))
    )
    return header + audio


_DUMMY_WAV_BYTES = _make_dummy_wav_bytes()


def _warm_pipeline() -> None:
    """Ejecuta el pipeline completo con audio dummy — warmer post-init.

    Cubre las rutas con costo de "primera ejecucion" que NO se warman solo con
    init module-level:
      1. ``import librosa`` (lazy en _load_audio_bytes y _classify_synthetic).
      2. ``librosa.load`` con resample 22050->48000 (resampy/soxr numba JIT).
      3. ``librosa.feature.spectral_flatness`` / ``rms`` / ``spectral_bandwidth``
         — las 3 son numba-JIT y suman ~5-15 s la primera vez. Las llamamos
         directamente (no via ``_classify_synthetic``) porque ese helper
         puede saltarse ``spectral_bandwidth`` cuando flat_mean es alto, y
         queremos garantizar JIT compile de las 3.
      4. ``_embed`` ejecuta el TFLite interpreter con un tensor real (el init
         del module solo hace allocate_tensors, no invoca).
      5. ``_classify`` ejecuta la ONNX session por primera vez (graph
         optimization se completa, kernel cache se llena).

    NO toca DynamoDB ni S3 — branch en el handler garantiza que esta funcion
    solo corre cuando ``event["warm"] is True``, sin pasar por validation ni
    rate limit ni write paths.

    Costo medido (post-init, warm): ~150-250 ms. Costo en cold con todos los
    JIT pendientes: ~5-15 s. El cold corre dentro del init-in-handler del
    primer warm invoke; el resto son barato.
    """
    import librosa  # lazy en el path normal, lo forzamos aca para warmar

    y, _ = librosa.load(
        io.BytesIO(_DUMMY_WAV_BYTES), sr=SAMPLE_RATE, mono=True,
    )
    librosa.feature.spectral_flatness(y=y)
    librosa.feature.rms(y=y)
    librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    embedding, _, _ = _embed(y)
    _classify(embedding, top_k=DEFAULT_TOP_K)

# Fase 4c — constantes de validacion del endpoint /feedback.
# _FEEDBACK_ACTIONS: discriminator del body (patron OpenAPI oneOf); cada valor
#   define que campos extra son validos (ver _handle_feedback).
# _VALID_SPECIES: especies aceptables como corrected_species. Derivado del
#   modelo cargado — NO hardcodear: si el classifier se reentrena con otro set
#   de clases, esto lo sigue automaticamente.
_FEEDBACK_ACTIONS = frozenset({"confirmed", "corrected", "rejected_as_non_bird"})
_VALID_SPECIES = frozenset(_IDX_TO_SPECIES.values())

# boto3 S3 client module-level: reuse entre invocations (cold start cost
# solo la primera). Sigv4 + region explicita: sin esto, S3 puede devolver
# 307 Temporary Redirect que el browser no sigue en PUTs presigned
# (caso conocido AWS re:Post).
_S3_CLIENT = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)

# DynamoDB resource module-level: reuse entre invocations evita ~50-150 ms
# de init en cold start. Resource (no client) porque ofrece mejor DX para
# CRUD comun: sin {"S": ...}/{"N": ...} DynamoDB-JSON manual boilerplate,
# tipos Python nativos. Performance overhead vs client es negligible
# (~10 ms, irrelevante contra los 150ms de la inferencia ML).
#
# Config con adaptive retry: maneja throttling de DynamoDB mejor que
# defaults. connect_timeout/read_timeout previene Lambdas que se cuelgan
# esperando responses de DDB.
#
# Sin Config(signature_version="s3v4") — DynamoDB no firma presigned URLs,
# usa el default boto3. Agregarlo confundiría a un lector futuro asumiendo
# que hay una razon especial.
_DDB_RESOURCE = boto3.resource(
    "dynamodb",
    region_name="us-east-1",
    config=Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=10,
    ),
)
_DDB_TABLE = _DDB_RESOURCE.Table(DYNAMODB_TABLE_NAME)


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
# Fase 4: DynamoDB writes (rate limit + prediction logging)
# ---------------------------------------------------------------------------
def _check_and_increment_rate_limit(fingerprint: str) -> tuple[bool, dict]:
    """Rate limit atomico server-side por fingerprint via UpdateItem condicional.

    Devuelve (passed, info):

    passed=True (request OK, count incrementado):
        info = {
            "requests_today": <count tras este request, 1..DAILY_REQUEST_LIMIT>,
            "limit": DAILY_REQUEST_LIMIT,
            "remaining": DAILY_REQUEST_LIMIT - requests_today,
        }

    passed=False (limite excedido, sin incremento):
        info = {
            "requests_today": <count actual, == DAILY_REQUEST_LIMIT>,
            "limit": DAILY_REQUEST_LIMIT,
            "remaining": 0,
            "reset_at": "YYYY-MM-DDT00:00:00Z",   # proxima medianoche UTC
        }

    fail-open (DDB unavailable, error != ConditionalCheckFailed):
        info = {"degraded": True, "limit": DAILY_REQUEST_LIMIT}

    Atomico server-side. La ConditionExpression `request_count < :limit` permite
    EXACTAMENTE DAILY_REQUEST_LIMIT requests/dia (29<30 OK -> count=30;
    30<30 falla -> 429). Off-by-one consciente, documentado en la constante.

    Reset diario viene del sk DAY#<today_utc> (key cambia cada dia UTC), NO
    del TTL. UTC end-to-end elimina ambiguedad cross-timezone — un user a las
    23:30 ART (02:30 UTC del dia siguiente) ve reset_at consistente con el
    sort key que el backend acaba de usar.

    Truco para devolver `requests_today` en el reject sin GetItem extra:
    ReturnValuesOnConditionCheckFailure="ALL_OLD" hace que DDB incluya el
    item existente en el error response cuando la condition falla. Misma
    operacion, info gratis (botocore >=1.31.55; pinneado en requirements.txt
    a 1.43.9 para parity tests-local <-> Docker).

    Fail-open: errores != ConditionalCheckFailed dejan pasar la request
    (log WARN). Rate limit es proteccion anti-abuso, no fraud prevention.
    Mejor servir legitimas que cortar por un blip DDB.

    Aliasing TODOS los attribute names (best practice defensiva): ttl y
    date son reserved keywords; count/fp/itype no — pero aliasing uniforme
    evita bugs si renombramos atributos al futuro.
    """
    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    ttl_unix = int(time.time()) + RATE_LIMIT_TTL_SECONDS
    try:
        response = _DDB_TABLE.update_item(
            Key={"pk": f"RATE#{fingerprint}", "sk": f"DAY#{today}"},
            UpdateExpression=(
                "ADD #count :one "
                "SET #ttl = if_not_exists(#ttl, :ttl_val), "
                "#fp = :fp, #date = :date, #itype = :itype"
            ),
            ConditionExpression="attribute_not_exists(#count) OR #count < :limit",
            ExpressionAttributeNames={
                "#count": "request_count",
                "#ttl": "ttl",
                "#fp": "fingerprint",
                "#date": "date",
                "#itype": "item_type",
            },
            ExpressionAttributeValues={
                ":one": 1,
                ":limit": DAILY_REQUEST_LIMIT,
                ":ttl_val": ttl_unix,
                ":fp": fingerprint,
                ":date": today,
                ":itype": "RATE_LIMIT",
            },
            ReturnValues="UPDATED_NEW",
            ReturnValuesOnConditionCheckFailure="ALL_OLD",
        )
        new_count = int(response["Attributes"]["request_count"])
        return True, {
            "requests_today": new_count,
            "limit": DAILY_REQUEST_LIMIT,
            "remaining": DAILY_REQUEST_LIMIT - new_count,
        }
    except _DDB_TABLE.meta.client.exceptions.ConditionalCheckFailedException as e:
        # DDB devuelve el item viejo en e.response["Item"] gracias a
        # ReturnValuesOnConditionCheckFailure="ALL_OLD" — pero en DynamoDB-JSON
        # CRUDO: los Number vienen como {"N": "<str>"}, NO int/Decimal. El
        # deserializer del resource API solo aplica al output shape de la
        # operacion, no al Item de este error. Verificado empiricamente
        # (probe contra la tabla real, 2026-05-20).
        existing = e.response.get("Item", {})
        raw_count = existing.get("request_count", {})
        current_count = int(raw_count.get("N", DAILY_REQUEST_LIMIT))
        reset_at = (now_utc + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        return False, {
            "requests_today": current_count,
            "limit": DAILY_REQUEST_LIMIT,
            "remaining": 0,
            "reset_at": reset_at,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Rate limit check fallo (fail-open)",
            extra={
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "fingerprint": fingerprint,
            },
        )
        return True, {"degraded": True, "limit": DAILY_REQUEST_LIMIT}


def _write_prediction_item(
    prediction_id: str,
    fingerprint: str,
    timestamp_iso: str,
    result_status: str,
    top_predictions: list[dict[str, Any]] | None,
    max_birdnet_confidence: float | None,
    inference_time_ms: float,
    n_windows: int | None,
    audio_duration_s: float,
    audio_size_bytes: int,
    s3_key: str | None,
    training_consent: bool,
) -> bool:
    """Escribe un item PREDICTION en DynamoDB — un row por request a /predict.

    Cubre los 3 desenlaces del pipeline: detected (con top-3), rechazos del
    pre-filter sintetico (white_noise / pure_tone) y rechazo del gate BirdNET
    (not_a_bird). Los campos opcionales (top_*, max_birdnet_confidence,
    n_windows, s3_key) se OMITEN del item cuando no aplican — DynamoDB no
    necesita placeholders y omitir es mas barato que escribir NULL.

    Floats -> Decimal: el resource API de DynamoDB NO acepta float (levanta
    "Float types are not supported"). Patron `Decimal(str(round(x, n)))`: el
    round acota a la precision significativa y el paso por str evita arrastrar
    ruido binario de float64 (Decimal(0.1) != Decimal("0.1")). Los ints
    (audio_size_bytes, n_windows, ttl) son nativos, no se convierten.

    Idempotency: `ConditionExpression="attribute_not_exists(pk)"`. Si Lambda
    reintenta la misma invocacion, el segundo PutItem falla silente (INFO, no
    ERROR) en vez de duplicar el row o pisar feedback/review ya escritos.

    review_status="pending": todo item nace pendiente de revision humana. El
    annotation tool human-in-the-loop (Fase 5) lo transiciona a approved/
    rejected/skipped. El feedback de usuarios (feedback_status) NO promueve
    audio a training automaticamente — pasa por esa revision primero.

    Fail-LOUD (a diferencia de `_check_and_increment_rate_limit`, que es
    fail-open/WARN): un PutItem fallido se loguea a ERROR. Perder un item es
    perder data del dashboard Fase 5 y del pool de active learning, y queremos
    enterarnos (alarmable en CloudWatch Logs). NO re-lanza: la clasificacion ya
    fue exitosa y el usuario debe recibir su 200; romper la respuesta por un
    write de side-effect seria una regresion de UX.

    Returns:
        True  si el item se persistio, o si ya existia (ConditionalCheckFailed
              por la guarda de idempotency cuenta como exito — la prediccion ya
              quedo registrada).
        False si hubo un error de escritura distinto de ConditionalCheckFailed.
        El caller (Step 2.e) puede agregar el bool para trackear write success
        rate; Fase 5 lo expone como "% predictions successfully persisted".
    """
    item: dict[str, Any] = {
        "pk": f"PRED#{prediction_id}",
        # sk constante: pk (UUID4) ya es globalmente unico — 1 item por
        # particion, sin jerarquia. El timestamp vive como atributo top-level
        # (y como sort key del GSI), no aporta nada en la composite key.
        # "META" (label categorico) deja la particion abierta a sub-items
        # futuros (sk="AUDIT#...", etc.) y habilita el UpdateItem de /feedback
        # (Fase 4c) con solo pk + sk literal, sin Query previo.
        "sk": "META",
        "item_type": "PREDICTION",
        "prediction_id": prediction_id,
        "timestamp": timestamp_iso,
        "fingerprint": fingerprint,
        "result_status": result_status,
        "inference_time_ms": Decimal(str(round(inference_time_ms, 2))),
        "audio_duration_s": Decimal(str(round(audio_duration_s, 2))),
        "audio_size_bytes": audio_size_bytes,
        "model_version": MODEL_VERSION,
        "training_consent": training_consent,
        "feedback_status": "none",
        "review_status": "pending",
        "ttl": int(time.time()) + PREDICTION_TTL_SECONDS,
    }
    if s3_key is not None:
        item["s3_key"] = s3_key
    if n_windows is not None:
        item["n_windows"] = n_windows
    if max_birdnet_confidence is not None:
        item["max_birdnet_confidence"] = Decimal(str(round(max_birdnet_confidence, 4)))
    if top_predictions:
        top1 = top_predictions[0]
        item["top1_species"] = top1["species"]
        item["top1_confidence"] = Decimal(str(round(top1["confidence"], 4)))
        item["top3_predictions"] = [
            {
                "species": p["species"],
                "common_name": p.get("common_name"),
                "confidence": Decimal(str(round(p["confidence"], 4))),
            }
            for p in top_predictions[:3]
        ]

    try:
        _DDB_TABLE.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(pk)",
        )
        return True
    except _DDB_TABLE.meta.client.exceptions.ConditionalCheckFailedException:
        # Reintento de Lambda sobre la misma invocacion: el item ya existe.
        # No es error — la prediccion ya quedo registrada (idempotency = exito).
        logger.info(
            "PutItem idempotente: prediction_id ya existe, no se reescribe",
            extra={"prediction_id": prediction_id},
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(
            "PutItem PREDICTION fallo — item perdido (fail-loud)",
            extra={
                "prediction_id": prediction_id,
                "result_status": result_status,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            },
        )
        return False


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


def _predict_response(
    status: int,
    payload: dict,
    *,
    prediction_id: str | None = None,
    rate_info: dict | None = None,
) -> dict:
    """Como `_response`, pero inyecta `prediction_id` y `rate_info` si existen.

    El frontend (Fase 4c) lee `prediction_id` de cada respuesta de prediccion
    para el endpoint /feedback, y `rate_info` para mostrar la cuota restante
    debajo del boton. Ambos se omiten cuando todavia no existen en el flujo:
    los 400 de validacion (Fase 1, previos al rate limit) no llevan ninguno;
    el 429 lleva `rate_info` pero no `prediction_id` (ninguna prediccion ocurrio).
    """
    if prediction_id is not None:
        payload["prediction_id"] = prediction_id
    if rate_info is not None:
        payload["rate_info"] = rate_info
    return _response(status, payload)


def handler(event: dict, context: Any = None) -> dict:
    """AWS Lambda entry point — routea por path.

    Rutas (API Gateway HTTP API v2 envia `rawPath`):
        POST /upload-url   -> presigned PUT URL para subir audio a S3
        POST /feedback     -> aplica feedback del usuario a una prediccion
        POST /predict      -> clasifica audio (default si rawPath ausente, para
                              preservar compat con tests directos sin event API GW)

    El refactor a routing (Fase 3) en vez de un solo handler mantiene la
    misma Lambda function ARN — API Gateway routea 3 paths al mismo Lambda
    y nosotros despachamos internamente. Mas barato que 3 Lambdas separadas
    (sin cold start duplicado).

    Fase 5A — warm path: si el invoke trae ``{"warm": true}`` (EventBridge
    Scheduler cada 7 min + demo.load del frontend), ejecutamos el pipeline
    dummy ANTES del routing y retornamos. Este branch es deliberadamente
    PRE-validation y PRE-rate-limit: el warm NO debe consumir cuota del
    fingerprint default, NO debe loguear un PREDICTION en DDB, NO debe leer
    de S3. Solo ejercita CPU paths (librosa JIT, TFLite, ONNX) y se va.

    Detection: leemos ``event.get("warm")`` directo del top-level del event,
    NO de body. API Gateway nunca produce ese campo; lo unicos productores
    son EventBridge (invoca Lambda directo con el payload) y el cliente
    nuestro vía body (que parseamos abajo en el path normal con presencia
    de la key del top-level — ver "Warm via API GW" abajo).

    Warm via API GW: si el frontend manda ``{"warm": true}`` por HTTP POST
    a cualquier path, API GW wrappea ese JSON en event["body"] (string).
    Para soportar ese caso ademas del invoke directo, chequeamos ambos.
    """
    if _is_warm_invoke(event):
        try:
            _warm_pipeline()
        except Exception as e:  # noqa: BLE001 — warm no debe romper el container
            logger.warning(
                "warm pipeline fallo (ignorado)",
                extra={"error_type": type(e).__name__, "error_msg": str(e)},
            )
        return _response(200, {"warm": True})

    raw_path = event.get("rawPath") or event.get("path") or ""
    if raw_path.endswith("/upload-url"):
        return _handle_upload_url(event)
    if raw_path.endswith("/feedback"):
        return _handle_feedback(event)
    return _handle_predict(event)


def _is_warm_invoke(event: dict) -> bool:
    """True si el event es un warm — chequea top-level + body.

    Top-level ``event["warm"]`` -> invoke directo (EventBridge Scheduler).
    Body parseado ``event["body"]`` con ``{"warm": true}`` -> via API GW HTTP.

    No usa ``_parse_body`` porque queremos fallar SILENT si el body es
    invalido y dejar que el path normal lo maneje con un 400. El warm
    detection es additive, no debe interferir con el flow de error normal.
    """
    if event.get("warm") is True:
        return True
    body = event.get("body")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            return False
        return isinstance(parsed, dict) and parsed.get("warm") is True
    if isinstance(body, dict):
        return body.get("warm") is True
    return False


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
        {"audio_b64": "<base64>", "fingerprint": "fp_<32hex>",
         "top_k": 3, "training_consent": false}

    Body schema v2 (Fase 3 — audios largos via S3):
        {"s3_key": "uploads/<uuid4>.<ext>", "fingerprint": "fp_<32hex>",
         "top_k": 3, "training_consent": false}

    Orden de operaciones (Fase 4 Step 2.e — validado vs AWS Lambda docs):
        Fase 1  Parse & validate: body, top_k, fingerprint, audio source.
        Fase 2  Rate limit: 429 si excedido — NO escribe ni consume nada mas.
        Fase 3  Generar prediction_id + timestamp.
        Fase 4  Adquirir (b64 / S3) + decodificar audio.
        Fase 5  Pre-filter sintetico (Capa 1).
        Fase 6  Gate BirdNET (Capa 2).
        Fase 7  Clasificador ONNX (Capa 3).
        Fase 8  Escribir item PREDICTION (side effect, fail-loud, 4 desenlaces).
        Fase 9  Responder con prediction_id + rate_info.

    El rate limit va DESPUES de validar input: un fingerprint malformado (bug
    del frontend) no debe consumir cuota. Una vez pasado el rate limit, la
    cuota YA se consumio — un fallo posterior (audio corrupto, etc.) NO la
    devuelve. Anti-abuso intencional; no implementamos reserve+commit (ver
    Future Work).

    prediction_id va en toda respuesta desde Fase 3; rate_info desde Fase 2
    (incluido el 429). Los 400 de Fase 1 no llevan ninguno. En errores de
    Fase 4-7 el prediction_id es solo correlation ID (no se escribio item).
    """
    # --- Fase 1: parse & validate ------------------------------------------
    try:
        body = _parse_body(event)
    except ValueError as e:
        return _response(400, {"error": str(e)})

    top_k_raw = body.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = max(1, min(int(top_k_raw), _NUM_CLASSES))
    except (TypeError, ValueError):
        return _response(400, {"error": f"top_k inválido: {top_k_raw!r}"})

    fingerprint = body.get("fingerprint")
    if not isinstance(fingerprint, str) or not FINGERPRINT_PATTERN.match(fingerprint):
        return _response(400, {
            "error": "fingerprint faltante o inválido "
                     "(esperado 'fp_' + 32 caracteres hex)"
        })

    training_consent = bool(body.get("training_consent", False))

    has_b64 = "audio_b64" in body
    has_s3 = "s3_key" in body
    if has_b64 and has_s3:
        return _response(400, {
            "error": "body no puede tener ambos 'audio_b64' y 's3_key'"
        })
    if not has_b64 and not has_s3:
        return _response(400, {
            "error": "body debe tener 'audio_b64' o 's3_key'"
        })

    # --- Fase 2: rate limit ------------------------------------------------
    rate_ok, rate_info = _check_and_increment_rate_limit(fingerprint)
    if not rate_ok:
        return _predict_response(
            429,
            {"error": f"Llegaste al limite diario ({rate_info['limit']} audios). "
                      "Volvé mañana para clasificar mas."},
            rate_info=rate_info,
        )

    # --- Fase 3: generar identificadores -----------------------------------
    prediction_id = str(uuid.uuid4())
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # --- Fase 4: adquirir + decodificar audio ------------------------------
    # Retrocompat: audio_b64 sigue andando para clientes que no quieran el
    # round-trip extra del presigned URL. La lectura/decode va DESPUES del
    # rate limit — la cuota ya se consumio aca (anti-abuso intencional).
    s3_key = body["s3_key"] if has_s3 else None
    if has_b64:
        try:
            audio_bytes = base64.b64decode(body["audio_b64"], validate=True)
        except (ValueError, base64.binascii.Error) as e:
            return _predict_response(
                400, {"error": f"audio_b64 inválido: {e}"},
                prediction_id=prediction_id, rate_info=rate_info,
            )
    else:
        try:
            audio_bytes = _read_audio_from_s3(s3_key)
        except ValueError as e:
            # Key invalido (path traversal, prefix/formato wrong) — NO
            # tocamos S3, defense in depth.
            return _predict_response(
                400, {"error": str(e)},
                prediction_id=prediction_id, rate_info=rate_info,
            )
        except _S3_CLIENT.exceptions.NoSuchKey:
            return _predict_response(
                404, {"error": "s3_key no existe o expiró"},
                prediction_id=prediction_id, rate_info=rate_info,
            )
        except Exception as e:  # noqa: BLE001
            return _predict_response(
                500, {"error": f"S3 read falló: {type(e).__name__}: {e}"},
                prediction_id=prediction_id, rate_info=rate_info,
            )

    try:
        y = _load_audio_bytes(audio_bytes)
    except Exception as e:  # noqa: BLE001
        return _predict_response(
            400, {"error": f"no se pudo decodificar audio: {e}"},
            prediction_id=prediction_id, rate_info=rate_info,
        )

    audio_size_bytes = len(audio_bytes)
    audio_duration_s = len(y) / SAMPLE_RATE
    t0 = time.perf_counter()

    # --- Fase 5: pre-filter sintetico (Capa 1) -----------------------------
    # Capa 1 (pre-BirdNET): rechazo de audio sintetico. Ahorra invoke a BirdNET.
    try:
        synthetic_reason = _classify_synthetic(y)
    except Exception as e:  # noqa: BLE001
        return _predict_response(
            500, {"error": f"pre-filter falló: {type(e).__name__}: {e}"},
            prediction_id=prediction_id, rate_info=rate_info,
        )
    if synthetic_reason is not None:
        inference_ms = (time.perf_counter() - t0) * 1000.0
        # Fase 8 (reject sintetico). El bool de _write_prediction_item se
        # ignora a proposito: es fail-loud (ya logueo ERROR si fallo) y la
        # response NO se rompe por un write de side-effect.
        _write_prediction_item(
            prediction_id=prediction_id, fingerprint=fingerprint,
            timestamp_iso=timestamp_iso, result_status=synthetic_reason,
            top_predictions=None, max_birdnet_confidence=None,
            inference_time_ms=inference_ms, n_windows=None,
            audio_duration_s=audio_duration_s, audio_size_bytes=audio_size_bytes,
            s3_key=s3_key, training_consent=training_consent,
        )
        return _predict_response(200, {
            "detected": False,
            "reason": synthetic_reason,
            "model_version": MODEL_VERSION,
            "inference_time_ms": round(inference_ms, 2),
        }, prediction_id=prediction_id, rate_info=rate_info)

    # --- Fase 6 + 7: gate BirdNET (Capa 2) + clasificador ONNX (Capa 3) ----
    # max_birdnet_confidence se persiste SIEMPRE (gated y happy path) para
    # tener distribucion empirica y poder tunear el threshold con datos.
    try:
        embedding, max_conf_per_window, n_windows = _embed(y)
        max_birdnet_confidence, detected = _evaluate_detection(max_conf_per_window)
        predictions = _classify(embedding, top_k) if detected else None
    except Exception as e:  # noqa: BLE001
        return _predict_response(
            500, {"error": f"inferencia falló: {type(e).__name__}: {e}"},
            prediction_id=prediction_id, rate_info=rate_info,
        )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # --- Fase 8 (reject not_a_bird): el gate BirdNET no detectó ave --------
    if not detected:
        _write_prediction_item(
            prediction_id=prediction_id, fingerprint=fingerprint,
            timestamp_iso=timestamp_iso, result_status="not_a_bird",
            top_predictions=None, max_birdnet_confidence=max_birdnet_confidence,
            inference_time_ms=inference_ms, n_windows=n_windows,
            audio_duration_s=audio_duration_s, audio_size_bytes=audio_size_bytes,
            s3_key=s3_key, training_consent=training_consent,
        )
        return _predict_response(200, {
            "detected": False,
            "reason": "not_a_bird",
            "max_birdnet_confidence": round(max_birdnet_confidence, 4),
            "model_version": MODEL_VERSION,
            "n_windows": n_windows,
            "inference_time_ms": round(inference_ms, 2),
        }, prediction_id=prediction_id, rate_info=rate_info)

    # --- Fase 8 (detected): clasificación exitosa --------------------------
    _write_prediction_item(
        prediction_id=prediction_id, fingerprint=fingerprint,
        timestamp_iso=timestamp_iso, result_status="detected",
        top_predictions=predictions, max_birdnet_confidence=max_birdnet_confidence,
        inference_time_ms=inference_ms, n_windows=n_windows,
        audio_duration_s=audio_duration_s, audio_size_bytes=audio_size_bytes,
        s3_key=s3_key, training_consent=training_consent,
    )

    # --- Fase 9: responder -------------------------------------------------
    return _predict_response(200, {
        "predictions": predictions,
        "model_version": MODEL_VERSION,
        "n_windows": n_windows,
        "inference_time_ms": round(inference_ms, 2),
        "max_birdnet_confidence": round(max_birdnet_confidence, 4),
    }, prediction_id=prediction_id, rate_info=rate_info)


def _handle_feedback(event: dict) -> dict:
    """POST /feedback -> aplica feedback del usuario a una predicción.

    Validation pattern: OpenAPI discriminator (`action`) con oneOf schema
    variants. Cada valor de `action` define qué campos adicionales son
    válidos:
    - "confirmed": sin corrected_species.
    - "corrected": corrected_species requerido, debe estar en las 20 especies.
    - "rejected_as_non_bird": sin corrected_species.

    Combinaciones inválidas (ej. corrected_species con action="confirmed")
    se rechazan con 400. Campos extra desconocidos se ignoran silenciosamente.

    Body schema:
        {"prediction_id": "<uuid4>", "fingerprint": "fp_<32hex>",
         "action": "confirmed" | "corrected" | "rejected_as_non_bird",
         "corrected_species": "<especie>"}   # solo si action == "corrected"

    Efecto: UpdateItem sobre el item PREDICTION existente — Key pk + sk="META"
    (ver Schema decision 4c). Setea feedback_status, feedback_timestamp y, si
    aplica, feedback_corrected_species.

    Idempotency: ConditionExpression `attribute_exists(pk) AND feedback_status
    = "none"` — un PREDICTION acepta feedback UNA sola vez. Nota de semántica:
    `attribute_exists(pk)` NO significa "existe algún item con ese pk"; evalúa
    sobre el item de la Key EXACTA (pk + sk="META"). Como cada prediction_id
    tiene exactamente un item (sk constante), la semántica es la deseada.

    Responses:
        200  feedback aplicado.
        400  body inválido (parse / prediction_id / fingerprint / action /
             corrected_species).
        404  prediction_id no corresponde a ninguna predicción.
        409  esa predicción ya tiene feedback (doble submit).
        500  error de DynamoDB, o item PREDICTION malformado (sin
             feedback_status) — ambos son bugs del backend, logueados a ERROR.

    El 404 / 409 / 500-malformado se disambiguan con
    ReturnValuesOnConditionCheckFailure="ALL_OLD": el error de condición trae
    el item viejo SOLO si existía. Sin item -> 404. Con item y feedback_status
    presente -> 409. Con item pero sin feedback_status -> schema drift -> 500.
    ⚠️ Ese item viejo viene en DynamoDB-JSON CRUDO (values {"S": ...}/{"N":
    ...}; keys planas), NO deserializado por el resource API — quirk de la
    excepción, verificado empíricamente. De ahí los accesos `.get("feedback_
    status", {}).get("S")`.
    """
    # --- Parse & validate body ---------------------------------------------
    try:
        body = _parse_body(event)
    except ValueError as e:
        return _response(400, {"error": str(e)})

    prediction_id = body.get("prediction_id")
    if not isinstance(prediction_id, str):
        return _response(400, {"error": "prediction_id faltante o no es string"})
    try:
        parsed_uuid = uuid.UUID(prediction_id)
    except ValueError:
        return _response(400, {"error": "prediction_id no es un UUID válido"})
    if parsed_uuid.version != 4 or str(parsed_uuid) != prediction_id:
        return _response(400, {
            "error": "prediction_id debe ser un UUID4 en formato canónico"
        })

    # fingerprint: se valida por consistencia de contrato con /predict, pero
    # NO se cruza contra el de la predicción — el feedback no es un límite de
    # seguridad y la ConditionExpression deliberadamente no lo incluye.
    fingerprint = body.get("fingerprint")
    if not isinstance(fingerprint, str) or not FINGERPRINT_PATTERN.match(fingerprint):
        return _response(400, {
            "error": "fingerprint faltante o inválido "
                     "(esperado 'fp_' + 32 caracteres hex)"
        })

    action = body.get("action")
    if action not in _FEEDBACK_ACTIONS:
        return _response(400, {
            "error": "action inválido "
                     "(esperado: confirmed | corrected | rejected_as_non_bird)"
        })

    # corrected_species discriminado por `action`. JSON null se trata como
    # ausente (None). Matriz de validación 4c (discriminator oneOf).
    corrected_species = body.get("corrected_species")
    if action == "corrected":
        if corrected_species is None:
            return _response(400, {
                "error": "corrected_species requerido para action='corrected'"
            })
        if corrected_species not in _VALID_SPECIES:
            return _response(400, {
                "error": "corrected_species no es una especie soportada"
            })
    elif corrected_species is not None:
        return _response(400, {
            "error": f"corrected_species no aplica a action='{action}'"
        })

    # --- UpdateItem sobre el item PREDICTION existente ---------------------
    # Alias de TODOS los attribute names (best practice del proyecto, aunque
    # ninguno sea reserved keyword). :none se usa solo en la ConditionExpression.
    feedback_ts = datetime.now(timezone.utc).isoformat()
    update_expr = "SET #fstatus = :status, #fts = :ts"
    expr_names = {"#fstatus": "feedback_status", "#fts": "feedback_timestamp"}
    expr_values = {":status": action, ":ts": feedback_ts, ":none": "none"}
    if action == "corrected":
        update_expr += ", #fcs = :cs"
        expr_names["#fcs"] = "feedback_corrected_species"
        expr_values[":cs"] = corrected_species

    try:
        _DDB_TABLE.update_item(
            Key={"pk": f"PRED#{prediction_id}", "sk": "META"},
            UpdateExpression=update_expr,
            ConditionExpression="attribute_exists(pk) AND #fstatus = :none",
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
            ReturnValuesOnConditionCheckFailure="ALL_OLD",
        )
    except _DDB_TABLE.meta.client.exceptions.ConditionalCheckFailedException as e:
        # ALL_OLD: el error trae "Item" SOLO si el item existía, en DynamoDB-JSON
        # CRUDO (las VALUES son {"S": ...}/{"N": ...}; las KEYS, planas).
        old_item = e.response.get("Item")

        # Caso 1: no existía ningún item con esa Key -> 404.
        if not old_item:
            return _response(404, {"error": "prediction_id no encontrado"})

        # Caso 2: existe pero sin feedback_status. _write_prediction_item SIEMPRE
        # lo escribe ("none") -> si falta, es schema drift / bug del backend, no
        # error del usuario. Log a ERROR (alarmable), 500 al cliente.
        if "feedback_status" not in old_item:
            logger.error(
                "Item PREDICTION malformado: feedback_status ausente",
                extra={
                    "prediction_id": prediction_id,
                    "old_item_keys": list(old_item.keys()),
                },
            )
            return _response(500, {
                "error": "estado de la predicción inconsistente"
            })

        # Caso 3: existe con feedback_status != "none" -> ya tiene feedback.
        # current_status le dice al frontend qué quedó registrado.
        return _response(409, {
            "error": "esta predicción ya tiene feedback registrado",
            "current_status": old_item.get("feedback_status", {}).get("S", "unknown"),
        })
    except Exception as e:  # noqa: BLE001
        # Fail-loud: un error real de DynamoDB se loguea a ERROR (alarmable en
        # CloudWatch). A diferencia del write best-effort de /predict, acá el
        # feedback ES la operación pedida — si falla, el usuario recibe 500.
        logger.error(
            "UpdateItem /feedback falló",
            extra={
                "prediction_id": prediction_id,
                "action": action,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            },
        )
        return _response(500, {
            "error": f"no se pudo registrar el feedback: {type(e).__name__}"
        })

    payload = {"prediction_id": prediction_id, "feedback_status": action}
    if action == "corrected":
        payload["feedback_corrected_species"] = corrected_species
    return _response(200, payload)


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
