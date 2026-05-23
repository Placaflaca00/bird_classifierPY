"""Cliente HTTP del flow de prediccion + feedback (Fase 3 S3 + Fase 4).

Flujo de ``predict(audio)`` por llamada (3 requests):
    1. POST ``/upload-url`` -> presigned PUT URL + s3_key
    2. PUT audio directo a S3 (no pasa por API GW)
    3. POST ``/predict`` con ``{s3_key, top_k, fingerprint, training_consent}``
       -> PredictResult

``send_feedback()`` (Fase 4c) hace POST a ``/feedback`` para registrar la
correccion del usuario sobre una prediccion previa.

El backend Lambda mantiene retrocompat con ``audio_b64`` (clientes externos /
mobile native podrian usarla), pero este cliente solo usa el flow S3. Razon:
simpler is better, un solo data path mejor para mantener y debuggear.

URLs:
- ``/predict``: viene de env ``API_GATEWAY_URL`` (full URL terminando en /predict).
- ``/upload-url``: se DERIVA reemplazando ``/predict`` -> ``/upload-url`` (no
  necesita env var nueva). Asume convencion: ambos endpoints viven en el mismo
  stage del mismo API.

Errores: mapeados a un unico ``PredictResult.error_kind`` para que la UI tenga
un solo code path. ``predict()`` nunca lanza por errores de red/HTTP.

No depende de ``src/`` — el Space HuggingFace no debe importar el paquete pesado.

Test local (exportar URL primero):
    $env:API_GATEWAY_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
    python app/client.py data/test_sets/xc_hard/ortalis_canicollis/XC343334.mp3
    python app/client.py <audio.mp3> --top-k 5
    python app/client.py <audio.mp3> --api-url https://otra-url/predict
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Carga de .env (sin dependencia python-dotenv)
# ---------------------------------------------------------------------------
# En HF Spaces no existe .env (las env vars vienen del Settings del Space),
# asi que el bloque es no-op alla. Localmente, evita que el usuario tenga que
# exportar la URL en cada sesion PowerShell.
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # setdefault: una env var ya exportada en la shell gana sobre .env.
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass  # silencioso: el caller maneja "config" si falla la URL.


_load_dotenv()


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API_URL_ENV = "API_GATEWAY_URL"
DEFAULT_TOP_K = 3

# (connect, read) en segundos. El read es alto a propósito: el cap de
# integración del HTTP API es 30 s, así que esperar más del lado cliente solo
# acumula latencia sin servir — si la API GW no respondió en 30 s, ya cerró
# la conexión.
REQUEST_TIMEOUT: tuple[float, float] = (5.0, 35.0)

# Cold-start retry: si el primer POST /predict falla con cierre abrupto de
# conexión por API GW (cap 30 s) o 5xx, se reintenta UNA vez con backoff +
# read timeout extendido. El read 60 s del reintento NO espera más que API GW
# (que sigue capeada a 30 s) — espera más que el cap por dos razones:
#   1. El container ya está warm en el reintento (Lambda quedó vivo después
#      de servir el primero) y la respuesta llega en ~1-5 s. El extra es
#      headroom defensivo.
#   2. Algunas latencias borderline (conexión 4G floja, edge CloudFront frío)
#      se manifiestan como read lento pero no como cierre. El extra cubre eso.
COLD_START_RETRY_TIMEOUT: tuple[float, float] = (5.0, 60.0)
COLD_START_RETRY_BACKOFF_S = 2.0
# error_kind retryables: timeout (504 o cliente lo detectó), network
# (ConnectionError = API GW cortó conexión sin response — esta es la firma
# real del cold start observado en prod), server (5xx, incluído 502/504 del
# integration layer), throttled (503 transitorio cuando API GW se queda sin
# capacity post-cold-start).
COLD_START_RETRY_ERROR_KINDS = frozenset({"timeout", "network", "server", "throttled"})

# Fase 3: timeouts separados.
# - /upload-url: sin BirdNET, super rapido (solo genera URL). Cap chico.
# - PUT a S3: no pasa por API GW, sin cap de integration. Read mas largo
#   para audios grandes (15 MB en ~5 s con buena conexion). NO usa retries
#   del Session: presigned tiene TTL 10min, reintentar es seguro pero si
#   falla mejor pedir URL nueva.
UPLOAD_URL_TIMEOUT: tuple[float, float] = (5.0, 15.0)
S3_PUT_TIMEOUT: tuple[float, float] = (5.0, 30.0)

# Reintentos: backoff 1.5s / 3s / 6s.
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
# 429 NO se reintenta: el rate limit del backend es terminal (cuota
# diaria), no transitorio. Reintentar gasta tiempo en algo que no va a
# cambiar hasta reset_at del próximo día UTC. 503 sí se reintenta:
# API Gateway throttling es transitorio (Fase 1 verificó que el HTTP API
# devuelve 503, no el 429 que documenta AWS).
RETRY_STATUS = (503,)


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass
class UploadUrlResult:
    """Resultado de POST /upload-url. Mismo patron que PredictResult.

    Si ``ok=True``: ``upload_url`` + ``s3_key`` + ``content_type`` listos
    para usar (PUT a S3 + post-predict con s3_key). ``tagging`` (Fase 5B)
    es el string a mandar en el header ``x-amz-tagging`` del PUT; None si
    el backend es pre-5B (retrocompat).
    Si ``ok=False``: ``error_kind`` + ``error_message`` explican que paso.
    """

    ok: bool
    upload_url: str | None = None
    s3_key: str | None = None
    content_type: str | None = None
    tagging: str | None = None
    error_kind: str | None = None
    error_message: str | None = None


@dataclass
class PredictResult:
    """Resultado de una llamada a /predict.

    Objeto único para que la UI tenga un solo code path: si ``ok`` es False,
    ``error_kind`` / ``error_message`` explican qué pasó. ``predict()`` nunca
    lanza por errores de red o HTTP.

    Si ``ok`` es True y ``detected`` es False, el backend gating decidio que
    el audio no es ave: ``reject_reason`` indica por que. La UI muestra
    mensaje especifico, no error.
    """

    ok: bool
    predictions: list[dict[str, Any]] = field(default_factory=list)
    model_version: str | None = None
    n_windows: int | None = None
    inference_time_ms: float | None = None
    # Fase 2 - Nivel 2: gating del backend. detected=True implica
    # predictions presentes (happy path). detected=False implica gating
    # rechazo: reject_reason in {"not_a_bird","white_noise","pure_tone"}.
    # Default True para retrocompat con Lambda Fase 1 (que no devuelve
    # la key "detected" — el flag default cubre ese caso transparente).
    detected: bool = True
    reject_reason: str | None = None
    max_birdnet_confidence: float | None = None
    # Fase 4: prediction_id identifica la prediccion para /feedback;
    # rate_info ({requests_today, limit, remaining, [reset_at]}) deja a la UI
    # mostrar la cuota. Presentes en todo 200 de /predict; rate_info ademas
    # en el 429 (donde prediction_id es None — no hubo prediccion).
    prediction_id: str | None = None
    rate_info: dict[str, Any] | None = None
    # "throttled" | "timeout" | "server" | "bad_request" | "rate_limited"
    #   | "network" | "bad_response" | "config" | "upload_failed"
    error_kind: str | None = None
    error_message: str | None = None


@dataclass
class FeedbackResult:
    """Resultado de ``send_feedback()`` -> POST /feedback.

    Mismo patron que PredictResult: si ``ok`` es False, ``error_kind`` /
    ``error_message`` explican que paso. ``send_feedback()`` nunca lanza por
    errores de red o HTTP.

    ``current_status`` solo se puebla en el 409 (la prediccion ya tenia
    feedback): es el ``feedback_status`` previo, para que la UI de un mensaje
    especifico ("ya marcaste esto como X") en vez de uno generico.
    """

    ok: bool
    feedback_status: str | None = None
    current_status: str | None = None
    # "bad_request" | "not_found" | "already_submitted" | "server"
    #   | "throttled" | "timeout" | "network" | "bad_response" | "config"
    error_kind: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Sesión HTTP con reintentos
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    """``requests.Session`` con reintentos automáticos vía urllib3.

    - ``status_forcelist=[503]``: reintenta solo el throttling del HTTP API
      (Fase 1 verificó que devuelve 503). El 429 NO se reintenta — el rate
      limit del backend (Fase 4) es terminal, no transitorio.
    - ``read=False``: NO reintenta read-timeouts. Si ya esperamos 35 s, reintentar
      otros 35 s es peor que fallar rápido; se re-lanza como ReadTimeout.
    - ``connect`` (hereda de ``total``): SÍ reintenta errores de conexión
      transitorios (red que parpadea, DNS).
    - ``allowed_methods=["POST"]``: por defecto urllib3 no reintenta POST. Acá es
      seguro: /predict no muta estado, y /feedback (Fase 4c) es idempotente
      server-side vía el ConditionExpression del UpdateItem. Además solo se
      reintenta 503, que viene de API Gateway ANTES de invocar el Lambda — el
      UpdateItem nunca llegó a correr, así que el retry es la primera ejecución.
    - ``respect_retry_after_header=True``: si la API manda Retry-After algún día,
      lo respetamos gratis.
    """
    retry = Retry(
        total=MAX_RETRIES,
        read=False,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=list(RETRY_STATUS),
        allowed_methods=["POST"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    """Sesión module-level reutilizada entre llamadas (pool de conexiones)."""
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_api_url(api_url: str | None) -> str:
    """URL explícita o ``$API_GATEWAY_URL``. Vacío => predict() lo maneja."""
    return (api_url or os.environ.get(API_URL_ENV, "")).strip()


def _derive_upload_url_url(predict_url: str) -> str:
    """Deriva la URL de /upload-url a partir de la de /predict.

    Asume convencion: ambos endpoints viven en el mismo stage del mismo API.
    Si predict_url termina en '/predict', reemplaza ese segmento.
    Fallback: si no termina en '/predict' (config rara), append '/upload-url'
    al base — defensa pasiva, no crashea pero el caller probablemente vea
    un 404 despues si la convencion no se cumple.
    """
    if predict_url.endswith("/predict"):
        return predict_url[: -len("/predict")] + "/upload-url"
    return predict_url.rstrip("/") + "/upload-url"


def _derive_feedback_url(predict_url: str) -> str:
    """Deriva la URL de /feedback a partir de la de /predict.

    Misma convencion que ``_derive_upload_url_url``: ambos endpoints viven en
    el mismo stage del mismo API. Si predict_url termina en '/predict',
    reemplaza ese segmento; si no, append '/feedback' al base.
    """
    if predict_url.endswith("/predict"):
        return predict_url[: -len("/predict")] + "/feedback"
    return predict_url.rstrip("/") + "/feedback"


def _get_upload_url(ext: str, api_url: str | None = None) -> UploadUrlResult:
    """POST /upload-url con {'ext': ext}. Devuelve presigned + key + MIME.

    Usa la session global (reusa pool de conexiones + retries 429/503).
    Errores mapeados al mismo set de error_kind del predict().
    Nunca lanza — el caller chequea UploadUrlResult.ok.
    """
    base = _resolve_api_url(api_url)
    if not base:
        return UploadUrlResult(
            ok=False, error_kind="config",
            error_message=f"falta la variable de entorno {API_URL_ENV}",
        )
    url = _derive_upload_url_url(base)

    try:
        resp = _get_session().post(
            url, json={"ext": ext}, timeout=UPLOAD_URL_TIMEOUT,
        )
    except requests.exceptions.RetryError:
        return UploadUrlResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    except requests.exceptions.ConnectionError:
        return UploadUrlResult(
            ok=False, error_kind="network",
            error_message="no se pudo conectar con el servidor",
        )
    except requests.exceptions.Timeout:
        return UploadUrlResult(
            ok=False, error_kind="timeout",
            error_message="el servidor tardó demasiado en responder",
        )
    except requests.exceptions.RequestException as e:
        return UploadUrlResult(
            ok=False, error_kind="network",
            error_message=f"error de red: {e}",
        )

    status = resp.status_code
    if status == 200:
        try:
            data = resp.json()
        except ValueError:
            return UploadUrlResult(
                ok=False, error_kind="bad_response",
                error_message="respuesta de /upload-url no es JSON válido",
            )
        if not isinstance(data, dict) or not data.get("upload_url") or not data.get("s3_key"):
            return UploadUrlResult(
                ok=False, error_kind="bad_response",
                error_message="respuesta de /upload-url incompleta",
            )
        return UploadUrlResult(
            ok=True,
            upload_url=data["upload_url"],
            s3_key=data["s3_key"],
            content_type=data.get("content_type"),
            tagging=data.get("tagging"),  # Fase 5B; None si backend pre-5B
        )

    detail = _extract_error_detail(resp)
    if status == 400:
        return UploadUrlResult(
            ok=False, error_kind="bad_request",
            error_message=detail or "el servidor rechazó el pedido de subida",
        )
    if status == 404:
        return UploadUrlResult(
            ok=False, error_kind="config",
            error_message="el endpoint /upload-url no existe (config wrong)",
        )
    if status in RETRY_STATUS:
        return UploadUrlResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    if 500 <= status < 600:
        return UploadUrlResult(
            ok=False, error_kind="server",
            error_message=detail or f"error interno del servidor (HTTP {status})",
        )
    return UploadUrlResult(
        ok=False, error_kind="bad_request",
        error_message=detail or f"respuesta inesperada del servidor (HTTP {status})",
    )


def _upload_to_s3(
    presigned_url: str, audio_bytes: bytes, content_type: str,
    tagging: str | None = None,
) -> tuple[bool, str | None]:
    """PUT bytes a S3 con presigned URL. Headers Content-Type + Tagging EXPLICITOS.

    S3 valida que el Content-Type Y el x-amz-tagging del PUT matcheen lo
    firmado en el presigned. Si difiere -> 403 SignatureDoesNotMatch.

    Fase 5B — ``tagging`` (e.g. ``"retain=false"``): se manda como header
    ``x-amz-tagging`` cuando el caller lo provee. El backend lo firma en el
    presigned a partir de Fase 5B. Default ``None`` (sin header) por
    retrocompat con presigned viejos en flight (TTL 10 min); cuando todos
    los presigned activos sean Fase 5B el default puede dejarse como tal o
    pasar a obligatorio.

    NO usa la session con retries: presigned tiene TTL 10min, si falla mejor
    pedir URL nueva que reintentar la misma.

    Returns (ok, error_message). El error_message es para logging del caller,
    no se muestra directo al usuario (el caller mapea a 'upload_failed').
    """
    headers = {"Content-Type": content_type}
    if tagging:
        headers["x-amz-tagging"] = tagging
    try:
        resp = requests.put(
            presigned_url,
            data=audio_bytes,
            headers=headers,
            timeout=S3_PUT_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return False, "timeout al subir a S3"
    except requests.exceptions.ConnectionError:
        return False, "no se pudo conectar con S3"
    except requests.exceptions.RequestException as e:
        return False, f"error de red al subir a S3: {e}"

    if resp.status_code in (200, 204):
        return True, None
    return False, f"S3 rechazó el upload (HTTP {resp.status_code})"


def _extract_error_detail(resp: requests.Response) -> str | None:
    """Mensaje legible del body de una respuesta de error, si lo trae.

    El handler manda ``{"error": ...}``; API Gateway manda ``{"message": ...}``.
    """
    try:
        data = resp.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        return data.get("error") or data.get("message")
    return None


def _map_response(resp: requests.Response) -> PredictResult:
    """Mapea una respuesta HTTP a PredictResult."""
    status = resp.status_code

    if status == 200:
        try:
            data = resp.json()
        except ValueError:
            return PredictResult(
                ok=False, error_kind="bad_response",
                error_message="la respuesta del servidor no es JSON válido",
            )
        if not isinstance(data, dict):
            return PredictResult(
                ok=False, error_kind="bad_response",
                error_message="la respuesta no tiene el campo 'predictions'",
            )

        # Fase 2 - Nivel 2 — gating del backend: "no detecte ave".
        # El handler omite "predictions" en este caso. NO es bad_response:
        # es decision legitima del gate (reason in {not_a_bird, white_noise,
        # pure_tone}). Retrocompat con Lambda vieja: si la key "detected" no
        # esta presente, este branch no se activa y caemos al path happy.
        if data.get("detected") is False:
            return PredictResult(
                ok=True,
                detected=False,
                reject_reason=data.get("reason"),
                max_birdnet_confidence=data.get("max_birdnet_confidence"),
                model_version=data.get("model_version"),
                n_windows=data.get("n_windows"),
                inference_time_ms=data.get("inference_time_ms"),
                prediction_id=data.get("prediction_id"),
                rate_info=data.get("rate_info"),
            )

        # Happy path: predicciones presentes. Cubre handler nuevo
        # (detected=True implicito ya que la key existe pero es True) y
        # handler viejo Fase 1 (sin key "detected" — default detected=True
        # del dataclass aplica).
        if not isinstance(data.get("predictions"), list):
            return PredictResult(
                ok=False, error_kind="bad_response",
                error_message="la respuesta no tiene el campo 'predictions'",
            )
        return PredictResult(
            ok=True,
            predictions=data["predictions"],
            model_version=data.get("model_version"),
            n_windows=data.get("n_windows"),
            inference_time_ms=data.get("inference_time_ms"),
            max_birdnet_confidence=data.get("max_birdnet_confidence"),
            prediction_id=data.get("prediction_id"),
            rate_info=data.get("rate_info"),
        )

    detail = _extract_error_detail(resp)

    if status == 400:
        return PredictResult(
            ok=False, error_kind="bad_request",
            error_message=detail or "el servidor rechazó el audio enviado",
        )
    if status == 404:
        # La URL no existe. El usuario de Gradio nunca elige la URL (la hardcodea
        # el dev), así que un 404 es backend mal configurado, no input inválido.
        return PredictResult(
            ok=False, error_kind="config",
            error_message="el endpoint configurado no existe (revisar API_GATEWAY_URL)",
        )
    if status == 504:
        return PredictResult(
            ok=False, error_kind="timeout",
            error_message="el servidor tardó demasiado (audio largo o cold start)",
        )
    # 429 = rate limit del backend (Fase 4): cuota diaria agotada. Terminal —
    # la Session NO lo reintenta (429 no está en RETRY_STATUS). El body trae
    # rate_info con reset_at; lo propagamos para que la UI diga cuándo vuelve.
    if status == 429:
        rl_body: dict = {}
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                rl_body = parsed
        except ValueError:
            pass
        return PredictResult(
            ok=False, error_kind="rate_limited",
            rate_info=rl_body.get("rate_info"),
            error_message=rl_body.get("error")
            or "alcanzaste el límite diario de clasificaciones",
        )
    # 503 lo reintenta la Session; si se agotan los retries llega como
    # RetryError (no acá). Este branch es defensivo por si raise_on_status cambia.
    if status in RETRY_STATUS:
        return PredictResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    if 500 <= status < 600:
        return PredictResult(
            ok=False, error_kind="server",
            error_message=detail or f"error interno del servidor (HTTP {status})",
        )
    return PredictResult(
        ok=False, error_kind="bad_request",
        error_message=detail or f"respuesta inesperada del servidor (HTTP {status})",
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def predict(
    audio: str | Path | bytes,
    top_k: int = DEFAULT_TOP_K,
    *,
    fingerprint: str,
    training_consent: bool = False,
    api_url: str | None = None,
) -> PredictResult:
    """Sube ``audio`` a S3 y dispara la prediccion. /predict consume cuota
    del usuario (rate limited).

    ``fingerprint`` es keyword-only y requerido para hacer EXPLICITO que cada
    llamada consume cuota del usuario. Auto-generacion interna ocultaria esta
    dependency, llevando a CLI loops sin rate limit y a callers confundidos
    sobre por que prod hits 429.

    Flow (3 requests):
        1. POST /upload-url -> presigned URL + s3_key
        2. PUT audio a S3 (directo, no por API GW)
        3. POST /predict {s3_key, top_k, fingerprint, training_consent}

    Nunca lanza por errores de red/HTTP: todo se mapea a ``error_kind``
    (incluido ``rate_limited`` cuando el backend devuelve 429).
    ``audio`` puede ser path (str/Path) o bytes crudos del archivo.

    ``ext`` para el presigned se infiere del path. Si ``audio`` es bytes
    (no hay path), asume ``mp3`` (formato mas comun). La validation del
    frontend (``app/validation.py``) ya rechazo formatos no soportados
    antes de llegar aca, asi que en practica el ext es valido.
    """
    url = _resolve_api_url(api_url)
    if not url:
        return PredictResult(
            ok=False, error_kind="config",
            error_message=f"falta la variable de entorno {API_URL_ENV}",
        )

    # --- Step 1: leer bytes + inferir extension del path ----------------
    try:
        if isinstance(audio, bytes):
            audio_bytes = audio
            ext = "mp3"  # fallback razonable; validation ya filtro arriba
        else:
            audio_path = Path(audio)
            audio_bytes = audio_path.read_bytes()
            ext = audio_path.suffix.lstrip(".").lower() or "mp3"
        if not audio_bytes:
            return PredictResult(
                ok=False, error_kind="bad_request",
                error_message="el audio está vacío",
            )
    except (OSError, ValueError) as e:
        return PredictResult(
            ok=False, error_kind="bad_request",
            error_message=f"no se pudo leer el audio: {e}",
        )

    # --- Step 2: pedir presigned URL ------------------------------------
    upload = _get_upload_url(ext, api_url=url)
    if not upload.ok:
        # Propaga error_kind del upload (config/network/throttled/server/bad_request).
        return PredictResult(
            ok=False, error_kind=upload.error_kind,
            error_message=upload.error_message,
        )

    # --- Step 3: PUT a S3 ----------------------------------------------
    # Fase 5B — pasamos el tagging que el backend firmó en el presigned. Si
    # el backend es pre-5B, ``upload.tagging`` es None y no mandamos header
    # (retrocompat).
    s3_ok, s3_err = _upload_to_s3(
        upload.upload_url, audio_bytes,
        upload.content_type or "application/octet-stream",
        tagging=upload.tagging,
    )
    if not s3_ok:
        return PredictResult(
            ok=False, error_kind="upload_failed",
            error_message=f"no se pudo subir el audio: {s3_err}",
        )

    # --- Step 4: POST /predict con s3_key + cold-start retry -----------
    body = {
        "s3_key": upload.s3_key,
        "top_k": int(top_k),
        "fingerprint": fingerprint,
        "training_consent": bool(training_consent),
    }
    return _post_predict_with_retry(url, body)


def _do_post_predict(
    url: str, body: dict, timeout: tuple[float, float],
) -> PredictResult:
    """Un solo intento de POST /predict. Mapea respuesta o excepción a
    PredictResult sin reintentar.

    Extraído para que ``_post_predict_with_retry`` pueda invocarlo dos veces
    con timeouts distintos sin duplicar el bloque try/except.
    """
    try:
        resp = _get_session().post(url, json=body, timeout=timeout)
    except requests.exceptions.RetryError:
        return PredictResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    except requests.exceptions.ConnectionError:
        return PredictResult(
            ok=False, error_kind="network",
            error_message="no se pudo conectar con el servidor",
        )
    except requests.exceptions.Timeout:
        return PredictResult(
            ok=False, error_kind="timeout",
            error_message="el servidor tardó demasiado en responder",
        )
    except requests.exceptions.RequestException as e:
        return PredictResult(
            ok=False, error_kind="network",
            error_message=f"error de red: {e}",
        )

    return _map_response(resp)


def _post_predict_with_retry(url: str, body: dict) -> PredictResult:
    """POST /predict con retry de 1 en caso de cold start.

    Patología medida en prod (CloudWatch + Fase 5A.1):
      - Container cold + init phase de 10 s (cap del Lambda image init) →
        fallback a init-in-handler → Duration 30-42 s.
      - API Gateway HTTP API tiene cap de integración de 30 s. Cuando se
        excede, **cierra la conexión sin response** (en vez de un 504 limpio);
        ``requests`` ve un ``ConnectionError`` o ``Timeout``, no un response
        HTTP. Después del cierre, Lambda sigue ejecutando hasta su propio
        timeout (90 s) y el container queda warm para la próxima invocación.
      - El reintento llega a un container ya warm → respuesta en ~1-5 s.

    Estrategia:
      1. Primer intento con timeout normal (REQUEST_TIMEOUT, read 35 s).
      2. Si falla con error_kind retryable (timeout/network/server/throttled),
         backoff de 2 s + retry con timeout extendido (60 s read).
      3. Si el primer intento fue exitoso o falla con un error NO transitorio
         (bad_request, rate_limited, config, bad_response), NO reintentar:
         son errores terminales del request o config.

    Rate-limited NO se reintenta (es la directiva explícita del .Session vía
    RETRY_STATUS = (503,) y vía rate_limited not in COLD_START_RETRY_ERROR_KINDS):
    cuota diaria agotada del backend, reintentar gasta tiempo en algo que no
    va a cambiar hasta el próximo reset.
    """
    first = _do_post_predict(url, body, REQUEST_TIMEOUT)
    if first.ok or first.error_kind not in COLD_START_RETRY_ERROR_KINDS:
        return first

    time.sleep(COLD_START_RETRY_BACKOFF_S)
    return _do_post_predict(url, body, COLD_START_RETRY_TIMEOUT)


# ---------------------------------------------------------------------------
# Warm pre-load (Fase 5A.2)
# ---------------------------------------------------------------------------
def warm_lambda(api_url: str | None = None) -> None:
    """Fire-and-forget POST ``{"warm": true}`` al endpoint — pre-load del Lambda.

    Pensado para llamarse desde un thread daemon en ``demo.load()`` del
    frontend: dispara el warm pipeline del Lambda (ver
    ``lambda/handler.py:_warm_pipeline``) cuando un usuario abre el Space,
    para que el container esté warm + JIT compilado antes del primer
    Clasificar real.

    El backend detecta el flag ``warm`` y branch antes de cualquier write a
    DDB / S3 — esta llamada NO consume cuota del rate limit ni loguea
    PREDICTION items.

    Errores se silencian: warm es defensa, no critical path. Si el endpoint
    está caido o sin red, el usuario verá el error real cuando intente
    clasificar — preferible a romper el demo.load.

    Timeout (5/15): warm sobre container warm vuelve en <1s, sobre cold
    puede tardar mucho (init phase del Lambda). 15s es un cap razonable que
    libera el thread sin esperar el cold start completo — el container sigue
    warmandose del lado del Lambda igual, beneficiando requests siguientes.
    """
    url = _resolve_api_url(api_url)
    if not url:
        return
    try:
        _get_session().post(url, json={"warm": True}, timeout=(5.0, 15.0))
    except Exception as e:  # noqa: BLE001 — fire-and-forget defensivo
        print(f"[warm_lambda] ignored: {type(e).__name__}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Feedback (Fase 4c)
# ---------------------------------------------------------------------------
def _map_feedback_response(resp: requests.Response) -> FeedbackResult:
    """Mapea una respuesta HTTP de /feedback a FeedbackResult."""
    status = resp.status_code

    if status == 200:
        try:
            data = resp.json()
        except ValueError:
            return FeedbackResult(
                ok=False, error_kind="bad_response",
                error_message="la respuesta de /feedback no es JSON válido",
            )
        return FeedbackResult(
            ok=True,
            feedback_status=data.get("feedback_status")
            if isinstance(data, dict) else None,
        )

    detail = _extract_error_detail(resp)

    if status == 400:
        return FeedbackResult(
            ok=False, error_kind="bad_request",
            error_message=detail or "el servidor rechazó el feedback",
        )
    if status == 404:
        return FeedbackResult(
            ok=False, error_kind="not_found",
            error_message=detail or "la predicción no existe o expiró",
        )
    if status == 409:
        # El feedback ya estaba registrado. El handler (4c) devuelve
        # current_status: el feedback_status que ya tenía la predicción.
        current = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                current = body.get("current_status")
        except ValueError:
            pass
        return FeedbackResult(
            ok=False, error_kind="already_submitted", current_status=current,
            error_message=detail or "esta predicción ya tiene feedback registrado",
        )
    # 503 lo reintenta la Session; defensivo por si raise_on_status cambia.
    if status in RETRY_STATUS:
        return FeedbackResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    if 500 <= status < 600:
        return FeedbackResult(
            ok=False, error_kind="server",
            error_message=detail or f"error interno del servidor (HTTP {status})",
        )
    return FeedbackResult(
        ok=False, error_kind="bad_request",
        error_message=detail or f"respuesta inesperada del servidor (HTTP {status})",
    )


def send_feedback(
    prediction_id: str,
    fingerprint: str,
    action: str,
    corrected_species: str | None = None,
    api_url: str | None = None,
) -> FeedbackResult:
    """POST /feedback — registra el feedback del usuario sobre una prediccion.

    Idempotency-safe a retries de 503. /feedback usa ``_get_session()`` (que
    reintenta 503) con seguridad porque:

    1. El 503 viene de API Gateway, ANTES de invocar el Lambda.
    2. Si hay 503, el UpdateItem NUNCA corrio en DynamoDB.
    3. El retry llega al Lambda y ejecuta el UpdateItem por primera vez.

    Si el Lambda DEVOLVIERA 503 (improbable, seria un bug interno), el
    ConditionExpression del UpdateItem garantiza idempotency a nivel
    DynamoDB: un segundo retry recibiria 409.

    AWS Builders Library: "idempotent operations are safe to retry,
    allowing client code to simplify error handling."

    Args:
        prediction_id: el UUID4 que devolvio /predict en ``PredictResult``.
        fingerprint: mismo fingerprint del usuario (``fp_<32hex>``).
        action: "confirmed" | "corrected" | "rejected_as_non_bird".
        corrected_species: nombre cientifico, SOLO si action == "corrected".

    Nunca lanza — el caller chequea ``FeedbackResult.ok``.
    """
    base = _resolve_api_url(api_url)
    if not base:
        return FeedbackResult(
            ok=False, error_kind="config",
            error_message=f"falta la variable de entorno {API_URL_ENV}",
        )
    url = _derive_feedback_url(base)

    body: dict[str, Any] = {
        "prediction_id": prediction_id,
        "fingerprint": fingerprint,
        "action": action,
    }
    if corrected_species is not None:
        body["corrected_species"] = corrected_species

    try:
        resp = _get_session().post(url, json=body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RetryError:
        return FeedbackResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    except requests.exceptions.ConnectionError:
        return FeedbackResult(
            ok=False, error_kind="network",
            error_message="no se pudo conectar con el servidor",
        )
    except requests.exceptions.Timeout:
        return FeedbackResult(
            ok=False, error_kind="timeout",
            error_message="el servidor tardó demasiado en responder",
        )
    except requests.exceptions.RequestException as e:
        return FeedbackResult(
            ok=False, error_kind="network",
            error_message=f"error de red: {e}",
        )

    return _map_feedback_response(resp)


# ---------------------------------------------------------------------------
# Test local (CLI)
# ---------------------------------------------------------------------------
def _local_main() -> int:
    parser = argparse.ArgumentParser(
        description="Test local del cliente HTTP de /predict."
    )
    parser.add_argument("audio", type=Path, help="Path a un mp3/wav.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--api-url", default=None,
        help=f"Override de la URL. Si no se pasa, se usa ${API_URL_ENV}.",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"audio no existe: {args.audio}", file=sys.stderr)
        return 2

    # /predict exige fingerprint; el CLI genera uno efimero por invocacion
    # (no necesita persistencia cross-run como el frontend con BrowserState).
    fingerprint = f"fp_{uuid.uuid4().hex}"
    result = predict(
        args.audio, top_k=args.top_k,
        fingerprint=fingerprint, api_url=args.api_url,
    )

    if not result.ok:
        print(
            f"ERROR  kind={result.error_kind}  {result.error_message}",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK  model={result.model_version}  n_windows={result.n_windows}  "
        f"inference_ms={result.inference_time_ms}"
    )
    for i, p in enumerate(result.predictions, 1):
        conf = p.get("confidence")
        conf_str = f"{conf:.4f}" if isinstance(conf, (int, float)) else "?"
        common = p.get("common_name") or "-"
        print(f"  {i}. {str(p.get('species')):<28} {conf_str}  ({common})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_local_main())
