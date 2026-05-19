"""Cliente HTTP del flow de prediccion (Fase 3 — S3 presigned URLs).

Flujo de ``predict(audio)`` por llamada (3 requests):
    1. POST ``/upload-url`` -> presigned PUT URL + s3_key
    2. PUT audio directo a S3 (no pasa por API GW)
    3. POST ``/predict`` con ``{s3_key, top_k}`` -> PredictResult

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

# (connect, read) en segundos. El read es alto a propósito: el cold start
# medido en Fase 1 fue ~26.8 s y el cap de integración del HTTP API es 30 s.
REQUEST_TIMEOUT: tuple[float, float] = (5.0, 35.0)

# Fase 3: timeouts separados.
# - /upload-url: sin BirdNET, super rapido (solo genera URL). Cap chico.
# - PUT a S3: no pasa por API GW, sin cap de integration. Read mas largo
#   para audios grandes (15 MB en ~5 s con buena conexion). NO usa retries
#   del Session: presigned tiene TTL 10min, reintentar es seguro pero si
#   falla mejor pedir URL nueva.
UPLOAD_URL_TIMEOUT: tuple[float, float] = (5.0, 15.0)
S3_PUT_TIMEOUT: tuple[float, float] = (5.0, 30.0)

# Reintentos: backoff 1.5s / 3s / 6s. status_forcelist incluye 429 y 503 porque
# Fase 1 verificó que el throttling del HTTP API devuelve 503 (no el 429 que
# documenta AWS).
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
RETRY_STATUS = (429, 503)


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass
class UploadUrlResult:
    """Resultado de POST /upload-url. Mismo patron que PredictResult.

    Si ``ok=True``: ``upload_url`` + ``s3_key`` + ``content_type`` listos
    para usar (PUT a S3 + post-predict con s3_key).
    Si ``ok=False``: ``error_kind`` + ``error_message`` explican que paso.
    """

    ok: bool
    upload_url: str | None = None
    s3_key: str | None = None
    content_type: str | None = None
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
    # "throttled" | "timeout" | "server" | "bad_request"
    #   | "network" | "bad_response" | "config" | "upload_failed"
    error_kind: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Sesión HTTP con reintentos
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    """``requests.Session`` con reintentos automáticos vía urllib3.

    - ``status_forcelist=[429, 503]``: reintenta el throttling del HTTP API, que
      devuelve 503 (verificado en Fase 1), no el 429 que documenta AWS.
    - ``read=False``: NO reintenta read-timeouts. Si ya esperamos 35 s, reintentar
      otros 35 s es peor que fallar rápido; se re-lanza como ReadTimeout.
    - ``connect`` (hereda de ``total``): SÍ reintenta errores de conexión
      transitorios (red que parpadea, DNS).
    - ``allowed_methods=["POST"]``: por defecto urllib3 no reintenta POST. Acá es
      seguro porque /predict NO muta estado (solo devuelve una predicción), así
      que un POST repetido es efectivamente idempotente. OJO: el futuro endpoint
      /flag (Fase 4) escribe en DynamoDB — eso SÍ muta estado y va a necesitar
      idempotency keys de verdad; no copiar este patrón allá.
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
) -> tuple[bool, str | None]:
    """PUT bytes a S3 con presigned URL. Header Content-Type EXPLICITO.

    S3 valida que el Content-Type del PUT matchee el ContentType firmado en
    el presigned. Si difiere -> 403 SignatureDoesNotMatch.

    NO usa la session con retries: presigned tiene TTL 10min, si falla mejor
    pedir URL nueva que reintentar la misma.

    Returns (ok, error_message). El error_message es para logging del caller,
    no se muestra directo al usuario (el caller mapea a 'upload_failed').
    """
    try:
        resp = requests.put(
            presigned_url,
            data=audio_bytes,
            headers={"Content-Type": content_type},
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
    # 429/503 normalmente los reintenta la Session y, si se agotan, llegan como
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
    api_url: str | None = None,
) -> PredictResult:
    """Sube ``audio`` a S3 y dispara la prediccion. Devuelve PredictResult.

    Flow (3 requests):
        1. POST /upload-url -> presigned URL + s3_key
        2. PUT audio a S3 (directo, no por API GW)
        3. POST /predict {s3_key, top_k}

    Nunca lanza por errores de red/HTTP: todo se mapea a ``error_kind``.
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
    s3_ok, s3_err = _upload_to_s3(
        upload.upload_url, audio_bytes, upload.content_type or "application/octet-stream",
    )
    if not s3_ok:
        return PredictResult(
            ok=False, error_kind="upload_failed",
            error_message=f"no se pudo subir el audio: {s3_err}",
        )

    # --- Step 4: POST /predict con s3_key ------------------------------
    body = {"s3_key": upload.s3_key, "top_k": int(top_k)}

    try:
        resp = _get_session().post(url, json=body, timeout=REQUEST_TIMEOUT)
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

    result = predict(args.audio, top_k=args.top_k, api_url=args.api_url)

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
