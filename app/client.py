"""Cliente HTTP del endpoint /predict (API Gateway de Fase 1).

Envuelve el POST a ``${API_GATEWAY_URL}`` con:
    - reintentos automáticos en 429/503 y errores de conexión (urllib3 Retry),
    - manejo de errores mapeado a un único ``PredictResult`` (la UI no atrapa
      excepciones: chequea ``result.ok`` y, si es False, ``error_kind``).

No depende de ``src/`` — el Space de HuggingFace no debe importar el paquete
pesado. La URL se lee de la variable de entorno ``API_GATEWAY_URL``.

Test local (exportar la URL primero):
    $env:API_GATEWAY_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
    python app/client.py data/test_sets/xc_hard/ortalis_canicollis/XC343334.mp3
    python app/client.py <audio.mp3> --top-k 5
    python app/client.py <audio.mp3> --api-url https://otra-url/predict
"""
from __future__ import annotations

import argparse
import base64
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
    #   | "network" | "bad_response" | "config"
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


def _encode_audio(audio: str | Path | bytes) -> str:
    """Path o bytes de audio -> base64 ascii."""
    raw = audio if isinstance(audio, bytes) else Path(audio).read_bytes()
    if not raw:
        raise ValueError("el audio está vacío")
    return base64.b64encode(raw).decode("ascii")


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
    """Envía ``audio`` a /predict y devuelve un PredictResult.

    Nunca lanza por errores de red/HTTP: todo se mapea a ``error_kind``.
    ``audio`` puede ser un path (str/Path) o los bytes crudos del archivo.
    """
    url = _resolve_api_url(api_url)
    if not url:
        return PredictResult(
            ok=False, error_kind="config",
            error_message=f"falta la variable de entorno {API_URL_ENV}",
        )

    try:
        audio_b64 = _encode_audio(audio)
    except (OSError, ValueError) as e:
        return PredictResult(
            ok=False, error_kind="bad_request",
            error_message=f"no se pudo leer el audio: {e}",
        )

    body = {"audio_b64": audio_b64, "top_k": int(top_k)}

    try:
        resp = _get_session().post(url, json=body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RetryError:
        # 429/503 reintentados y agotados.
        return PredictResult(
            ok=False, error_kind="throttled",
            error_message="el servicio está saturado, probá de nuevo en unos segundos",
        )
    except requests.exceptions.ConnectionError:
        # Incluye ConnectTimeout. Errores de conexión ya reintentados y agotados.
        return PredictResult(
            ok=False, error_kind="network",
            error_message="no se pudo conectar con el servidor",
        )
    except requests.exceptions.Timeout:
        # ReadTimeout (no reintentado, read=False) o timeout genérico.
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
