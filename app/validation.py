"""Validacion inline de audios subidos al frontend (Fase 2 - Nivel 1).

Defense-in-depth, capa 1: rechaza archivos obviamente invalidos ANTES de
hacer el POST a /predict. Ahorra cold starts inutiles y le da al usuario
un mensaje especifico en vez de un "timeout" o "bad_request" generico.

Pure-Python: usa ``mutagen`` para parsear headers MP3/WAV/OGG/FLAC sin
decodificar el audio. Cero binarios extras (ffprobe NO esta disponible
en el sandbox de HuggingFace Spaces).

API publica: ``validate_audio(path) -> ValidationResult``. Nunca lanza
(mismo contrato que ``client.PredictResult``).

Test local:
    python -m app.validation data/test_sets/xc_hard/<species>/<file>.mp3
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mutagen


# ---------------------------------------------------------------------------
# Limites
# ---------------------------------------------------------------------------
MIN_DURATION_S = 0.5

# 30 s = hard cap del API Gateway HTTP API (integration timeout). Si dejamos
# pasar audios mas largos, Lambda los procesa pero API Gateway corta antes
# de recibir la respuesta y el usuario ve "timeout" generico en vez de un
# mensaje accionable. Si algun dia migramos a REST API (cap 29 s) o async,
# revisar este numero.
MAX_DURATION_S = 30.0

MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_SAMPLE_RATE_HZ = 8_000
MAX_CHANNELS = 2

ALLOWED_FORMATS = frozenset({"mp3", "wav", "ogg", "flac"})


# ---------------------------------------------------------------------------
# Deteccion de formato: mutagen reconoce por magic bytes, no por extension.
# Esto bloquea trolls que renombran "meme.png" a "meme.mp3" porque mutagen
# devuelve None o lanza, en vez de aceptar la extension a ciegas.
# ---------------------------------------------------------------------------
_MUTAGEN_FORMAT_MAP = {
    "MP3": "mp3",
    "WAVE": "wav",
    "OggVorbis": "ogg",
    "OggOpus": "ogg",
    "OggFLAC": "ogg",
    "FLAC": "flac",
}


# ---------------------------------------------------------------------------
# Mensajes user-facing (es-ES). Centralizados aca para tuning facil.
# ---------------------------------------------------------------------------
_MESSAGES = {
    "too_short": "El audio es muy corto (menos de 0.5 s). Subi uno mas largo.",
    "too_long": "El audio supera los 30 s. Recortalo antes de subirlo.",
    "too_big": "El archivo pesa mas de 5 MB. Subi uno mas chico.",
    "unsupported_format": (
        "Formato no soportado. Subi un archivo mp3, wav, ogg o flac."
    ),
    "bad_sample_rate": (
        "El audio tiene calidad muy baja (menos de 8 kHz). Subi otro."
    ),
    "too_many_channels": (
        "El audio tiene demasiados canales. Subi mono o estereo."
    ),
    "unreadable": (
        "No pude leer el archivo. Puede estar corrupto o no contener audio."
    ),
}


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass
class AudioInfo:
    """Metadata extraida del header. Solo valida si ``ValidationResult.ok``."""

    format: str  # "mp3" | "wav" | "ogg" | "flac"
    duration_s: float
    sample_rate_hz: int
    channels: int
    size_bytes: int


@dataclass
class ValidationResult:
    """Mismo patron que ``client.PredictResult``: un solo objeto, ``ok`` bool.

    Si ``ok=False``, ``error_code`` + ``error_message`` explican que paso.
    ``error_detail`` es para logs (no para mostrar al usuario).
    """

    ok: bool
    info: AudioInfo | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_detail: dict | None = None


def _fail(code: str, **detail) -> ValidationResult:
    """Construye un ValidationResult fallido con el mensaje correspondiente."""
    return ValidationResult(
        ok=False,
        error_code=code,
        error_message=_MESSAGES[code],
        error_detail=detail or None,
    )


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def validate_audio(path: str | Path) -> ValidationResult:
    """Inspecciona ``path`` y devuelve un ``ValidationResult``.

    Chequeos (corta en el primer fallo, orden por costo creciente):
        1. Existe + size > 0 + size <= MAX_SIZE_BYTES.
        2. mutagen parsea el header (rechaza corruptos, renombrados, no-audio).
        3. Formato en ALLOWED_FORMATS.
        4. duration_s en [MIN_DURATION_S, MAX_DURATION_S].
        5. sample_rate_hz >= MIN_SAMPLE_RATE_HZ.
        6. channels en [1, MAX_CHANNELS].

    Nunca lanza: cualquier excepcion de mutagen se mapea a ``unreadable``.
    Acepta ``None`` / "" sin crashear (defensa secundaria; la state machine
    UI ya garantiza path no-vacio).
    """
    if not path:
        return _fail("unreadable", reason="empty_path")
    p = Path(path)

    # 1a: existe.
    if not p.is_file():
        return _fail("unreadable", reason="not_a_file", path=str(p))

    # 1b: size.
    try:
        size = p.stat().st_size
    except OSError as e:
        return _fail("unreadable", reason=f"stat_failed: {e!s}")
    if size == 0:
        return _fail("unreadable", reason="empty_file")
    if size > MAX_SIZE_BYTES:
        return _fail("too_big", size_bytes=size, max_bytes=MAX_SIZE_BYTES)

    # 2: parse con mutagen. Catch amplio porque archivos renombrados pueden
    # lanzar de varias formas (MutagenError, IOError, ValueError, ...).
    try:
        audio = mutagen.File(str(p))
    except Exception as e:  # noqa: BLE001 — ver docstring arriba
        return _fail(
            "unreadable",
            reason=f"mutagen_error: {type(e).__name__}: {e!s}",
        )
    if audio is None or audio.info is None:
        return _fail("unreadable", reason="mutagen_returned_none")

    # 3: formato detectado por mutagen contra whitelist.
    fmt_class = type(audio).__name__
    fmt = _MUTAGEN_FORMAT_MAP.get(fmt_class)
    if fmt is None or fmt not in ALLOWED_FORMATS:
        return _fail("unsupported_format", detected=fmt_class)

    # Extract metadata. Si algun campo falta, lo tratamos como "unreadable":
    # un archivo de audio valido siempre tiene duration/sr/channels.
    try:
        duration_s = float(audio.info.length)
        sample_rate_hz = int(getattr(audio.info, "sample_rate", 0))
        channels = int(getattr(audio.info, "channels", 0))
    except (TypeError, ValueError, AttributeError) as e:
        return _fail("unreadable", reason=f"missing_metadata: {e!s}")

    # 4: duration.
    if duration_s < MIN_DURATION_S:
        return _fail("too_short", duration_s=duration_s)
    if duration_s > MAX_DURATION_S:
        return _fail("too_long", duration_s=duration_s, max_s=MAX_DURATION_S)

    # 5: sample rate. 0 o negativo => archivo malformado, no "calidad baja".
    if sample_rate_hz <= 0:
        return _fail(
            "unreadable", reason=f"invalid_sample_rate: {sample_rate_hz}"
        )
    if sample_rate_hz < MIN_SAMPLE_RATE_HZ:
        return _fail("bad_sample_rate", sample_rate_hz=sample_rate_hz)

    # 6: channels. 0 o negativo => malformado, no "muchos canales".
    if channels < 1:
        return _fail("unreadable", reason=f"invalid_channels: {channels}")
    if channels > MAX_CHANNELS:
        return _fail("too_many_channels", channels=channels)

    return ValidationResult(
        ok=True,
        info=AudioInfo(
            format=fmt,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            size_bytes=size,
        ),
    )


# ---------------------------------------------------------------------------
# CLI local (debug rapido)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("uso: python -m app.validation <audio_path>", file=sys.stderr)
        sys.exit(2)
    result = validate_audio(sys.argv[1])
    if result.ok:
        info = result.info
        print(
            f"OK  format={info.format}  duration={info.duration_s:.2f}s  "
            f"sr={info.sample_rate_hz}Hz  channels={info.channels}  "
            f"size={info.size_bytes}B"
        )
        sys.exit(0)
    print(
        f"FAIL  code={result.error_code}  msg={result.error_message}",
        file=sys.stderr,
    )
    if result.error_detail:
        print(f"      detail={result.error_detail}", file=sys.stderr)
    sys.exit(1)
