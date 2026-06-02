"""Tests de app/validation.py — defense-in-depth capa 1 (Fase 2 - Nivel 1).

Todos los fixtures se generan sinteticamente con stdlib ``wave`` para no
depender de ``data/test_sets/`` (gitignored, no llega a CI). El caso "valido"
usa un WAV en vez de un MP3 porque la API publica acepta ambos y WAV se
genera trivialmente sin binarios externos.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

# El frontend no esta en src/ — agregamos app/ al path para importarlo.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

from validation import (  # noqa: E402
    MAX_DURATION_S,
    MAX_SIZE_BYTES,
    validate_audio,
)


# ---------------------------------------------------------------------------
# Helper: sintetiza un WAV PCM 16-bit de silencio
# ---------------------------------------------------------------------------
def _make_wav(
    path: Path,
    *,
    duration_s: float = 1.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    n_frames = max(1, int(duration_s * sample_rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames * channels)
    return path


# ---------------------------------------------------------------------------
# Caso valido
# ---------------------------------------------------------------------------
def test_valid_wav_mono_16khz(tmp_path: Path) -> None:
    path = _make_wav(tmp_path / "ok.wav", duration_s=1.0, sample_rate=16_000)
    result = validate_audio(path)
    assert result.ok, (
        f"esperaba ok, fallo con {result.error_code}: {result.error_message}"
    )
    assert result.info is not None
    assert result.info.format == "wav"
    assert result.info.duration_s == pytest.approx(1.0, rel=0.05)
    assert result.info.sample_rate_hz == 16_000
    assert result.info.channels == 1
    assert result.info.size_bytes > 0


def test_valid_wav_stereo(tmp_path: Path) -> None:
    """Estereo (channels=2) tambien debe pasar — es el limite, no la excepcion."""
    path = _make_wav(tmp_path / "stereo.wav", channels=2, sample_rate=44_100)
    result = validate_audio(path)
    assert result.ok, result.error_message
    assert result.info.channels == 2


# ---------------------------------------------------------------------------
# Casos invalidos: uno por error_code
# ---------------------------------------------------------------------------
def test_too_short(tmp_path: Path) -> None:
    path = _make_wav(tmp_path / "short.wav", duration_s=0.1)
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "too_short"


def test_too_long(tmp_path: Path) -> None:
    # Generamos un WAV de duracion > MAX_DURATION_S pero pesando < MAX_SIZE_BYTES
    # (8 kHz mono 16-bit a 35s = ~547 KB, muy por debajo del cap de 5 MB).
    path = _make_wav(
        tmp_path / "long.wav",
        duration_s=MAX_DURATION_S + 5,
        sample_rate=8_000,
        channels=1,
    )
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "too_long"


def test_too_big(tmp_path: Path) -> None:
    # 6 MB de ceros con extension .mp3. Size es el primer check, corta antes
    # de que mutagen siquiera intente parsear, asi que no importa que el
    # contenido no sea audio valido.
    path = tmp_path / "big.mp3"
    path.write_bytes(b"\x00" * (MAX_SIZE_BYTES + 1024))
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "too_big"


def test_unsupported_format_png_renamed(tmp_path: Path) -> None:
    """PNG renombrado a .mp3: el ataque clasico de troll.

    mutagen sniffea por magic bytes (no por extension), asi que debe rechazar.
    Aceptamos cualquiera de los dos error_codes posibles:
    - ``unsupported_format``: mutagen detecto pero no es audio (raro con PNG).
    - ``unreadable``: mutagen devolvio None o lanzo (caso esperado para PNG).
    """
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"  # signature
        + b"\x00\x00\x00\rIHDR"
        + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        + b"\x1f\x15\xc4\x89"
        + b"\x00" * 200
    )
    path = tmp_path / "fake.mp3"
    path.write_bytes(png_bytes)
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code in {"unsupported_format", "unreadable"}


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp3"
    path.write_bytes(b"")
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "unreadable"
    assert result.error_detail and "empty" in str(result.error_detail).lower()


def test_bad_sample_rate(tmp_path: Path) -> None:
    path = _make_wav(tmp_path / "low_sr.wav", duration_s=1.0, sample_rate=4_000)
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "bad_sample_rate"
    assert result.error_detail["sample_rate_hz"] == 4_000


def test_too_many_channels(tmp_path: Path) -> None:
    path = _make_wav(tmp_path / "5ch.wav", duration_s=1.0, channels=5)
    result = validate_audio(path)
    assert not result.ok
    assert result.error_code == "too_many_channels"


def test_nonexistent_path(tmp_path: Path) -> None:
    result = validate_audio(tmp_path / "does_not_exist.wav")
    assert not result.ok
    assert result.error_code == "unreadable"


# ---------------------------------------------------------------------------
# Ogg/Opus (notas de voz de WhatsApp): regresion 2026-06-02
# ---------------------------------------------------------------------------
# WhatsApp graba en Ogg/Opus. mutagen.OggOpusInfo NO expone ``sample_rate``
# (Opus siempre opera a 48 kHz), asi que el getattr daba 0 y la validacion
# rechazaba TODO audio de WhatsApp como "unreadable". El fix asume 48000 para
# OggOpus. Generamos el fixture con ffmpeg (workstation lo tiene via winget);
# en CI sin ffmpeg el test se skipea en vez de fallar.
def test_ogg_opus_whatsapp_voice_note(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg no disponible (CI); fixture Ogg/Opus no generable")
    opus_path = tmp_path / "voice_note.opus"
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=2000:duration=2",
            "-c:a", "libopus", "-b:a", "24k", str(opus_path), "-y",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not opus_path.exists():
        pytest.skip("ffmpeg sin encoder libopus; fixture no generable")

    result = validate_audio(opus_path)
    assert result.ok, (
        f"Ogg/Opus rechazado: {result.error_code} {result.error_message}"
    )
    assert result.info is not None
    assert result.info.format == "ogg"
    # Opus siempre 48 kHz; el fix lo asume cuando mutagen omite el atributo.
    assert result.info.sample_rate_hz == 48_000


# ---------------------------------------------------------------------------
# Sanidad: el contrato "nunca lanza" debe sostenerse en input raro
# ---------------------------------------------------------------------------
def test_directory_path_is_unreadable(tmp_path: Path) -> None:
    """Pasar un directorio no debe lanzar — debe devolver unreadable."""
    result = validate_audio(tmp_path)
    assert not result.ok
    assert result.error_code == "unreadable"
