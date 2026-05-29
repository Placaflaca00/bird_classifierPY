"""Decoder de audio compartido entre Lambda (handler.py) y baseline-gen
(scripts/promote.py:generate_baseline). Misma funcion = mismo behavior =
ningun skew silencioso entre baseline y prod (Fase 6.6.a paso 2).

Por que existe este modulo:
    El bug que motivo esto fue un mismatch entre dos paths del pipeline:

    - Lambda decodifica con ``librosa.load(io.BytesIO(audio_bytes), sr=...)``,
      que internamente delega a ``soundfile`` (libsndfile). Cualquier MP3 que
      libsndfile no reconoce -> ``LibsndfileError: Format not recognised`` ->
      HTTP 400 al usuario.

    - ``promote.py:generate_baseline()`` usaba
      ``load_waveform(audio_path)`` con un PATH string, que internamente
      hacia ``librosa.load(path, ...)``. Cuando soundfile falla, librosa
      ANTIGUAMENTE caia en ``audioread`` (que delega a ffmpeg externo).
      Resultado: baseline-gen decodificaba audios que Lambda no podia,
      generando entries falsamente "exitosas" en smoke_baseline.json que
      causaban un MISS deterministico al correr smoke contra prod.

    Caso concreto observado en commit 1207a9b (2026-05-28):
      data/raw/columba_livia/GBIF2243550468.mp3 — MP3 valido para ffmpeg
      pero no para libsndfile 1.1.x/1.2.x. Baseline lo aceptaba con
      conf=0.9752; Lambda lo rechazaba con HTTP 400.

    Solucion estructural (este modulo): UNA funcion compartida. Cualquier
    caller que la llame ve EL MISMO behavior. No se puede divergir porque
    no hay dos copias.

Decisiones tecnicas:

    - **NO usar audioread / NO usar librosa.load(path)**. audioread esta
      deprecado en librosa y se elimina en librosa 1.0 (issue librosa/librosa
      #1267). Ademas no acepta BytesIO — solo paths — asi que ya no podemos
      apoyarnos en el fallback para audios sin path en disco (Lambda recibe
      audio_b64, no un file).

    - **Si en el futuro queremos soportar mas formatos** (VBR de
      Xeno-Canto/GBIF, m4a uploads de usuario, etc.), la fix correcta es un
      DECODER FFMPEG EXPLICITO en este modulo (subprocess sobre tempfile o
      pipe stdin). Eso es ``Prod decoder hardening`` — tarea Fase 7+, no
      esta. Cuando se haga, entra acá y baseline-gen lo hereda solo.

    - **El error se PROPAGA**, no se atrapa. Cada caller decide:
        handler.py -> HTTP 400 al cliente
        generate_baseline -> log warning + skip de ese audio
      Atrapar aca enmascararia el problema y reintroduciria el bug que
      este modulo busca eliminar.

Imports:
    Desde Lambda: ``from audio_io import decode_audio_bytes`` (LAMBDA_TASK_ROOT
    en sys.path por defecto).
    Desde scripts: ``sys.path.insert(0, str(ROOT / 'lambda')); from audio_io
    import decode_audio_bytes``. La dependencia scripts->lambda es explicita
    e intencional: lambda/ es la fuente de verdad del codigo de prod, scripts/
    consume.
"""
from __future__ import annotations

import io

import librosa
import numpy as np

# Sample rate de produccion (handler.py:SAMPLE_RATE = 48000) y de
# precompute_embeddings (mismo valor). BirdNET TFLite espera 48 kHz mono.
# Si cambia uno, sincronizar manualmente con el otro — sino, drift.
DECODE_SAMPLE_RATE = 48_000


def decode_audio_bytes(
    audio_bytes: bytes, sr: int = DECODE_SAMPLE_RATE,
) -> np.ndarray:
    """Decodifica bytes de audio a waveform mono float32 a ``sr`` Hz.

    Implementacion: ``librosa.load(io.BytesIO(bytes), sr=sr, mono=True)``.
    Internamente usa soundfile (libsndfile) — el MISMO decoder que Lambda.

    Args:
        audio_bytes: contenido binario del archivo (mp3/wav/flac/ogg/...).
        sr: sample rate target (default DECODE_SAMPLE_RATE = 48000).

    Returns:
        ndarray mono float32 con N = duracion_segundos * sr.

    Raises:
        soundfile.LibsndfileError: si libsndfile no reconoce el formato.
            NO atrapar aca — el caller decide HTTP 400 vs skip-with-warning.
        Cualquier otra exception de librosa/soundfile.
    """
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return y.astype(np.float32)
