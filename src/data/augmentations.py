"""Augmentations de audio y de embeddings.

Dos niveles:

1. **Waveform** (antes de BirdNET): dos pipelines disponibles:
   - ``build_audio_pipeline_conservative()``: SNR + Shift + Gain. Original;
     conservador. Mantenido para reproducir runs viejos. Probado con K=2 en
     ``aug-waveform-k2-cons`` — empeoró test_hard.
   - ``build_audio_pipeline_aggressive()``: + PitchShift + TimeStretch +
     LowPassFilter. OOD-aware: simula audio lejano, mala mic, ruido de campo.
     Para Iteración 2 Fase 5.

   Los embeddings aumentados se generan offline en
   ``scripts/precompute_embeddings.py --augment K`` (uniforme) o
   ``--aug-policy tiered`` (K por especie).

2. **Embedding** (después de BirdNET): ``mixup`` está implementado en
   ``src/models/classifier.py``. Probado con resultado negativo. Se mantiene
   el código pero la receta activa usa ``mixup_alpha=0``.
"""

from __future__ import annotations

from audiomentations import (
    AddGaussianSNR,
    Compose,
    Gain,
    LowPassFilter,
    PitchShift,
    Shift,
    TimeStretch,
)


def build_audio_pipeline_conservative(
    *,
    p_noise: float = 0.7,
    snr_db_min: float = 15.0,
    snr_db_max: float = 30.0,
    p_shift: float = 0.7,
    shift_max_fraction: float = 0.3,
    p_gain: float = 0.5,
    gain_db_min: float = -6.0,
    gain_db_max: float = 6.0,
) -> Compose:
    """Pipeline conservador (la versión original de Fase 3).

    Defaults:
    - SNR 15–30 dB: ruido perceptible pero el canto del ave sigue dominando.
    - Shift ±30% del largo (≈ ±0.9 s en ventanas de 3 s) con rollover.
    - Gain ±6 dB: simula variación de distancia/ganancia del micrófono.
    """
    return Compose(
        [
            AddGaussianSNR(min_snr_db=snr_db_min, max_snr_db=snr_db_max, p=p_noise),
            Shift(
                min_shift=-shift_max_fraction,
                max_shift=shift_max_fraction,
                shift_unit="fraction",
                rollover=True,
                p=p_shift,
            ),
            Gain(min_gain_db=gain_db_min, max_gain_db=gain_db_max, p=p_gain),
        ]
    )


def build_audio_pipeline_aggressive(
    *,
    p_noise: float = 0.8,
    snr_db_min: float = 5.0,
    snr_db_max: float = 20.0,
    p_shift: float = 0.7,
    shift_max_fraction: float = 0.4,
    p_gain: float = 0.6,
    gain_db_min: float = -12.0,
    gain_db_max: float = 12.0,
    p_pitch: float = 0.5,
    pitch_semitones: float = 2.0,
    p_time: float = 0.5,
    time_min_rate: float = 0.85,
    time_max_rate: float = 1.15,
    p_lowpass: float = 0.4,
    lp_cutoff_min: float = 4000.0,
    lp_cutoff_max: float = 12000.0,
) -> Compose:
    """Pipeline agresivo OOD-aware. Simula degradaciones típicas de field
    recording crudo (audio lejano, mala mic, ruido natural).

    Diferencias con ``conservative``:
    - SNR 5–20 dB: mucho más ruido. Simula recording campo no-controlado.
    - Shift ±40%: más variación temporal.
    - Gain ±12 dB: simula distancia mayor al micrófono.
    - PitchShift ±2 semitones: variabilidad individual del ave.
    - TimeStretch 0.85–1.15: variación de tempo de canto.
    - LowPassFilter 4–12 kHz random cutoff: simula audio lejano o mic limitado.

    Diseñado para Iteración 2 Fase 5 (target hard acc ≥0.90).
    """
    return Compose(
        [
            AddGaussianSNR(min_snr_db=snr_db_min, max_snr_db=snr_db_max, p=p_noise),
            Shift(
                min_shift=-shift_max_fraction,
                max_shift=shift_max_fraction,
                shift_unit="fraction",
                rollover=True,
                p=p_shift,
            ),
            Gain(min_gain_db=gain_db_min, max_gain_db=gain_db_max, p=p_gain),
            PitchShift(min_semitones=-pitch_semitones, max_semitones=pitch_semitones, p=p_pitch),
            TimeStretch(min_rate=time_min_rate, max_rate=time_max_rate, p=p_time),
            LowPassFilter(min_cutoff_freq=lp_cutoff_min, max_cutoff_freq=lp_cutoff_max, p=p_lowpass),
        ]
    )


# Backward-compat: el alias original sigue apuntando al conservador para no
# romper código viejo (ej. precompute_embeddings.py sin --aug-pipeline arg).
def build_audio_pipeline(**kwargs) -> Compose:
    return build_audio_pipeline_conservative(**kwargs)
