"""Augmentations de audio y de embeddings.

Dos niveles:

1. **Waveform** (antes de BirdNET): ``build_audio_pipeline()`` devuelve un
   ``audiomentations.Compose`` conservador (Gaussian SNR, Shift, Gain). Los
   embeddings aumentados se generan offline en
   ``scripts/precompute_embeddings.py --augment K`` y se persisten en
   ``data/processed/embeddings.parquet`` con la columna ``is_aug``.

2. **Embedding** (después de BirdNET): ``mixup`` está implementado en
   ``src/models/classifier.py``. Probado con resultado negativo en el run
   baseline; se mantiene el código pero la receta activa usa ``mixup_alpha=0``.
"""

from __future__ import annotations

from audiomentations import AddGaussianSNR, Compose, Gain, Shift


def build_audio_pipeline(
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
    """Pipeline conservador para aumentar waveforms de aves antes de BirdNET.

    Parámetros default:
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
