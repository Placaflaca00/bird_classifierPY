"""Augmentations de audio y de embeddings.

Dos niveles:

1. **Augmentations sobre waveform** (aplicadas ANTES de extraer embeddings con BirdNET):
   - ``AddGaussianNoise``
   - ``TimeShift``
   - ``PitchShift``
   - ``Gain``
   - ``BackgroundNoise`` (e.g. lluvia, viento de field-recordings).
   Se construyen con ``audiomentations.Compose([...])``.

2. **Augmentations sobre embeddings** (aplicadas DESPUÉS de extraer):
   - ``mixup`` con ``alpha`` configurable: combina linealmente dos embeddings
     y dos one-hot labels.

TODO: exportar ``build_audio_pipeline(cfg)`` y ``mixup(x, y, alpha)``.
"""
