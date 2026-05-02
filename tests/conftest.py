"""Fixtures compartidas de pytest.

Fixtures planeadas:
- ``tmp_metadata_df``  : DataFrame mínimo válido contra ``MetadataSchema``.
- ``fake_embeddings``  : ndarray (N=16, 320) con valores aleatorios reproducibles.
- ``fake_labels``      : ndarray (N=16,) con clases entre 0 y K-1.
- ``trained_classifier``: ``BirdClassifier`` con pesos random (sin entrenar)
                          para tests de forward/inferencia.
- ``audio_clip_3s``    : waveform sintético 48 kHz × 3 s para tests de pipeline.

TODO: implementar fixtures.
"""
