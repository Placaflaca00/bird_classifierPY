"""Pandera schemas para validación de metadata.

Define el schema para ``metadata.parquet`` (un row por clip de audio):

- ``filepath``         : str — path relativo a ``data/raw/`` o ``data/processed/``.
- ``species``          : str — nombre científico (e.g. ``Turdus rufiventris``).
- ``rating``           : str — calidad Xeno-canto (A, B, C, D, E).
- ``source``           : str — origen del clip (``xenocanto`` | ``own_recording``).
- ``duration_seconds`` : float — duración del clip en segundos.
- ``sample_rate``      : int — frecuencia de muestreo (esperado 48000 o 44100).

Validaciones a aplicar:
- ``species`` ∈ lista permitida (especies de Paraguay).
- ``rating`` ∈ {"A", "B", "C", "D", "E"}.
- ``duration_seconds > 0``.
- ``sample_rate ∈ {44100, 48000}``.

TODO: implementar ``MetadataSchema`` (pa.DataFrameSchema) y exportar helper ``validate(df)``.
"""
