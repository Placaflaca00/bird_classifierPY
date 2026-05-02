"""Descarga curada desde Xeno-canto.

Cliente para la API pública de Xeno-canto (``https://www.xeno-canto.org/api/2/recordings``).

Funcionalidad esperada:
- ``query_xenocanto(species, country="Paraguay", quality="A")`` → lista de metadatos.
- ``download_recordings(records, out_dir)`` → baja los .mp3 con ``requests`` + retry.
- Genera ``metadata.parquet`` con las columnas definidas en ``src.data.schemas``.
- Filtra por:
    * país = Paraguay (o vecinos si la especie es transfronteriza).
    * rating ∈ {A, B} por defecto (configurable).
    * duración mínima/máxima.

TODO: implementar el cliente con manejo de rate limits y resumen post-descarga.
"""
