"""Tests del subpaquete ``src.data``.

Cobertura objetivo:
- ``schemas.MetadataSchema`` valida un DataFrame correcto.
- ``schemas.MetadataSchema`` rechaza filas con rating inválido / duración negativa.
- ``BirdEmbeddingsDataset.__len__`` y ``__getitem__`` devuelven shapes esperadas.
- ``augmentations.mixup`` preserva la suma de pesos (= 1) en y_mixed.

TODO: implementar tests.
"""
