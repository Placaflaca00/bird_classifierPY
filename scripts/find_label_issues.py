"""CLI: detecta etiquetas posiblemente erróneas con Cleanlab.

Pasos:
1. Cargar embeddings + labels + el mejor modelo entrenado.
2. Obtener probas out-of-fold (cross-val).
3. Pasar a ``cleanlab.filter.find_label_issues(...)``.
4. Exportar a ``data/processed/label_issues.parquet`` con columnas:
   ``filepath, current_label, suggested_label, confidence``.

Uso:
    python scripts/find_label_issues.py \\
        --embeddings data/processed/embeddings.npy \\
        --labels data/processed/labels.npy \\
        --out data/processed/label_issues.parquet

TODO: implementar.
"""
