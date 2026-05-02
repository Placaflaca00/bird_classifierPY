"""CLI: pre-cálculo de embeddings BirdNET sobre todo el dataset.

Misma lógica del notebook 02 pero ejecutable como script reproducible.

Uso:
    python scripts/precompute_embeddings.py \\
        --raw-dir data/raw/ \\
        --out-dir data/processed/ \\
        --tflite-model models/birdnet.tflite

Salidas en ``data/processed/``:
- ``embeddings.npy``  shape (N, 320), dtype float32.
- ``labels.npy``      shape (N,),     dtype int64.
- ``index.parquet``   mapeo i → metadata.

TODO: implementar.
"""
