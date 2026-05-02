"""CLI: descarga curada desde Xeno-canto.

Wrapper one-shot sobre ``src.data.download``.

Uso:
    python scripts/download_xenocanto.py --species-list configs/species_paraguay.txt \\
        --quality A,B --out data/raw/

Genera:
- ``data/raw/<species>/<recording_id>.mp3``
- ``data/raw/metadata.parquet``

TODO: implementar parser de argumentos (argparse o typer) y logging a stdout.
"""
