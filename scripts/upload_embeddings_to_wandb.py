"""Sube data/processed/embeddings.parquet como embeddings:v0 con lineage.

Declara los artifacts de input (raw_audios:v0, test_hard:v0, metadata:v0)
antes de crear el nuevo, para que el grafo de dependencias quede registrado.

Uso:
    python scripts/upload_embeddings_to_wandb.py             # dry-run
    python scripts/upload_embeddings_to_wandb.py --upload    # sube
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not EMB_PATH.exists():
        print(f"! Falta {EMB_PATH}", file=sys.stderr)
        return 1

    df = pd.read_parquet(EMB_PATH)
    print(f"embeddings:v0 (artifact type=dataset)")
    print(f"  rows: {len(df)}")
    print(f"  splits: {df.groupby('split').size().to_dict()}")
    print(f"  especies: {df['species'].nunique()}")
    print(f"  embedding dim: {len(df['embedding'].iloc[0])}")
    print(f"  tamaño: {EMB_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    if not args.upload:
        print("\nDry-run.")
        return 0

    import wandb

    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()

    print(f"\nwandb.init(entity='{entity}', project='{project}')")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="embeddings",
        name="embeddings-v0",
        notes=(
            "BirdNET V2.4 (1024-dim) mean-pooled sobre ventanas de 3s. "
            "Cubre 23 especies en train + 21 en test_hard."
        ),
    )

    print("Declarando inputs (lineage)...")
    for ref in ["raw_audios:v0", "test_hard:v0", "metadata:v0"]:
        try:
            run.use_artifact(ref)
            print(f"  use_artifact({ref}) OK")
        except wandb.errors.CommError as e:
            print(f"  ! use_artifact({ref}) falló: {e}", file=sys.stderr)

    print("\nConstruyendo artifact embeddings...")
    artifact = wandb.Artifact(
        name="embeddings",
        type="dataset",
        description=(
            "Embeddings BirdNET V2.4 (1024-dim) mean-pooled por audio. "
            "Schema: filepath, species, split (train/test_hard), embedding (1024 floats). "
            "split=test_hard NUNCA usar para training."
        ),
        metadata={
            "model": "BirdNET_GLOBAL_6K_V2.4_FP32",
            "embedding_dim": 1024,
            "aggregation": "mean_pool_over_3sec_windows",
            "sample_rate": 48000,
            "num_rows": int(len(df)),
            "num_train": int((df["split"] == "train").sum()),
            "num_test_hard": int((df["split"] == "test_hard").sum()),
            "num_species": int(df["species"].nunique()),
        },
    )
    artifact.add_file(str(EMB_PATH))
    run.log_artifact(artifact)

    print("\nEsperando subida (~15 MB, rápido)...")
    run.finish()
    print(f"\nListo. Ver en https://wandb.ai/{entity}/{project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
