"""Sube data/processed/embeddings.parquet a W&B como artifact ``embeddings``.

W&B versiona automáticamente: la 1ra subida queda como :v0, la 2da como :v1, etc.
Si el parquet contiene filas con ``is_aug=True`` el script detecta el K máximo y
agrega esa info al metadata + notes. Aliases adicionales se pasan con --alias.

Declara los artifacts de input (raw_audios:v0, test_hard:v0, metadata:v0) antes
de crear el nuevo, para que el grafo de dependencias quede registrado.

Uso:
    python scripts/upload_embeddings_to_wandb.py                          # dry-run
    python scripts/upload_embeddings_to_wandb.py --upload                 # sube (sin alias)
    python scripts/upload_embeddings_to_wandb.py --upload --alias aug-k2  # sube + alias
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
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Alias adicional para la nueva versión (repetible). Ej: --alias aug-k2",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not EMB_PATH.exists():
        print(f"! Falta {EMB_PATH}", file=sys.stderr)
        return 1

    df = pd.read_parquet(EMB_PATH)
    has_aug = "is_aug" in df.columns and bool(df["is_aug"].any())
    n_aug = int(df["is_aug"].sum()) if "is_aug" in df.columns else 0
    aug_k = int(df.loc[df.get("is_aug", False), "aug_id"].max()) if has_aug else 0

    print(f"artifact name: embeddings  (W&B autoversiona; alias extra={args.alias or '—'})")
    print(f"  rows: {len(df)}")
    print(f"  splits: {df.groupby('split').size().to_dict()}")
    print(f"  especies: {df['species'].nunique()}")
    print(f"  embedding dim: {len(df['embedding'].iloc[0])}")
    print(f"  tamaño: {EMB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    if has_aug:
        print(f"  aumentación detectada: K={aug_k}, filas aug={n_aug}")

    if not args.upload:
        print("\nDry-run.")
        return 0

    import wandb

    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()

    run_name = f"embeddings-aug-k{aug_k}" if has_aug else "embeddings-v0"
    notes = (
        f"BirdNET V2.4 (1024-dim) mean-pooled sobre ventanas de 3s. "
        f"Cubre 23 especies en train + 21 en test_hard."
        + (
            f" Augmentación waveform K={aug_k} (Gaussian SNR + Shift + Gain) "
            f"aplicada sólo al split=train; {n_aug} filas aumentadas."
            if has_aug
            else ""
        )
    )

    print(f"\nwandb.init(entity='{entity}', project='{project}')")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="embeddings",
        name=run_name,
        notes=notes,
    )

    print("Declarando inputs (lineage)...")
    for ref in ["raw_audios:v0", "test_hard:v0", "metadata:v0"]:
        try:
            run.use_artifact(ref)
            print(f"  use_artifact({ref}) OK")
        except wandb.errors.CommError as e:
            print(f"  ! use_artifact({ref}) falló: {e}", file=sys.stderr)

    print("\nConstruyendo artifact embeddings...")
    metadata: dict = {
        "model": "BirdNET_GLOBAL_6K_V2.4_FP32",
        "embedding_dim": 1024,
        "aggregation": "mean_pool_over_3sec_windows",
        "sample_rate": 48000,
        "num_rows": int(len(df)),
        "num_train": int((df["split"] == "train").sum()),
        "num_test_hard": int((df["split"] == "test_hard").sum()),
        "num_species": int(df["species"].nunique()),
    }
    if has_aug:
        metadata.update({
            "augmented": True,
            "aug_k": aug_k,
            "num_aug_rows": n_aug,
            "aug_pipeline": "AddGaussianSNR(15-30dB) + Shift(±30%) + Gain(±6dB)",
        })

    artifact = wandb.Artifact(
        name="embeddings",
        type="dataset",
        description=(
            "Embeddings BirdNET V2.4 (1024-dim) mean-pooled por audio. "
            "Schema: filepath, species, split (train/test_hard), embedding, is_aug, aug_id. "
            "split=test_hard NUNCA usar para training. "
            "is_aug=True sólo aparece en split=train."
        ),
        metadata=metadata,
    )
    artifact.add_file(str(EMB_PATH))
    run.log_artifact(artifact, aliases=args.alias or None)

    print("\nEsperando subida (~15 MB, rápido)...")
    run.finish()
    print(f"\nListo. Ver en https://wandb.ai/{entity}/{project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
