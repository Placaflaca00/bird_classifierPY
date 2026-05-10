"""Sube splits.parquet como splits:v0 con lineage hacia embeddings:v0.

Uso:
    python scripts/upload_splits_to_wandb.py             # dry-run
    python scripts/upload_splits_to_wandb.py --upload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--alias", action="append", default=[],
                        help="Alias adicional para esta versión (repetible). Ej: --alias wa-drop3")
    parser.add_argument("--embeddings-artifact", default="embeddings:v0",
                        help="Artifact de embeddings al que linkea para lineage. Default: embeddings:v0")
    parser.add_argument("--run-name", default=None,
                        help="Override run name. Default: splits-<aliases> o splits-v0")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not SPLITS_PATH.exists():
        print(f"! Falta {SPLITS_PATH}", file=sys.stderr)
        return 1

    df = pd.read_parquet(SPLITS_PATH)
    counts = df["fold"].value_counts().to_dict()
    print(f"splits:v0 (artifact type=dataset)")
    print(f"  filas: {len(df)}")
    print(f"  folds: {counts}")
    print(f"  tamaño: {SPLITS_PATH.stat().st_size/1024:.1f} KB")

    if not args.upload:
        print("\nDry-run.")
        return 0

    import wandb

    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()
    run_name = args.run_name or (f"splits-{'-'.join(args.alias)}" if args.alias else "splits-v0")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="splits",
        name=run_name,
        notes="Tiered split por especie: 70/15/15 + mínimos según N. seed=42.",
    )
    run.use_artifact(args.embeddings_artifact)
    print(f"  use_artifact({args.embeddings_artifact}) OK")

    artifact = wandb.Artifact(
        name="splits",
        type="dataset",
        description=(
            "Folds train/val/test_clean/test_hard. Tiered por especie: "
            "N≥50 -> 70/15/15; N≥20 -> min 3 val/test; N≥10 -> min 2 val/test; "
            "N<10 -> todo a train. seed=42, reproducible."
        ),
        metadata={
            "seed": 42,
            "policy": "tiered_per_species",
            "fold_counts": {k: int(v) for k, v in counts.items()},
            "embeddings_artifact": args.embeddings_artifact,
        },
    )
    artifact.add_file(str(SPLITS_PATH))
    run.log_artifact(artifact, aliases=args.alias or None)
    run.finish()
    print(f"\nListo. https://wandb.ai/{entity}/{project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
