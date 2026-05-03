"""Sube data/test_sets/xc_hard/ a W&B como artifact test_hard:v0.

A diferencia de raw_audios + metadata (que viven en artifacts separados porque
metadata se actualiza independientemente), el test set es un único artifact
inmutable que incluye audio + metadata.parquet juntos. Bajar uno = bajar todo.

Uso:
    python scripts/upload_test_hard_to_wandb.py             # dry-run
    python scripts/upload_test_hard_to_wandb.py --upload    # sube
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test_sets" / "xc_hard"
METADATA_PATH = TEST_DIR / "metadata.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true",
                        help="Subir a W&B (default: dry-run)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not METADATA_PATH.exists():
        print(f"! Falta {METADATA_PATH}", file=sys.stderr)
        print("    Corré primero: python scripts/download_xc_test_hard.py --download")
        return 1

    df = pd.read_parquet(METADATA_PATH)
    audio_files = list(TEST_DIR.rglob("*.mp3"))
    audio_size = sum(f.stat().st_size for f in audio_files)

    print(f"test_hard:v0 (artifact type=dataset, único)")
    print(f"  archivos audio: {len(audio_files)}")
    print(f"  filas metadata: {len(df)}")
    print(f"  especies:       {df['species'].nunique()}")
    print(f"  ratings:")
    for rating, count in df["rating"].value_counts().items():
        print(f"    {rating}: {count}")
    print(f"  tamaño audio:   {audio_size / 1024 / 1024:.1f} MB")
    print(f"  metadata:       {METADATA_PATH.stat().st_size / 1024:.1f} KB")

    if len(audio_files) != len(df):
        print(f"\n! WARNING: archivos en disco ({len(audio_files)}) != filas metadata ({len(df)})")

    if not args.upload:
        print("\nDry-run: no se subió nada. Re-ejecutar con --upload.")
        return 0

    import wandb

    project = os.environ.get("WANDB_PROJECT", "").strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if not project or not entity:
        print("! Falta WANDB_PROJECT o WANDB_ENTITY en .env", file=sys.stderr)
        return 1

    print(f"\nwandb.init(entity='{entity}', project='{project}')")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="data-ingest",
        name="test-hard-v0",
        notes=(
            "Test set difícil: audios D/E de xeno-canto, "
            "21 especies (Eudromia y Phoenicopterus sin D/E disponibles). "
            "NUNCA usar para entrenar."
        ),
    )

    artifact = wandb.Artifact(
        name="test_hard",
        type="dataset",
        description=(
            "Test set HARD: audios xeno-canto rating D/E (peor calidad), "
            "para evaluar robustez del modelo en condiciones reales. "
            "NUNCA tocar para training. Metadata schema = mismo que raw_audios."
        ),
        metadata={
            "num_files": int(len(df)),
            "num_species": int(df["species"].nunique()),
            "rating_distribution": {k: int(v) for k, v in
                                    df["rating"].value_counts().to_dict().items()},
            "warning": "test_only_never_train",
            "missing_species": ["Eudromia formosa", "Phoenicopterus chilensis"],
        },
    )

    artifact.add_dir(str(TEST_DIR))
    run.log_artifact(artifact)
    print("  log_artifact(test_hard) -> upload en background")

    print("\nEsperando subida (~200 MB, unos minutos)...")
    run.finish()
    print(f"\nListo. Ver en https://wandb.ai/{entity}/{project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
