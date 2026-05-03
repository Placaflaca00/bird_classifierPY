"""Sube data/raw/ a W&B Artifacts: raw_audios:v0 + metadata:v0.

Crea un único run de tipo 'data-ingest' que loggea dos artifacts:
    - raw_audios (type=dataset): los .mp3 organizados por <species>/<file>.mp3
    - metadata (type=dataset): data/raw/metadata.parquet (canónico, no el _gbif)

Uso:
    python scripts/upload_data_to_wandb.py             # dry-run (lista qué subiría)
    python scripts/upload_data_to_wandb.py --upload    # sube de verdad

Requiere en .env (o env vars del proceso):
    WANDB_API_KEY
    WANDB_PROJECT
    WANDB_ENTITY
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
METADATA_PATH = RAW_DIR / "metadata.parquet"


def collect_audio_by_species(raw_dir: Path) -> dict[str, list[Path]]:
    species: dict[str, list[Path]] = {}
    for sp_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = sorted(sp_dir.glob("*.mp3"))
        if files:
            species[sp_dir.name] = files
    return species


def report(species: dict[str, list[Path]], metadata_path: Path) -> None:
    total_files = sum(len(v) for v in species.values())
    total_bytes = sum(f.stat().st_size for files in species.values() for f in files)
    print(f"raw_audios:v0 (artifact type=dataset)")
    print(f"  especies: {len(species)}")
    print(f"  archivos: {total_files} mp3")
    print(f"  tamaño:   {total_bytes / 1024 / 1024:.1f} MB")
    print()
    print(f"metadata:v0 (artifact type=dataset)")
    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
        print(f"  archivo: {metadata_path.relative_to(ROOT)}")
        print(f"  filas:   {len(df)}")
        print(f"  cols:    {list(df.columns)}")
        print(f"  tamaño:  {metadata_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"  ! NO ENCONTRADO: {metadata_path}")


def upload(species: dict[str, list[Path]], metadata_path: Path) -> int:
    import wandb

    project = os.environ.get("WANDB_PROJECT", "").strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if not project or not entity:
        print("! Falta WANDB_PROJECT o WANDB_ENTITY en .env", file=sys.stderr)
        return 1

    print(f"wandb.init(entity='{entity}', project='{project}')")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="data-ingest",
        name="raw-audio-v0",
        notes="Snapshot inicial: 23 especies, 2335 mp3, metadata consolidado",
    )

    print("\nConstruyendo artifact raw_audios...")
    audio_artifact = wandb.Artifact(
        name="raw_audios",
        type="dataset",
        description=(
            "Audios crudos descargados de GBIF (1267) + xeno-canto (1068). "
            "23 especies de aves de Paraguay. Estructura: <species_folder>/<file>.mp3."
        ),
        metadata={
            "num_species": len(species),
            "num_files": sum(len(v) for v in species.values()),
            "format": "mp3",
            "sources": ["gbif", "xenocanto"],
        },
    )
    for sp_name in species:
        audio_artifact.add_dir(str(RAW_DIR / sp_name), name=sp_name)
    run.log_artifact(audio_artifact)
    print("  log_artifact(raw_audios) -> upload en background")

    print("\nConstruyendo artifact metadata...")
    meta_artifact = wandb.Artifact(
        name="metadata",
        type="dataset",
        description=(
            "metadata.parquet consolidado: filepath, species, rating, source, "
            "duration_seconds, sample_rate. Validado con Pandera (ver scripts/build_metadata.py)."
        ),
        metadata={
            "schema": ["filepath", "species", "rating", "source",
                       "duration_seconds", "sample_rate"],
            "validated_by": "pandera",
        },
    )
    meta_artifact.add_file(str(metadata_path))
    run.log_artifact(meta_artifact)
    print("  log_artifact(metadata) -> upload en background")

    print("\nEsperando que termine la subida (puede tardar 15-30 min para ~2.3 GB)...")
    run.finish()
    print("\nListo. Revisá en https://wandb.ai/{}/{}".format(entity, project))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true",
                        help="Subir a W&B (default: dry-run sin subir)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not RAW_DIR.exists():
        print(f"! No existe {RAW_DIR}", file=sys.stderr)
        return 1

    species = collect_audio_by_species(RAW_DIR)
    if not species:
        print(f"! No se encontraron .mp3 en {RAW_DIR}", file=sys.stderr)
        return 1

    report(species, METADATA_PATH)

    if not METADATA_PATH.exists():
        print(f"\n! Falta {METADATA_PATH}. Corre primero:")
        print("    python scripts/build_metadata.py --apply")
        return 1

    if not args.upload:
        print("\nDry-run: no se subió nada. Re-ejecutar con --upload para aplicar.")
        return 0

    return upload(species, METADATA_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
