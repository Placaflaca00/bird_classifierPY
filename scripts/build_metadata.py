"""Construye data/raw/metadata.parquet consolidado y validado.

Fuentes:
    - data/raw/metadata.parquet      (xeno-canto, 1068 filas)
    - data/raw/metadata_gbif.parquet (gbif, 1553 filas con duracion/SR=0)
    - data/raw/<species>/*.mp3       (2335 archivos en disco - VERDAD)

Salida (6 columnas, schema validado con Pandera):
    filepath, species, rating, source, duration_seconds, sample_rate

Lógica:
    1. Walk disco -> lista canónica de archivos.
    2. Para cada archivo, lookup en parquets existentes por filepath.
    3. Re-probar duration + sample_rate vía ffprobe (autoritativo, los GBIF
       tienen 0 que no sirven).
    4. Construir DataFrame con folder name -> "Genus species" como species.
    5. Validar con Pandera (estricto: lista cerrada de ratings/sources).
    6. Si hay archivos corruptos (ffprobe falla), reportar y opcionalmente
       borrar (--delete-corrupt).
    7. Guardar como data/raw/metadata.parquet (sobrescribe).

Uso:
    python scripts/build_metadata.py             # dry-run, no escribe
    python scripts/build_metadata.py --apply     # escribe metadata.parquet
    python scripts/build_metadata.py --apply --delete-corrupt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "metadata.parquet"
GBIF_PATH = RAW_DIR / "metadata_gbif.parquet"
XC_PATH = RAW_DIR / "metadata.parquet"
WA_CSV = RAW_DIR / "wikiaves_metadata.csv"

ALLOWED_RATINGS = {"A", "B", "no-score"}
ALLOWED_SOURCES = {"gbif", "xenocanto", "wikiaves"}


@dataclass
class Probe:
    filepath: str
    duration: float | None
    sample_rate: int | None
    error: str | None = None


def probe_audio(filepath: str) -> Probe:
    """Devuelve duración (s) y sample_rate (Hz) vía ffprobe."""
    abs_path = RAW_DIR / filepath
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate:format=duration",
                "-of", "json",
                str(abs_path),
            ],
            capture_output=True, timeout=30, check=True,
        )
        data = json.loads(out.stdout.decode("utf-8", errors="replace"))
        duration = float(data["format"]["duration"])
        sr = int(data["streams"][0]["sample_rate"])
        return Probe(filepath, duration, sr)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        return Probe(filepath, None, None, error=type(e).__name__)


def folder_to_species(folder: str) -> str:
    """rhea_americana -> Rhea americana."""
    parts = folder.split("_")
    return " ".join([parts[0].capitalize(), *parts[1:]])


def collect_disk_files(raw_dir: Path) -> list[str]:
    """Lista relativa: '<species_folder>/<file>.mp3'."""
    files: list[str] = []
    for sp_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for f in sorted(sp_dir.glob("*.mp3")):
            files.append(f"{sp_dir.name}/{f.name}")
    return files


def build_lookup(parquets: list[Path], wa_csv: Path | None = None) -> dict[str, dict]:
    """Mergea parquets existentes (XC, GBIF) y opcionalmente CSV de WikiAves
    en un dict filepath -> {rating, source}."""
    lookup: dict[str, dict] = {}
    for p in parquets:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            fp = row["filepath"]
            if fp not in lookup:
                lookup[fp] = {
                    "rating": str(row.get("rating", "no-score")),
                    "source": str(row.get("source", "unknown")),
                }
    if wa_csv is not None and wa_csv.exists():
        wa = pd.read_csv(wa_csv)
        for _, row in wa.iterrows():
            fp = row["filepath"]
            if fp not in lookup:
                lookup[fp] = {"rating": "no-score", "source": "wikiaves"}
    return lookup


def make_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "filepath": pa.Column(
                str,
                checks=[
                    pa.Check.str_endswith(".mp3"),
                    pa.Check(lambda s: s.is_unique, error="filepath must be unique"),
                ],
                nullable=False,
            ),
            "species": pa.Column(str, nullable=False),
            "rating": pa.Column(
                str,
                checks=pa.Check.isin(ALLOWED_RATINGS),
                nullable=False,
            ),
            "source": pa.Column(
                str,
                checks=pa.Check.isin(ALLOWED_SOURCES),
                nullable=False,
            ),
            "duration_seconds": pa.Column(
                float,
                checks=pa.Check.gt(0.5),
                nullable=False,
            ),
            "sample_rate": pa.Column(
                int,
                checks=pa.Check.gt(0),
                nullable=False,
            ),
        },
        strict=True,
        coerce=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Escribir metadata.parquet (default: dry-run)")
    parser.add_argument("--delete-corrupt", action="store_true",
                        help="Borrar archivos donde ffprobe falla")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    disk_files = collect_disk_files(RAW_DIR)
    print(f"Archivos en disco: {len(disk_files)}")

    lookup = build_lookup([XC_PATH, GBIF_PATH], wa_csv=WA_CSV)
    print(f"Lookup combinado (XC+GBIF+WA): {len(lookup)} filepaths")
    missing = [f for f in disk_files if f not in lookup]
    if missing:
        print(f"  ! {len(missing)} archivos sin metadata:")
        for f in missing[:5]:
            print(f"    - {f}")

    print(f"\nProbando audio (ffprobe x {args.workers} workers)...")
    probes: dict[str, Probe] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(probe_audio, f): f for f in disk_files}
        for fut in as_completed(futures):
            p = fut.result()
            probes[p.filepath] = p
            done += 1
            if done % 200 == 0 or done == len(disk_files):
                print(f"  {done}/{len(disk_files)}", file=sys.stderr)

    corrupt = [p for p in probes.values() if p.error]
    if corrupt:
        print(f"\n! {len(corrupt)} archivos corruptos (ffprobe falló):")
        for p in corrupt[:10]:
            print(f"  {p.filepath}  -> {p.error}")
        if args.delete_corrupt:
            for p in corrupt:
                try:
                    (RAW_DIR / p.filepath).unlink()
                    print(f"  borrado: {p.filepath}")
                except OSError as e:
                    print(f"  ! falló borrar {p.filepath}: {e}")
            disk_files = [f for f in disk_files if probes[f].error is None]
        else:
            print("  (re-ejecutar con --delete-corrupt para borrarlos)")
            return 1
    else:
        print("\nSin archivos corruptos.")

    rows = []
    for fp in disk_files:
        probe = probes[fp]
        if probe.error:
            continue
        meta = lookup.get(fp, {"rating": "no-score", "source": "unknown"})
        species_folder = fp.split("/", 1)[0]
        rows.append({
            "filepath": fp,
            "species": folder_to_species(species_folder),
            "rating": meta["rating"],
            "source": meta["source"],
            "duration_seconds": probe.duration,
            "sample_rate": probe.sample_rate,
        })
    df = pd.DataFrame(rows)

    print(f"\nDataFrame construido: {len(df)} filas, {df.shape[1]} columnas")
    print(f"Especies: {df['species'].nunique()}")
    print(f"Sources:\n{df['source'].value_counts().to_string()}")
    print(f"Ratings:\n{df['rating'].value_counts().to_string()}")
    print(f"Duration (s): min={df['duration_seconds'].min():.1f}, "
          f"max={df['duration_seconds'].max():.1f}, "
          f"mean={df['duration_seconds'].mean():.1f}")
    print(f"Sample rate: {sorted(df['sample_rate'].unique())}")

    print("\nValidando con Pandera...")
    schema = make_schema()
    try:
        df = schema.validate(df, lazy=True)
        print("OK: schema válido.")
    except pa.errors.SchemaErrors as e:
        print("! Validación falló:")
        print(e.failure_cases.to_string())
        return 1

    if args.apply:
        df.to_parquet(OUT_PATH, index=False)
        print(f"\nEscrito: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    else:
        print("\nDry-run: no se escribió nada. Re-ejecutar con --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
