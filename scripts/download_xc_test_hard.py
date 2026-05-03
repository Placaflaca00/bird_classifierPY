"""Descarga audios D/E de Xeno-canto para construir el test_hard set.

Las 23 especies salen de data/raw/metadata.parquet. Para cada una se busca
en XC API v3 con el filtro q:"<C" (= ratings D y E, peor calidad que C).

Salida:
    data/test_sets/xc_hard/<species_folder>/XC<id>.mp3
    data/test_sets/xc_hard/metadata.parquet

Uso:
    python scripts/download_xc_test_hard.py --survey       # solo consulta, no baja
    python scripts/download_xc_test_hard.py --download     # baja archivos
    python scripts/download_xc_test_hard.py --download --max-per-species 30

NUNCA mezclar este directorio con data/raw/. Es un test set aislado.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_METADATA = ROOT / "data" / "raw" / "metadata.parquet"
OUT_DIR = ROOT / "data" / "test_sets" / "xc_hard"
OUT_METADATA = OUT_DIR / "metadata.parquet"

XC_BASE = "https://xeno-canto.org/api/3/recordings"
THROTTLE_SEC = 0.3
DOWNLOAD_TIMEOUT = 60
ALLOWED_RATINGS = {"D", "E"}


def species_list_from_metadata() -> list[str]:
    df = pd.read_parquet(RAW_METADATA)
    return sorted(df["species"].unique().tolist())


def species_to_folder(species: str) -> str:
    """Rhea americana -> rhea_americana."""
    return species.lower().replace(" ", "_")


def query_xc(species: str, key: str, page: int = 1) -> dict:
    """Una página de resultados D/E para la especie."""
    parts = species.split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"species inválida: {species!r}")
    gen, sp = parts
    query = f'gen:{gen} sp:{sp} q:"<C" grp:birds'
    params = {
        "query": query,
        "key": key,
        "per_page": 500,
        "page": page,
    }
    r = requests.get(XC_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_refs(species: str, key: str) -> list[dict]:
    """Junta todas las páginas de una query en una lista de recordings."""
    all_recs: list[dict] = []
    page = 1
    while True:
        payload = query_xc(species, key, page=page)
        if "error" in payload:
            print(f"  ! API error: {payload['error']}")
            return []
        recs = payload.get("recordings", [])
        all_recs.extend(recs)
        num_pages = int(payload.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(THROTTLE_SEC)
    return all_recs


def survey(species_list: list[str], key: str) -> dict[str, dict]:
    """Para cada especie devuelve {total: N, by_rating: {D: a, E: b}}."""
    summary: dict[str, dict] = {}
    for sp in species_list:
        try:
            recs = collect_refs(sp, key)
        except requests.HTTPError as e:
            print(f"  ! {sp}: HTTP {e.response.status_code}")
            summary[sp] = {"total": 0, "by_rating": {}, "error": str(e)}
            continue
        ratings: dict[str, int] = {}
        for r in recs:
            q = r.get("q", "?")
            ratings[q] = ratings.get(q, 0) + 1
        summary[sp] = {"total": len(recs), "by_rating": ratings}
        time.sleep(THROTTLE_SEC)
    return summary


def print_survey(summary: dict[str, dict]) -> None:
    print(f"{'species':<28} {'total':>6} {'D':>4} {'E':>4} {'others':>8}")
    print("-" * 56)
    grand_total = grand_d = grand_e = 0
    for sp, info in sorted(summary.items()):
        ratings = info["by_rating"]
        d = ratings.get("D", 0)
        e = ratings.get("E", 0)
        others = info["total"] - d - e
        marker = "" if (d + e) > 0 else "  <-- 0 D/E"
        print(f"{sp:<28} {info['total']:>6} {d:>4} {e:>4} {others:>8}  {marker}")
        grand_total += info["total"]
        grand_d += d
        grand_e += e
    print("-" * 56)
    print(f"{'TOTAL':<28} {grand_total:>6} {grand_d:>4} {grand_e:>4}")


def filename_for(rec: dict, species_folder: str) -> Path:
    rec_id = rec["id"]
    return OUT_DIR / species_folder / f"XC{rec_id}.mp3"


def download_one(rec: dict, dest: Path) -> bool:
    """Devuelve True si quedó un archivo válido en disco."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    file_url = rec.get("file", "")
    if not file_url:
        print(f"    ! XC{rec.get('id')}: sin file URL (¿especie restringida?)")
        return False
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    try:
        with requests.get(file_url, stream=True, timeout=DOWNLOAD_TIMEOUT,
                          allow_redirects=True) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        return True
    except requests.RequestException as e:
        print(f"    ! XC{rec.get('id')}: {e}")
        return False


def probe_audio(path: Path) -> tuple[float | None, int | None]:
    """ffprobe -> (duration_s, sample_rate). None si falla."""
    import json
    import subprocess
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, timeout=30, check=True,
        )
        data = json.loads(out.stdout.decode("utf-8", errors="replace"))
        return float(data["format"]["duration"]), int(data["streams"][0]["sample_rate"])
    except Exception:
        return None, None


def download_all(species_list: list[str], key: str, max_per_species: int) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for sp in species_list:
        folder = species_to_folder(sp)
        print(f"\n=== {sp} ({folder}) ===")
        try:
            recs = collect_refs(sp, key)
        except requests.HTTPError as e:
            print(f"  ! API error: {e}")
            continue

        de_recs = [r for r in recs if r.get("q") in ALLOWED_RATINGS]
        de_recs = de_recs[:max_per_species]
        print(f"  candidatos D/E: {len(de_recs)}")

        for rec in de_recs:
            dest = filename_for(rec, folder)
            ok = download_one(rec, dest)
            if not ok:
                continue
            duration, sr = probe_audio(dest)
            if duration is None or sr is None:
                print(f"    ! ffprobe falló: {dest.name}")
                dest.unlink(missing_ok=True)
                continue
            rows.append({
                "filepath": f"{folder}/{dest.name}",
                "species": sp,
                "rating": rec.get("q", "no-score"),
                "source": "xenocanto",
                "duration_seconds": duration,
                "sample_rate": sr,
            })
            time.sleep(THROTTLE_SEC)

    if not rows:
        print("\n! No se descargó nada.")
        return 1

    df = pd.DataFrame(rows)
    schema = pa.DataFrameSchema(
        {
            "filepath": pa.Column(str, checks=pa.Check.str_endswith(".mp3")),
            "species": pa.Column(str),
            "rating": pa.Column(str, checks=pa.Check.isin(ALLOWED_RATINGS)),
            "source": pa.Column(str, checks=pa.Check.eq("xenocanto")),
            "duration_seconds": pa.Column(float, checks=pa.Check.gt(0.5)),
            "sample_rate": pa.Column(int, checks=pa.Check.gt(0)),
        },
        strict=True,
        coerce=True,
    )
    print("\nValidando con Pandera...")
    df = schema.validate(df, lazy=True)
    df.to_parquet(OUT_METADATA, index=False)
    print(f"OK. {len(df)} filas escritas en {OUT_METADATA}")
    print(f"\nDistribución por especie:")
    print(df.groupby("species").size().to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--survey", action="store_true",
                   help="Solo consultar XC, no descargar")
    g.add_argument("--download", action="store_true",
                   help="Descargar archivos")
    parser.add_argument("--max-per-species", type=int, default=30)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("XENOCANTO_API_KEY", "").strip()
    if not key:
        print("! Falta XENOCANTO_API_KEY en .env", file=sys.stderr)
        return 1

    if not RAW_METADATA.exists():
        print(f"! Falta {RAW_METADATA}", file=sys.stderr)
        return 1

    species = species_list_from_metadata()
    print(f"Especies target: {len(species)}")

    if args.survey:
        summary = survey(species, key)
        print()
        print_survey(summary)
        return 0

    return download_all(species, key, args.max_per_species)


if __name__ == "__main__":
    raise SystemExit(main())
