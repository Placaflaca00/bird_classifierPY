"""CLI: descarga audios desde GBIF (occurrence/search) para una lista de especies.

GBIF agrega Xeno-canto, Macaulay Library, iNaturalist, eBird, etc.
Este script usa el endpoint sincrónico ``/occurrence/search`` (NO requiere auth).

Uso típico (Paraguay primero, fallback a Sudamérica):

    python scripts/download_gbif.py \\
        --species-list configs/species_paraguay.txt \\
        --country PY \\
        --min-per-species 10 \\
        --fallback-continent SOUTH_AMERICA \\
        --max-per-species 100 \\
        --out data/raw/

Comportamiento:
1. Resuelve cada nombre científico a su ``taxonKey`` GBIF (``/species/match``).
2. Cuenta registros con ``mediaType=Sound`` filtrados por ``--country``.
3. Si la cuenta es < ``--min-per-species``, reintenta con ``--fallback-continent``
   (o sin filtro geográfico si el fallback está vacío).
4. Itera resultados, extrae URLs de audio (puede haber > 1 audio por registro)
   y descarga hasta ``--max-per-species`` audios totales por especie.
5. Genera ``<out>/metadata.parquet`` con un row por audio bajado.

Bandera ``--dry-run``: solo cuenta y reporta, no descarga nada.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.data.download import download_recording, make_resilient_session
from src.data.gbif import (
    count_occurrences,
    extract_audio_media,
    guess_extension,
    match_species,
    occurrence_to_metadata,
    search_occurrences,
)

logger = logging.getLogger("gbif_download")


FAILED_URLS_FILENAME = ".failed_urls.txt"


def slugify(name: str) -> str:
    return "_".join(name.lower().strip().split())


def load_failed_urls(out_dir: Path) -> set[str]:
    """Carga URLs que fallaron en runs previos para no reintentarlas."""
    path = out_dir / FAILED_URLS_FILENAME
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def record_failed_url(out_dir: Path, url: str) -> None:
    """Append-only: registra una URL que acaba de fallar."""
    path = out_dir / FAILED_URLS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--species-list",
        type=Path,
        default=Path("configs/species_paraguay.txt"),
    )
    p.add_argument(
        "--country",
        default="PY",
        help="Código ISO 3166 alpha-2 del país (PY, AR, BR, ...). '' = sin filtro.",
    )
    p.add_argument(
        "--fallback-continent",
        default="SOUTH_AMERICA",
        choices=[
            "",
            "AFRICA",
            "ANTARCTICA",
            "ASIA",
            "EUROPE",
            "NORTH_AMERICA",
            "OCEANIA",
            "SOUTH_AMERICA",
        ],
        help="Continente a usar si --country no alcanza ('' = saltar este nivel).",
    )
    p.add_argument(
        "--no-global-fallback",
        action="store_true",
        help="Desactiva el tercer nivel de fallback (búsqueda global sin filtro geo).",
    )
    p.add_argument(
        "--media-type",
        default="Sound",
        choices=["Sound", "StillImage", "MovingImage"],
    )
    p.add_argument(
        "--min-per-species",
        type=int,
        default=10,
        help="Si el país tiene menos que esto, hace fallback al continente.",
    )
    p.add_argument(
        "--max-per-species",
        type=int,
        default=100,
        help="Tope de audios por especie (0 = sin tope).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
    )
    p.add_argument(
        "--metadata-name",
        default="metadata_gbif.parquet",
        help="Nombre del parquet (separado del de XC para no pisar).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def read_species_list(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"species list not found: {path}")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    if not out:
        sys.exit(f"species list is empty: {path}")
    return out


def resolve_filter_for_species(
    *,
    taxon_key: int,
    country: str | None,
    fallback_continent: str | None,
    fallback_global: bool,
    media_type: str,
    min_per_species: int,
    session: requests.Session,
) -> tuple[int, dict]:
    """Decide el filtro a usar (country → continent → global).

    Sube de nivel solo si el actual no alcanza ``min_per_species``.
    Si ningún nivel alcanza, devuelve el que tenga más records.
    """
    candidates: list[tuple[int, dict]] = []  # [(n, filter_kwargs), ...]

    if country:
        n = count_occurrences(
            taxon_key=taxon_key, country=country, media_type=media_type, session=session
        )
        candidates.append((n, {"country": country, "continent": None}))
        if n >= min_per_species:
            return candidates[-1]

    if fallback_continent:
        n = count_occurrences(
            taxon_key=taxon_key,
            continent=fallback_continent,
            media_type=media_type,
            session=session,
        )
        candidates.append((n, {"country": None, "continent": fallback_continent}))
        if n >= min_per_species:
            return candidates[-1]

    if fallback_global:
        n = count_occurrences(taxon_key=taxon_key, media_type=media_type, session=session)
        candidates.append((n, {"country": None, "continent": None}))
        # No retornamos temprano: queremos elegir el mayor entre todos.

    if not candidates:
        return 0, {"country": None, "continent": None}
    return max(candidates, key=lambda c: c[0])


def filter_label(filter_kwargs: dict) -> str:
    if filter_kwargs.get("country"):
        return f"country:{filter_kwargs['country']}"
    if filter_kwargs.get("continent"):
        return f"continent:{filter_kwargs['continent']}"
    return "global"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    species_list = read_species_list(args.species_list)
    country = args.country or None
    fallback_continent = args.fallback_continent or None
    fallback_global = not args.no_global_fallback

    logger.info(
        "species=%d country=%r fallback_continent=%r fallback_global=%s media_type=%s "
        "min_per_species=%d max_per_species=%d dry_run=%s",
        len(species_list),
        country,
        fallback_continent,
        fallback_global,
        args.media_type,
        args.min_per_species,
        args.max_per_species,
        args.dry_run,
    )

    session = make_resilient_session()

    summary: list[dict] = []
    metadata_rows: list[dict] = []
    failed_urls = load_failed_urls(args.out)
    if failed_urls:
        logger.info("loaded %d previously-failed URLs to skip", len(failed_urls))

    try:
        run_loop(
            args,
            species_list,
            session,
            summary,
            metadata_rows,
            country,
            fallback_continent,
            fallback_global,
            failed_urls,
        )
    finally:
        flush_results(args, summary, metadata_rows)

    return 0


def run_loop(
    args,
    species_list,
    session,
    summary,
    metadata_rows,
    country,
    fallback_continent,
    fallback_global,
    failed_urls: set[str],
):
    for species in species_list:
        # 1. Resolver taxonKey
        try:
            match = match_species(species, session=session)
        except requests.RequestException as e:
            logger.error("match error for %s: %s", species, e)
            summary.append(
                {
                    "species": species,
                    "taxon_key": None,
                    "available": -1,
                    "downloaded": 0,
                    "source": "MATCH_ERROR",
                }
            )
            continue
        if not match:
            logger.warning("no GBIF match for %s", species)
            summary.append(
                {
                    "species": species,
                    "taxon_key": None,
                    "available": 0,
                    "downloaded": 0,
                    "source": "NO_MATCH",
                }
            )
            continue
        taxon_key = match["usageKey"]

        # 2. Decidir filtro (país vs continente)
        try:
            n_total, filter_kwargs = resolve_filter_for_species(
                taxon_key=taxon_key,
                country=country,
                fallback_continent=fallback_continent,
                fallback_global=fallback_global,
                media_type=args.media_type,
                min_per_species=args.min_per_species,
                session=session,
            )
        except requests.RequestException as e:
            logger.error("count error for %s: %s", species, e)
            summary.append(
                {
                    "species": species,
                    "taxon_key": taxon_key,
                    "available": -1,
                    "downloaded": 0,
                    "source": "COUNT_ERROR",
                }
            )
            continue

        source_label = filter_label(filter_kwargs)
        cap = args.max_per_species if args.max_per_species > 0 else None
        will_download = 0 if args.dry_run else min(n_total, cap or n_total)
        logger.info(
            "%-28s taxon_key=%-10s available=%5d  source=%-22s  will_download=%d",
            species,
            taxon_key,
            n_total,
            source_label,
            will_download,
        )

        if args.dry_run or n_total == 0:
            summary.append(
                {
                    "species": species,
                    "taxon_key": taxon_key,
                    "available": n_total,
                    "downloaded": 0,
                    "source": source_label,
                }
            )
            continue

        # 3. Iterar y bajar
        species_dir = args.out / slugify(species)
        species_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        try:
            for occ in tqdm(
                iter_capped(
                    search_occurrences(
                        taxon_key=taxon_key,
                        media_type=args.media_type,
                        session=session,
                        **filter_kwargs,
                    ),
                    cap,
                ),
                total=cap or n_total,
                desc=species,
                unit="occ",
            ):
                if cap is not None and downloaded >= cap:
                    break
                for media_idx, media in enumerate(extract_audio_media(occ)):
                    if cap is not None and downloaded >= cap:
                        break
                    media_url = media["url"]
                    if media_url in failed_urls:
                        continue
                    ext = guess_extension(media_url, media["format"])
                    occ_key = occ.get("key", "unknown")
                    suffix = "" if media_idx == 0 else f"_{media_idx}"
                    dest = species_dir / f"GBIF{occ_key}{suffix}.{ext}"
                    try:
                        download_recording(media_url, dest, session=session)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "GBIF%s failed (recording URL added to skip list): %s", occ_key, e
                        )
                        record_failed_url(args.out, media_url)
                        failed_urls.add(media_url)
                        continue
                    relpath = dest.relative_to(args.out).as_posix()
                    metadata_rows.append(occurrence_to_metadata(occ, media, relpath, species))
                    downloaded += 1
        except requests.RequestException as e:
            logger.error("search error for %s mid-iter: %s", species, e)

        summary.append(
            {
                "species": species,
                "taxon_key": taxon_key,
                "available": n_total,
                "downloaded": downloaded,
                "source": source_label,
            }
        )
        logger.info("done %s: downloaded=%d", species, downloaded)


def flush_results(args, summary: list[dict], metadata_rows: list[dict]) -> None:
    """Imprime resumen y persiste metadata.parquet. Se llama siempre, incluso si crashea."""
    if summary:
        summary_df = pd.DataFrame(summary)
        print("\n=== SUMMARY ===")
        print(summary_df.to_string(index=False))
        print(
            f"\nTotal available={summary_df['available'].clip(lower=0).sum()}  "
            f"downloaded={summary_df['downloaded'].sum()}"
        )

    if not args.dry_run and metadata_rows:
        args.out.mkdir(parents=True, exist_ok=True)
        meta_df = pd.DataFrame(metadata_rows)
        out_meta = args.out / args.metadata_name
        meta_df.to_parquet(out_meta, index=False)
        logger.info("metadata saved → %s (%d rows)", out_meta, len(meta_df))


def iter_capped(it: Iterable[dict], cap: int | None) -> Iterable[dict]:
    if cap is None:
        yield from it
        return
    for i, x in enumerate(it):
        if i >= cap:
            return
        yield x


if __name__ == "__main__":
    sys.exit(main())
