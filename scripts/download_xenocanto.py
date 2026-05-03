"""CLI: descarga curada desde Xeno-canto (API v3) para una lista de especies.

REQUISITO: necesitás una API key de Xeno-canto.
1. Registrate en https://xeno-canto.org/account
2. Verificá tu email.
3. Copiá tu key desde la página de cuenta.
4. Guardala en `.env` como ``XENOCANTO_API_KEY=<tu-key>`` (o pasala con ``--api-key``).

Uso típico (Paraguay, calidad >= B, máx 100 por especie):

    python scripts/download_xenocanto.py \\
        --species-list configs/species_paraguay.txt \\
        --country Paraguay \\
        --min-quality B \\
        --min-per-species 10 \\
        --fallback-area america \\
        --max-per-species 100 \\
        --out data/raw/

Comportamiento:
1. Para cada especie del archivo, primero busca en ``--country``.
2. Si la cantidad de grabaciones encontradas es menor a ``--min-per-species``,
   reintenta sin el filtro de país y con ``area:--fallback-area`` (e.g. america).
3. Descarga hasta ``--max-per-species`` (0 = sin tope) a
   ``<out>/<genus_species>/XC<id>.mp3``.
4. Genera ``<out>/metadata.parquet`` con un row por grabación bajada.

Bandera ``--dry-run``: solo cuenta y reporta, no descarga nada
(útil para decidir si conviene ampliar el filtro).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.data.download import (
    build_query,
    count_recordings,
    download_recording,
    make_resilient_session,
    min_quality_to_xc,
    recording_to_metadata,
    search_recordings,
    slugify,
)

logger = logging.getLogger("xenocanto_download")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--species-list",
        type=Path,
        default=Path("configs/species_paraguay.txt"),
        help="Archivo de texto con un nombre científico por línea (ignora # y vacías).",
    )
    p.add_argument(
        "--country",
        default="Paraguay",
        help="Filtro principal de país (use '' para deshabilitar).",
    )
    p.add_argument(
        "--min-quality",
        default="B",
        choices=["A", "B", "C", "D", "E"],
        help="Calidad mínima Xeno-canto (A es la mejor).",
    )
    p.add_argument(
        "--min-per-species",
        type=int,
        default=10,
        help="Si una especie tiene menos que esto en --country, hace fallback a --fallback-area.",
    )
    p.add_argument(
        "--fallback-area",
        default="america",
        choices=["", "africa", "america", "asia", "australia", "europe"],
        help="Área a usar si el país no alcanza ('' deshabilita el fallback).",
    )
    p.add_argument(
        "--max-per-species",
        type=int,
        default=100,
        help="Tope de descargas por especie (0 = sin tope).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Directorio raíz donde guardar audios y metadata.parquet.",
    )
    p.add_argument(
        "--metadata-name",
        default="metadata.parquet",
        help="Nombre del parquet de metadata dentro de --out.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Override de la API key (por defecto se lee de XENOCANTO_API_KEY en el env / .env).",
    )
    p.add_argument(
        "--skip-xc-ids-from",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path a un parquet con columna xc_id o xenocanto_id; los IDs ahí "
            "encontrados se saltan en este run (dedup cross-source). "
            "Repetible: --skip-xc-ids-from a.parquet --skip-xc-ids-from b.parquet"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo reportar conteos por especie; no descarga ni escribe nada.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_xc_ids_to_skip(paths: list[Path]) -> set[str]:
    """Cargar IDs de Xeno-canto ya bajados (de parquets previos) para evitar duplicados.

    Acepta tanto la columna ``xenocanto_id`` (escrita por este script) como
    ``xc_id`` (escrita por ``download_gbif.py``).
    """
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            logger.warning("skip-xc-ids-from: %s no existe; ignorando", path)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("no pude leer %s: %s", path, e)
            continue
        for col in ("xenocanto_id", "xc_id"):
            if col in df.columns:
                ids = df[col].astype(str).str.strip()
                seen.update(i for i in ids if i and i != "nan")
        logger.info("cargados %d IDs únicos desde %s", len(seen), path)
    return seen


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


def resolve_query_for_species(
    species: str,
    *,
    api_key: str,
    country: str | None,
    fallback_area: str | None,
    quality_filter: str | None,
    min_per_species: int,
    session: requests.Session,
) -> tuple[str, int, str]:
    """Decide qué query usar para una especie y devuelve (query, num_records, source_label)."""
    primary = build_query(species, country=country, quality_filter=quality_filter)
    n_primary, _ = count_recordings(primary, key=api_key, session=session)
    if n_primary >= min_per_species or not fallback_area:
        return primary, n_primary, country or "all"
    fallback = build_query(species, area=fallback_area, quality_filter=quality_filter)
    n_fb, _ = count_recordings(fallback, key=api_key, session=session)
    if n_fb > n_primary:
        return fallback, n_fb, f"area:{fallback_area}"
    return primary, n_primary, country or "all"


def resolve_api_key(cli_value: str | None) -> str:
    """Devuelve la API key desde --api-key, env, o .env."""
    if cli_value:
        return cli_value
    load_dotenv(override=False)  # carga .env si existe; no pisa el env real
    key = os.environ.get("XENOCANTO_API_KEY", "").strip()
    if not key:
        sys.exit(
            "ERROR: no se encontró XENOCANTO_API_KEY.\n"
            "  - Registrate en https://xeno-canto.org/account\n"
            "  - Copiá tu key y guardala en `.env` como XENOCANTO_API_KEY=<tu-key>\n"
            "  - O pasala por --api-key <tu-key>"
        )
    return key


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = resolve_api_key(args.api_key)
    species_list = read_species_list(args.species_list)
    quality_filter = min_quality_to_xc(args.min_quality)
    country = args.country or None
    fallback_area = args.fallback_area or None
    skip_ids = load_xc_ids_to_skip(args.skip_xc_ids_from)

    logger.info(
        "species=%d country=%r fallback_area=%r min_quality=%s "
        "min_per_species=%d max_per_species=%d skip_ids=%d dry_run=%s",
        len(species_list),
        country,
        fallback_area,
        args.min_quality,
        args.min_per_species,
        args.max_per_species,
        len(skip_ids),
        args.dry_run,
    )

    session = make_resilient_session()

    summary: list[dict] = []
    metadata_rows: list[dict] = []

    try:
        run_loop(
            args,
            species_list,
            session,
            api_key,
            country,
            fallback_area,
            quality_filter,
            skip_ids,
            summary,
            metadata_rows,
        )
    finally:
        flush_results(args, summary, metadata_rows)

    return 0


def run_loop(
    args,
    species_list,
    session,
    api_key,
    country,
    fallback_area,
    quality_filter,
    skip_ids: set[str],
    summary,
    metadata_rows,
):
    for species in species_list:
        try:
            query, n_total, source_label = resolve_query_for_species(
                species,
                api_key=api_key,
                country=country,
                fallback_area=fallback_area,
                quality_filter=quality_filter,
                min_per_species=args.min_per_species,
                session=session,
            )
        except requests.RequestException as e:
            logger.error("API error for %s: %s", species, e)
            summary.append(
                {
                    "species": species,
                    "available": -1,
                    "downloaded": 0,
                    "skipped_dup": 0,
                    "source": "ERROR",
                }
            )
            continue

        cap = args.max_per_species if args.max_per_species > 0 else None
        will_download = 0 if args.dry_run else min(n_total, cap or n_total)
        logger.info(
            "%-28s available=%4d  source=%-15s  will_download=%d  query=%s",
            species,
            n_total,
            source_label,
            will_download,
            query,
        )

        if args.dry_run or n_total == 0:
            summary.append(
                {
                    "species": species,
                    "available": n_total,
                    "downloaded": 0,
                    "skipped_dup": 0,
                    "source": source_label,
                }
            )
            continue

        species_dir = args.out / slugify(species)
        species_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        skipped_dup = 0
        try:
            for rec in tqdm(
                iter_capped(search_recordings(query, key=api_key, session=session), cap),
                total=cap or n_total,
                desc=species,
                unit="rec",
            ):
                xc_id = str(rec.get("id", ""))
                file_url = rec.get("file") or ""
                if not file_url or not xc_id:
                    continue
                if xc_id in skip_ids:
                    skipped_dup += 1
                    continue
                dest = species_dir / f"XC{xc_id}.mp3"
                try:
                    download_recording(file_url, dest, session=session)
                except Exception as e:  # noqa: BLE001
                    logger.error("XC%s failed: %s", xc_id, e)
                    continue
                relpath = dest.relative_to(args.out).as_posix()
                metadata_rows.append(recording_to_metadata(rec, relpath))
                downloaded += 1
        except requests.RequestException as e:
            logger.error("search error for %s mid-iter: %s", species, e)

        summary.append(
            {
                "species": species,
                "available": n_total,
                "downloaded": downloaded,
                "skipped_dup": skipped_dup,
                "source": source_label,
            }
        )
        logger.info("done %s: downloaded=%d skipped_dup=%d", species, downloaded, skipped_dup)


def flush_results(args, summary: list[dict], metadata_rows: list[dict]) -> None:
    """Imprime resumen y persiste metadata.parquet. Se llama siempre, incluso si crashea."""
    if summary:
        summary_df = pd.DataFrame(summary)
        print("\n=== SUMMARY ===")
        print(summary_df.to_string(index=False))
        total_avail = summary_df["available"].clip(lower=0).sum()
        total_dl = summary_df["downloaded"].sum()
        total_skip = (
            summary_df.get("skipped_dup", pd.Series(dtype=int)).sum()
            if "skipped_dup" in summary_df.columns
            else 0
        )
        print(f"\nTotal available={total_avail}  downloaded={total_dl}  skipped_dup={total_skip}")

    if not args.dry_run and metadata_rows:
        args.out.mkdir(parents=True, exist_ok=True)
        meta_df = pd.DataFrame(metadata_rows)
        out_meta = args.out / args.metadata_name
        meta_df.to_parquet(out_meta, index=False)
        logger.info("metadata saved → %s (%d rows)", out_meta, len(meta_df))

    return 0


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
