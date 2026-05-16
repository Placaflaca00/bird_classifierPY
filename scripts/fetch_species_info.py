"""Llena ``app/species_info.json`` con fotos + descripciones desde iNaturalist.

Para cada especie:
    1. Query a la iNat API v1 (sin auth) por nombre cientifico.
    2. Valida que el primer hit matchee el nombre cientifico exacto.
    3. Extrae ``wikipedia_summary`` (HTML) -> limpia tags -> ``description``.
    4. Baja ``default_photo.medium_url`` -> ``app/assets/birds/<species>.jpg``.
    5. Guarda atribucion (``photo_credit``, ``photo_license``).

Modos:
    python scripts/fetch_species_info.py             # dry-run (no toca disco)
    python scripts/fetch_species_info.py --apply     # baja fotos + escribe JSON

Si ``license_code`` viene null ("all rights reserved") la foto NO se baja: ese
caso requiere autorizacion individual del fotografo. Si ``wikipedia_summary``
viene vacio, ``description`` queda vacio (no inventamos contenido).

Ref: https://www.inaturalist.org/pages/api+reference
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SPECIES_INFO_PATH = ROOT / "app" / "species_info.json"
PHOTOS_DIR = ROOT / "app" / "assets" / "birds"

INAT_API = "https://api.inaturalist.org/v1"
USER_AGENT = (
    "birdClassifierPY/0.1 "
    "(https://github.com/Placaflaca00/bird_classifierPY)"
)
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 0.6  # ~100 req/min < iNat rate-limit (~60 req/min sostenido)

# Mapeo license_code de iNat -> nombre human-readable.
# `None` o no listado -> "all rights reserved" -> NO se baja.
_LICENSE_NAMES = {
    "cc0": "CC0",
    "pd": "Public Domain",
    "cc-by": "CC-BY",
    "cc-by-sa": "CC-BY-SA",
    "cc-by-nc": "CC-BY-NC",
    "cc-by-nc-sa": "CC-BY-NC-SA",
    "cc-by-nc-nd": "CC-BY-NC-ND",
    "cc-by-nd": "CC-BY-ND",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_html(s: str | None) -> str:
    """HTML -> texto plano: stripear tags, unescape entities, colapsar espacios."""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = html_mod.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_license(code: str | None) -> str | None:
    """license_code de iNat -> nombre. None si no es reutilizable."""
    if not code:
        return None
    return _LICENSE_NAMES.get(code.lower())


# ---------------------------------------------------------------------------
# Resultado por especie
# ---------------------------------------------------------------------------
@dataclass
class FetchReport:
    species: str
    matched: bool = False
    matched_name: str | None = None  # nombre que devolvio iNat (debug)
    description: str = ""
    photo_url: str | None = None
    photo_credit: str | None = None
    photo_license: str | None = None
    photo_license_raw: str | None = None  # license_code crudo
    skip_reason: str | None = None  # si no se va a bajar la foto
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Llamada a iNat
# ---------------------------------------------------------------------------
def fetch_one(session: requests.Session, sci_name: str) -> FetchReport:
    """Devuelve un FetchReport sin tocar disco."""
    report = FetchReport(species=sci_name)

    try:
        resp = session.get(
            f"{INAT_API}/taxa",
            params={
                "q": sci_name,
                "rank": "species",
                "locale": "es",
                "per_page": 5,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        report.errors.append(f"iNat HTTP error: {e}")
        return report
    except ValueError as e:
        report.errors.append(f"iNat respondio JSON invalido: {e}")
        return report

    results = data.get("results") or []
    if not results:
        report.errors.append("iNat no devolvio resultados")
        return report

    # Buscar match exacto entre los primeros 5; si no, tomar el primero.
    hit = next(
        (r for r in results if (r.get("name") or "").lower() == sci_name.lower()),
        results[0],
    )
    report.matched_name = hit.get("name")
    report.matched = (report.matched_name or "").lower() == sci_name.lower()
    if not report.matched:
        report.errors.append(
            f"primer match no exacto: pedimos '{sci_name}', iNat devolvio "
            f"'{report.matched_name}'"
        )

    # /taxa?q= devuelve un payload liviano sin wikipedia_summary. Hay que
    # ir a /taxa/{id} para el payload completo.
    taxon_id = hit.get("id")
    detail = hit
    if taxon_id:
        try:
            resp2 = session.get(
                f"{INAT_API}/taxa/{taxon_id}",
                params={"locale": "es"},
                timeout=REQUEST_TIMEOUT,
            )
            resp2.raise_for_status()
            data2 = resp2.json()
            results2 = data2.get("results") or []
            if results2:
                detail = results2[0]
        except (requests.RequestException, ValueError) as e:
            report.errors.append(f"detail por id fallo: {e}")

    report.description = _clean_html(detail.get("wikipedia_summary"))
    if not report.description:
        report.errors.append("sin wikipedia_summary en iNat")

    photo = hit.get("default_photo") or {}
    photo_url = photo.get("medium_url")
    raw_license = photo.get("license_code")
    license_name = _resolve_license(raw_license)
    attribution = photo.get("attribution")

    report.photo_url = photo_url
    report.photo_credit = attribution
    report.photo_license_raw = raw_license
    report.photo_license = license_name

    if not photo_url:
        report.skip_reason = "sin default_photo"
    elif license_name is None:
        report.skip_reason = (
            f"licencia no reutilizable ({raw_license or 'all rights reserved'})"
        )

    return report


# ---------------------------------------------------------------------------
# Bajar foto
# ---------------------------------------------------------------------------
def download_photo(
    session: requests.Session, url: str, dest: Path
) -> tuple[bool, str | None]:
    """Devuelve (ok, error_msg). Escribe a `dest` si ok."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return False, f"content-type inesperado: {ctype!r}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        if dest.stat().st_size < 1024:
            dest.unlink(missing_ok=True)
            return False, "foto bajada pesa < 1 KB"
        return True, None
    except requests.RequestException as e:
        return False, f"HTTP error: {e}"
    except OSError as e:
        return False, f"IO error: {e}"


# ---------------------------------------------------------------------------
# Procesamiento
# ---------------------------------------------------------------------------
def process_all(species_info: dict, apply: bool) -> list[FetchReport]:
    """Procesa las 20 especies. En dry-run solo arma reports."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    reports: list[FetchReport] = []
    items = list(species_info.items())
    n = len(items)

    for i, (sci, entry) in enumerate(items, 1):
        print(f"  [{i:>2}/{n}] {sci} ...", end=" ", flush=True)
        report = fetch_one(session, sci)
        reports.append(report)

        if apply and report.photo_url and not report.skip_reason:
            photo_rel = entry.get("photo") or f"assets/birds/{sci.lower().replace(' ', '_')}.jpg"
            dest = ROOT / "app" / photo_rel
            ok, err = download_photo(session, report.photo_url, dest)
            if not ok:
                report.errors.append(f"download falló: {err}")
                report.skip_reason = report.skip_reason or f"download falló: {err}"

        # Status linea
        tags = []
        tags.append("match-OK" if report.matched else "match-FUZZY")
        tags.append("wiki-OK" if report.description else "wiki-EMPTY")
        if report.skip_reason:
            tags.append(f"foto-SKIP({report.skip_reason})")
        else:
            tags.append(f"foto-OK[{report.photo_license}]")
        print(" ".join(tags))

        time.sleep(REQUEST_DELAY_SEC)

    return reports


def update_species_info(
    species_info_full: dict, reports: list[FetchReport]
) -> dict:
    """Devuelve el JSON actualizado (no escribe)."""
    species = species_info_full["species"]
    by_sci = {r.species: r for r in reports}

    for sci, entry in species.items():
        report = by_sci.get(sci)
        if not report or report.skip_reason and not report.description:
            continue
        if report.description:
            entry["description"] = report.description
        if not report.skip_reason and report.photo_url:
            # Mantenemos el path por defecto que ya esta en el JSON.
            entry["photo_credit"] = report.photo_credit
            entry["photo_license"] = report.photo_license

    return species_info_full


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------
def print_summary(reports: list[FetchReport]) -> None:
    n = len(reports)
    n_matched = sum(1 for r in reports if r.matched)
    n_desc = sum(1 for r in reports if r.description)
    n_photo = sum(1 for r in reports if r.photo_url and not r.skip_reason)
    print()
    print(f"Resumen: {n} especies procesadas")
    print(f"  match exacto         : {n_matched}/{n}")
    print(f"  con descripcion      : {n_desc}/{n}")
    print(f"  con foto reutilizable: {n_photo}/{n}")

    problemas = [r for r in reports if r.errors or r.skip_reason]
    if problemas:
        print(f"\n  {len(problemas)} especies con observaciones:")
        for r in problemas:
            msgs = list(r.errors)
            if r.skip_reason:
                msgs.append(f"foto skip: {r.skip_reason}")
            for m in msgs:
                print(f"    - {r.species}: {m}")

    # Distribucion de licencias
    licenses = {}
    for r in reports:
        if r.photo_license and not r.skip_reason:
            licenses[r.photo_license] = licenses.get(r.photo_license, 0) + 1
    if licenses:
        print("\n  Licencias de fotos:")
        for lic, count in sorted(licenses.items(), key=lambda x: -x[1]):
            print(f"    {lic}: {count}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Baja fotos a app/assets/birds/ y escribe el JSON (default: dry-run)",
    )
    args = parser.parse_args()

    if not SPECIES_INFO_PATH.exists():
        print(f"no existe {SPECIES_INFO_PATH}", file=sys.stderr)
        return 2

    with open(SPECIES_INFO_PATH, encoding="utf-8") as f:
        species_info_full = json.load(f)
    species_info = species_info_full.get("species", {})
    if not species_info:
        print("species_info.json no tiene la clave 'species'", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Modo: {mode}")
    print(f"Especies a procesar: {len(species_info)}")
    print(f"Endpoint: {INAT_API}")
    print()

    reports = process_all(species_info, apply=args.apply)
    print_summary(reports)

    if args.apply:
        updated = update_species_info(species_info_full, reports)
        with open(SPECIES_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        print(f"\nJSON actualizado: {SPECIES_INFO_PATH}")
    else:
        print("\nDry-run: no se toco disco. Re-ejecutar con --apply para aplicar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
