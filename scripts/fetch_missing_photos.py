"""Llena fotos faltantes desde Wikimedia Commons.

Para cada especie sin `photo_credit` en ``app/species_info.json``:
    1. Pide el summary de es.wikipedia.org para el nombre cientifico.
    2. Extrae el filename del Commons desde el ``originalimage`` / ``thumbnail``.
    3. Pide ``imageinfo|extmetadata`` a Commons -> licencia + autor.
    4. Baja un thumbnail de ~640 px de ancho (suficiente para card + retina).
    5. Actualiza ``species_info.json`` con ``photo_credit`` + ``photo_license``.

Modos:
    python scripts/fetch_missing_photos.py             # dry-run
    python scripts/fetch_missing_photos.py --apply     # baja + escribe JSON

Wikimedia Commons es por defecto CC/PD; aun asi validamos la licencia con
extmetadata y skipeamos si viene null o "all rights reserved".
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
from urllib.parse import quote, unquote

import requests

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SPECIES_INFO_PATH = ROOT / "app" / "species_info.json"

USER_AGENT = (
    "birdClassifierPY/0.1 "
    "(https://github.com/Placaflaca00/bird_classifierPY)"
)
WIKI_LANGS = ["es", "en"]  # intenta es primero, fallback a en
THUMB_WIDTH = 640          # px de ancho del thumb a bajar
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 0.6

# Licencias aceptables (lowercased substring match en LicenseShortName)
_OK_LICENSES = (
    "cc0", "public domain", "pd",
    "cc by 1.0", "cc by 2.0", "cc by 3.0", "cc by 4.0",
    "cc by-sa 1.0", "cc by-sa 2.0", "cc by-sa 2.5", "cc by-sa 3.0", "cc by-sa 4.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = html_mod.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _is_acceptable_license(short_name: str | None) -> bool:
    if not short_name:
        return False
    sn = short_name.lower().strip()
    return any(ok in sn for ok in _OK_LICENSES)


def _extract_commons_filename(url: str) -> str | None:
    """De una URL upload.wikimedia.org -> filename (sin prefijo File:).

    Maneja tanto URLs originales como thumbs:
      .../commons/x/yy/Filename.jpg
      .../commons/thumb/x/yy/Filename.jpg/640px-Filename.jpg
    """
    if not url:
        return None
    url = url.split("?")[0]
    if "/commons/thumb/" in url:
        rest = url.split("/commons/thumb/", 1)[1]
        parts = rest.split("/")
        # parts: ["x", "yy", "<Filename>", "<width>px-..."]
        if len(parts) >= 3:
            return unquote(parts[2])
    elif "/commons/" in url:
        rest = url.split("/commons/", 1)[1]
        parts = rest.split("/")
        if parts:
            return unquote(parts[-1])
    return None


def _build_thumb_url(filename: str, width: int = THUMB_WIDTH) -> str:
    """Reconstruye una URL de thumb a width custom desde el filename.

    Wikimedia pattern: si el filename original es 'Foo.jpg', el hash del path
    es md5(filename)[:2] -> primer char + primeros 2 chars. La API REST suele
    devolvernos el thumb_url ya con el hash; pero si solo tenemos el filename,
    armamos la URL via Special:FilePath que sirve un redirect.
    """
    return (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        f"{quote(filename, safe='')}?width={width}"
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
@dataclass
class PhotoReport:
    species: str
    wiki_title: str | None = None
    wiki_lang: str | None = None
    filename: str | None = None
    license: str | None = None
    artist: str | None = None
    thumb_url: str | None = None
    skip_reason: str | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wikipedia + Commons
# ---------------------------------------------------------------------------
def _wiki_summary(session: requests.Session, sci_name: str) -> tuple[dict, str] | None:
    """Devuelve (summary_json, lang). Prueba es primero, luego en."""
    for lang in WIKI_LANGS:
        url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(sci_name.replace(' ', '_'), safe='')}"
        )
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            if data.get("type") == "disambiguation":
                continue
            if (data.get("thumbnail") or {}).get("source") or (
                data.get("originalimage") or {}
            ).get("source"):
                return data, lang
        except (requests.RequestException, ValueError):
            continue
    return None


def _commons_imageinfo(
    session: requests.Session, filename: str
) -> tuple[str | None, str | None]:
    """Devuelve (license_short, artist_text) para File:<filename> en Commons."""
    try:
        r = session.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "titles": f"File:{filename}",
                "prop": "imageinfo",
                "iiprop": "extmetadata",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None, None

    pages = (data.get("query") or {}).get("pages") or {}
    if not pages:
        return None, None
    page = next(iter(pages.values()), {})
    iinfo_list = page.get("imageinfo") or []
    if not iinfo_list:
        return None, None
    meta = iinfo_list[0].get("extmetadata") or {}
    license_short = (meta.get("LicenseShortName") or {}).get("value")
    artist_html = (meta.get("Artist") or {}).get("value", "")
    artist = _strip_html(artist_html)
    return license_short, artist


# ---------------------------------------------------------------------------
# Por especie
# ---------------------------------------------------------------------------
def fetch_one(session: requests.Session, sci_name: str) -> PhotoReport:
    report = PhotoReport(species=sci_name)
    res = _wiki_summary(session, sci_name)
    if not res:
        report.errors.append("Wikipedia no devolvio summary con foto en es/en")
        report.skip_reason = "sin foto en Wikipedia"
        return report
    summary, lang = res
    report.wiki_title = summary.get("title")
    report.wiki_lang = lang

    src = (summary.get("originalimage") or {}).get("source") or (
        summary.get("thumbnail") or {}
    ).get("source")
    filename = _extract_commons_filename(src or "")
    if not filename:
        report.errors.append(f"no pude parsear filename de la URL: {src}")
        report.skip_reason = "filename no parseable"
        return report
    report.filename = filename

    license_short, artist = _commons_imageinfo(session, filename)
    report.license = license_short
    report.artist = artist
    if not _is_acceptable_license(license_short):
        report.skip_reason = f"licencia no aceptable ({license_short!r})"
        return report

    report.thumb_url = _build_thumb_url(filename, THUMB_WIDTH)
    return report


def download_photo(
    session: requests.Session, url: str, dest: Path
) -> tuple[bool, str | None]:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return False, f"content-type inesperado: {ctype!r}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
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
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Baja fotos y actualiza JSON (default: dry-run)")
    args = parser.parse_args()

    if not SPECIES_INFO_PATH.exists():
        print(f"no existe {SPECIES_INFO_PATH}", file=sys.stderr)
        return 2

    with open(SPECIES_INFO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    species = data.get("species") or {}

    missing = [(sci, info) for sci, info in species.items()
               if not (info.get("photo_credit") or "").strip()]
    if not missing:
        print("Nada para hacer: todas las especies tienen photo_credit.")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Modo: {mode}")
    print(f"Especies sin foto: {len(missing)}")
    for sci, _ in missing:
        print(f"  - {sci}")
    print()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    reports: list[PhotoReport] = []
    for i, (sci, info) in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {sci} ...", end=" ", flush=True)
        report = fetch_one(session, sci)
        reports.append(report)
        if report.skip_reason:
            print(f"SKIP ({report.skip_reason})")
        else:
            print(f"OK [{report.wiki_lang}] {report.license!r} - {report.filename}")
            if args.apply:
                photo_rel = info.get("photo") or f"assets/birds/{sci.lower().replace(' ', '_')}.jpg"
                dest = ROOT / "app" / photo_rel
                ok, err = download_photo(session, report.thumb_url, dest)
                if not ok:
                    print(f"      download fail: {err}")
                    report.skip_reason = f"download fail: {err}"
                    report.errors.append(f"download fail: {err}")
                else:
                    print(f"      bajado a {dest.relative_to(ROOT)}  ({dest.stat().st_size//1024} KB)")
        time.sleep(REQUEST_DELAY_SEC)

    # Resumen
    ok = [r for r in reports if r.thumb_url and not r.skip_reason]
    skipped = [r for r in reports if r.skip_reason]
    print()
    print(f"Resumen: {len(ok)} fotos resueltas, {len(skipped)} skipeadas")
    for r in skipped:
        print(f"  - {r.species}: {r.skip_reason}")
    for r in ok:
        print(f"  + {r.species}: {r.license} (por {r.artist or 'autor desconocido'})")

    if args.apply:
        # Update JSON
        for r in reports:
            if r.thumb_url and not r.skip_reason:
                entry = species.get(r.species, {})
                credit_parts = []
                if r.artist:
                    credit_parts.append(r.artist)
                credit_parts.append(f"via Wikimedia Commons ({r.license})")
                entry["photo_credit"] = " - ".join(credit_parts)
                entry["photo_license"] = r.license
                species[r.species] = entry
        with open(SPECIES_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nJSON actualizado: {SPECIES_INFO_PATH}")
    else:
        print("\nDry-run: no se toco disco. Re-ejecutar con --apply para aplicar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
