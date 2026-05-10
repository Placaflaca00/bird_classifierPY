"""Descarga audios de WikiAves para especies específicas.

Endpoint paginado:
    https://www.wikiaves.com.br/getRegistrosJSON.php?tm=s&t=s&s={id}&o=mp&p={page}

Cada item tiene:
    - id        : WikiAves record id (entero como string).
    - link      : URL de thumbnail JPG en S3. El MP3 se obtiene reemplazando
                  ".jpg" por ".mp3" (técnica replicada de github.com/Athospd/wikiaves).
    - autor, data, local, dura, sp.{id,nome,nvt}.

Filename de salida: ``data/raw/<species_slug>/WA<wa_id>.mp3`` (paralelo al
formato ``GBIF<id>.mp3`` que ya usamos para GBIF).

Metadata: ``data/raw/wikiaves_metadata.csv`` con (filepath, species, wa_id,
autor, fecha, local, dura). Después se mergea con metadata.parquet via
build_metadata.py.

Dry-run por default. Usar --apply para descargar de verdad.

Uso:
    python scripts/download_wikiaves.py                          # dry-run, las 3 sp
    python scripts/download_wikiaves.py --apply                  # descarga real
    python scripts/download_wikiaves.py --species "Aburria jacutinga" --apply --limit 1
                                                                  # smoke (1 archivo)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
META_CSV = RAW_DIR / "wikiaves_metadata.csv"
OUT_DIR_REPORTS = ROOT / "reports" / "wikiaves_analysis"

REGISTROS_URL = "https://www.wikiaves.com.br/getRegistrosJSON.php"
PAGE_SIZE = 18  # observado: cada página devuelve hasta 18 items
SLEEP_BETWEEN_REQUESTS = 0.6  # cortesía con el servidor
SLEEP_BETWEEN_DOWNLOADS = 0.4

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
HEADERS_API = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*",
    "Referer": "https://www.wikiaves.com.br/",
}
HEADERS_DL = {
    "User-Agent": UA,
    "Accept": "audio/mpeg, audio/*, */*",
    "Referer": "https://www.wikiaves.com.br/",
    "Accept-Encoding": "identity",  # evitar que el servidor mande gzip de un mp3 binario
}

# Mapping: especie en WA -> (especie tradicional/interna, folder slug)
# Aburria jacutinga es la nomenclatura WA; en nuestro dataset la llamamos
# "Pipile jacutinga" (folder pipile_jacutinga/) por compatibilidad histórica.
SPECIES_MAP: dict[str, tuple[int, str, str]] = {
    # WA name             -> (species_id, internal_name, folder_slug)
    "Aburria jacutinga":   (10065, "Pipile jacutinga", "pipile_jacutinga"),
    "Jabiru mycteria":     (10182, "Jabiru mycteria",  "jabiru_mycteria"),
    "Rhea americana":      (10001, "Rhea americana",   "rhea_americana"),
}


def jpg_to_mp3_url(link: str) -> str:
    """Convierte URL de thumbnail JPG a URL del MP3 real.

    El JSON de WikiAves pone el hash precedido de '#' (que el navegador trata
    como fragment y no envía al servidor). El filename físico en S3 NO tiene
    ese '#' — solo '_<hash>.mp3'. Verificado parseando la página HTML del
    registro (tag <audio src="...">). La técnica del repo R Athospd/wikiaves
    (solo cambiar extensión) ya no funciona contra este CDN.
    """
    if not link:
        return ""
    # Sacar el '#' del fragment (lo que parecía hash es parte del filename real)
    cleaned = link.replace("#", "")
    if cleaned.lower().endswith(".jpg"):
        return cleaned[:-4] + ".mp3"
    if ".jpg" in cleaned.lower():
        idx = cleaned.lower().rfind(".jpg")
        return cleaned[:idx] + ".mp3" + cleaned[idx + 4:]
    return cleaned


def fetch_page(species_id: int, page: int, session: requests.Session,
               retries: int = 3, timeout: int = 45) -> dict:
    params = {"tm": "s", "t": "s", "s": species_id, "o": "mp", "p": page}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(REGISTROS_URL, params=params, headers=HEADERS_API, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            wait = 2 ** attempt  # 1, 2, 4 s
            print(f"  ... fetch_page p={page} attempt {attempt+1}/{retries} failed ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def collect_items(species_id: int, session: requests.Session, max_pages: int | None = None) -> list[dict]:
    """Pagina hasta consumir todos los items o llegar a max_pages."""
    all_items: list[dict] = []
    total: int | None = None
    page = 1
    while True:
        try:
            data = fetch_page(species_id, page, session)
        except Exception as e:
            print(f"  ! page {page}: {type(e).__name__}: {e}", file=sys.stderr)
            break
        reg = data.get("registros", {}) if isinstance(data, dict) else {}
        if total is None:
            total = int(reg.get("total", 0) or 0)
        itens_obj = reg.get("itens", {})
        if isinstance(itens_obj, dict):
            page_items = list(itens_obj.values())
        elif isinstance(itens_obj, list):
            page_items = itens_obj
        else:
            page_items = []
        if not page_items:
            break
        all_items.extend(page_items)
        if total and len(all_items) >= total:
            break
        if max_pages and page >= max_pages:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return all_items


def safe_filename(wa_id: str | int) -> str:
    return f"WA{wa_id}.mp3"


def download_one(mp3_url: str, dest: Path, session: requests.Session,
                 retries: int = 3) -> tuple[bool, str | None]:
    """Devuelve (ok, error_str). Si ok, dest existe."""
    last_err: str | None = None
    for attempt in range(retries):
        try:
            with session.get(mp3_url, headers=HEADERS_DL, timeout=60, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if "html" in ct.lower() or "json" in ct.lower():
                    return False, f"unexpected Content-Type: {ct!r}"
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                if tmp.stat().st_size < 1024:
                    tmp.unlink(missing_ok=True)
                    return False, f"file too small ({tmp.stat().st_size if tmp.exists() else 0} bytes)"
                tmp.replace(dest)
            return True, None
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    return False, last_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Descargar de verdad. Sin esto, dry-run (lista pero no escribe).")
    parser.add_argument("--species", default=None,
                        help="Filtrar a una sola especie (formato WA, ej. 'Aburria jacutinga').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo N items por especie (smoke test).")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limitar paginación (debug).")
    args = parser.parse_args()

    if args.species and args.species not in SPECIES_MAP:
        print(f"--species debe ser uno de: {list(SPECIES_MAP.keys())}", file=sys.stderr)
        return 2

    targets = {args.species: SPECIES_MAP[args.species]} if args.species else SPECIES_MAP
    print(f"Especies target: {len(targets)}")
    for wa_name, (sid, internal, slug) in targets.items():
        print(f"  - WA={wa_name!r}  internal={internal!r}  slug={slug!r}  id={sid}")
    print(f"Mode: {'APPLY (descarga real)' if args.apply else 'DRY-RUN (sin escribir)'}")
    if args.limit:
        print(f"Limit por especie: {args.limit}")

    session = requests.Session()
    OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR_REPORTS / "download_log.csv"

    rows_log: list[dict] = []
    rows_meta_new: list[dict] = []
    n_skipped = n_downloaded = n_failed = 0

    for wa_name, (sid, internal, slug) in targets.items():
        species_dir = RAW_DIR / slug
        if args.apply:
            species_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{wa_name}] (id={sid}, slug={slug})")

        items = collect_items(sid, session, max_pages=args.max_pages)
        print(f"  items recolectados: {len(items)}")
        if args.limit:
            items = items[: args.limit]

        for it in items:
            wa_id = str(it.get("id", "")).strip()
            if not wa_id:
                continue
            link = it.get("link", "")
            mp3_url = jpg_to_mp3_url(link)
            fname = safe_filename(wa_id)
            dest = species_dir / fname
            rel = f"{slug}/{fname}"

            row_meta = {
                "filepath": rel,
                "species": internal,
                "wa_id": wa_id,
                "autor": it.get("autor", ""),
                "fecha": it.get("data", ""),
                "local": it.get("local", ""),
                "dura": it.get("dura", ""),
                "wa_name": (it.get("sp") or {}).get("nome", ""),
                "link_original": link,
                "mp3_url": mp3_url,
            }

            if dest.exists() and dest.stat().st_size >= 1024:
                n_skipped += 1
                rows_log.append({**row_meta, "status": "skip-exists"})
                continue

            if not args.apply:
                n_skipped += 1
                rows_log.append({**row_meta, "status": "dry-run"})
                continue

            ok, err = download_one(mp3_url, dest, session)
            if ok:
                n_downloaded += 1
                rows_log.append({**row_meta, "status": "ok"})
                rows_meta_new.append(row_meta)
            else:
                n_failed += 1
                rows_log.append({**row_meta, "status": f"fail: {err}"})
                print(f"  ! WA{wa_id} fail: {err}")

            time.sleep(SLEEP_BETWEEN_DOWNLOADS)

        print(f"  acum: downloaded={n_downloaded} skipped={n_skipped} failed={n_failed}")

    # Log
    if rows_log:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_log[0].keys()))
            w.writeheader()
            w.writerows(rows_log)
        print(f"\nLog: {log_path}")

    # Metadata acumulativa
    if args.apply and rows_meta_new:
        existing: list[dict] = []
        if META_CSV.exists():
            with open(META_CSV, encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        all_rows = existing + rows_meta_new
        # dedup por filepath
        seen = set()
        dedup: list[dict] = []
        for r in reversed(all_rows):
            if r["filepath"] in seen:
                continue
            seen.add(r["filepath"])
            dedup.append(r)
        dedup.reverse()
        with open(META_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dedup[0].keys()))
            w.writeheader()
            w.writerows(dedup)
        print(f"Metadata: {META_CSV}  ({len(dedup)} filas totales)")

    print(f"\nResumen: downloaded={n_downloaded}  skipped={n_skipped}  failed={n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
