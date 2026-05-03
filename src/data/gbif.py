"""Cliente para la API de GBIF (Global Biodiversity Information Facility).

GBIF agrega registros de biodiversidad de Xeno-canto, Macaulay Library,
iNaturalist, eBird y otras fuentes. Para nuestro caso, sirve como segundo
canal (complementario a Xeno-canto) para conseguir audios de las especies de
Paraguay y, si hace falta, de Sudamérica.

Approach:
- Usamos ``/occurrence/search`` (sincrónico, paginado, sin auth).
- NO usamos ``/occurrence/download/request`` (bulk, asincrónico, requiere
  cuenta y devuelve un ZIP con CSV; los audios siguen estando en URLs
  externas, así que no aporta nada para nuestro caso).

Funciones expuestas:
- ``match_species``        : resuelve nombre científico → ``usageKey`` (taxonKey GBIF).
- ``count_occurrences``    : cuenta registros sin paginar.
- ``search_occurrences``   : itera registros paginando con ``offset``.
- ``extract_audio_media``  : extrae lista ``[{url, format, license, ...}]`` de los media de tipo Sound.
- ``occurrence_to_metadata``: convierte un registro+media en un row para metadata.parquet.
- ``guess_extension``      : decide extensión (.mp3, .wav, ...) a partir del MIME o URL.

Notas sobre la API:
- País: ISO 3166 alpha-2 (Paraguay = ``PY``).
- Continentes: ``SOUTH_AMERICA``, ``NORTH_AMERICA``, etc. (no hay "AMERICAS" combo).
- Pagination: límite duro de ``offset`` ~100k en ``/search``. Para más, usar el
  endpoint de download (que aquí no implementamos).
- Tipo de media: ``mediaType=Sound`` filtra registros con al menos un audio.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import requests

GBIF_API = "https://api.gbif.org/v1"
XC_ID_RE = re.compile(r"XC(\d+)")
DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 300  # tope superior soportado por /occurrence/search
MAX_OFFSET = 100_000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolución de nombre → taxonKey
# ---------------------------------------------------------------------------
def match_species(
    name: str,
    *,
    rank: str | None = "SPECIES",
    session: requests.Session | None = None,
) -> dict | None:
    """Resuelve un nombre científico al taxon backbone de GBIF.

    Devuelve el dict completo (incluye ``usageKey``, ``scientificName``,
    ``rank``, ``kingdom``, ``confidence``, ...) o ``None`` si no hay match.
    """
    s = session or requests
    params: dict[str, object] = {"name": name}
    if rank:
        params["rank"] = rank
    r = s.get(f"{GBIF_API}/species/match", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("matchType") in (None, "NONE"):
        return None
    return data


# ---------------------------------------------------------------------------
# Búsqueda de ocurrencias
# ---------------------------------------------------------------------------
def _build_search_params(
    *,
    taxon_key: int,
    country: str | None,
    continent: str | None,
    media_type: str,
    limit: int,
    offset: int,
) -> dict:
    params: dict[str, object] = {
        "taxonKey": taxon_key,
        "mediaType": media_type,
        "limit": limit,
        "offset": offset,
    }
    if country:
        params["country"] = country
    if continent:
        params["continent"] = continent
    return params


def count_occurrences(
    *,
    taxon_key: int,
    country: str | None = None,
    continent: str | None = None,
    media_type: str = "Sound",
    session: requests.Session | None = None,
) -> int:
    """Cuenta registros para los filtros dados (sin paginar)."""
    params = _build_search_params(
        taxon_key=taxon_key,
        country=country,
        continent=continent,
        media_type=media_type,
        limit=1,
        offset=0,
    )
    s = session or requests
    r = s.get(f"{GBIF_API}/occurrence/search", params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return int(r.json().get("count", 0))


def search_occurrences(
    *,
    taxon_key: int,
    country: str | None = None,
    continent: str | None = None,
    media_type: str = "Sound",
    page_size: int = DEFAULT_PAGE_SIZE,
    max_records: int | None = None,
    session: requests.Session | None = None,
) -> Iterator[dict]:
    """Itera ocurrencias paginando con offset (tope ``MAX_OFFSET`` por límite de la API)."""
    s = session or requests
    yielded = 0
    offset = 0
    while True:
        params = _build_search_params(
            taxon_key=taxon_key,
            country=country,
            continent=continent,
            media_type=media_type,
            limit=page_size,
            offset=offset,
        )
        r = s.get(f"{GBIF_API}/occurrence/search", params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) or []
        if not results:
            return
        for occ in results:
            yield occ
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
        offset += page_size
        if data.get("endOfRecords", False):
            return
        if offset >= MAX_OFFSET:
            logger.warning("GBIF offset cap reached (%d); some results not retrieved", MAX_OFFSET)
            return


# ---------------------------------------------------------------------------
# Media + metadata
# ---------------------------------------------------------------------------
def extract_audio_media(occurrence: dict) -> list[dict]:
    """Devuelve los items de tipo Sound de un registro, normalizados.

    Cada item: ``{"url", "format", "license", "creator", "publisher", "references"}``.
    """
    media = occurrence.get("media") or []
    out: list[dict] = []
    for m in media:
        if (m.get("type") or "").lower() != "sound":
            continue
        url = m.get("identifier") or m.get("references")
        if not url:
            continue
        out.append(
            {
                "url": url,
                "format": m.get("format", "") or "",
                "license": m.get("license", "") or "",
                "creator": m.get("creator", "") or "",
                "publisher": m.get("publisher", "") or "",
                "references": m.get("references", "") or "",
            }
        )
    return out


def guess_extension(media_url: str, media_format: str) -> str:
    """Decide la extensión de archivo (sin punto) a partir de MIME o URL."""
    fmt = (media_format or "").lower()
    if "mpeg" in fmt or "mp3" in fmt:
        return "mp3"
    if "wav" in fmt:
        return "wav"
    if "ogg" in fmt or "vorbis" in fmt:
        return "ogg"
    if "flac" in fmt:
        return "flac"
    if "mp4" in fmt or "m4a" in fmt or "aac" in fmt:
        return "m4a"
    url_lower = media_url.lower()
    for ext in (".mp3", ".wav", ".ogg", ".flac", ".m4a"):
        if ext in url_lower:
            return ext.lstrip(".")
    return "mp3"


def extract_xc_id(occurrence: dict, media: dict | None = None) -> str:
    """Extrae el ID de Xeno-canto si el registro proviene de XC.

    Lo busca primero en ``occurrenceID`` (típicamente una URL como
    ``https://.../XC993748``), después en la URL del media. Devuelve la parte
    numérica como string (ej: ``"993748"``), o ``""`` si no aplica.
    """
    occ_id = str(occurrence.get("occurrenceID", "") or "")
    m = XC_ID_RE.search(occ_id)
    if m:
        return m.group(1)
    if media:
        url = str(media.get("url", "") or "")
        m = XC_ID_RE.search(url)
        if m:
            return m.group(1)
    return ""


def occurrence_to_metadata(
    occurrence: dict,
    media: dict,
    filepath: str,
    species: str,
) -> dict:
    """Convierte un registro GBIF + uno de sus audios en un row para ``metadata.parquet``.

    Mantiene las mismas columnas core que ``recording_to_metadata`` (XC) más
    extras específicos de GBIF (``gbif_id``, ``occurrence_id``, ``publisher``)
    y el ``xc_id`` cuando el audio originalmente vino de Xeno-canto (clave para
    dedupear contra una corrida posterior del script de XC).
    """
    return {
        "filepath": filepath,
        "species": species,
        "rating": "no-score",
        "source": "gbif",
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "gbif_id": str(occurrence.get("key", "")),
        "occurrence_id": str(occurrence.get("occurrenceID", "")),
        "xc_id": extract_xc_id(occurrence, media),
        "country": occurrence.get("country", "") or "",
        "english_name": occurrence.get("vernacularName", "") or "",
        "recordist": media.get("creator", "") or "",
        "license": media.get("license", "") or "",
        "url": media.get("url", "") or "",
        "publisher": (media.get("publisher", "") or occurrence.get("publishingOrgKey", "") or ""),
    }
