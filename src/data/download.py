"""Cliente para la API v3 de Xeno-canto.

Cambios respecto a v2 (importantes):
- Endpoint nuevo: ``https://xeno-canto.org/api/3/recordings``.
- ``key`` es **obligatorio** (registrate en xeno-canto.org y obtené la tuya en la
  página de tu cuenta).
- Las búsquedas requieren *tags* explícitos. Buscar ``Rhea americana`` ya no
  funciona; hay que usar ``gen:Rhea sp:americana`` (o ``en:`` para nombre en inglés).
- ``per_page`` configurable (50–500, default 100).
- En el response: ``lng`` se renombró a ``lon``.

Funciones expuestas:
- ``species_to_tags``      : ``"Rhea americana"`` → ``"gen:Rhea sp:americana"``.
- ``min_quality_to_xc``    : ``--min-quality B`` → ``q:">C"`` (sintaxis v3 con operador).
- ``build_query``          : combina tags de especie + país/área + calidad.
- ``query_xenocanto``      : llamada cruda con ``key`` y ``per_page``.
- ``count_recordings``     : devuelve ``(numRecordings, numSpecies)``.
- ``search_recordings``    : itera sobre todas las páginas.
- ``download_recording``   : baja un .mp3 con retry exponencial.
- ``recording_to_metadata``: mapea un dict de la API a un row del parquet.
- ``slugify``              : normaliza nombres a snake_case para carpetas.
- ``parse_length``         : ``"MM:SS"`` o ``"HH:MM:SS"`` → segundos float.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

XENOCANTO_API = "https://xeno-canto.org/api/3/recordings"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 1  # rely on session-level urllib3 retry; app-level retry adds little
DEFAULT_BACKOFF = 1.5
DEFAULT_PAGE_SLEEP = 0.4
DEFAULT_PER_PAGE = 500  # tope superior; reduce viajes de red
DEFAULT_DOWNLOAD_TIMEOUT = 30  # antes era 60; un host muerto no merece 60s

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sesión resiliente
# ---------------------------------------------------------------------------
def make_resilient_session(
    *,
    user_agent: str = "bird_classifierPY/0.0.1 (research)",
    read_retries: int = 4,
    connect_retries: int = 2,
    backoff_factor: float = 0.5,
) -> requests.Session:
    """Sesión HTTP con retry diferenciado para connect vs read.

    - ``connect_retries=2``: si el host está caído no tiene sentido reintentar 5 veces.
    - ``read_retries=4`` : transient hiccups una vez establecida la conexión sí se reintentan.
    - ``backoff_factor=0.5``: 0.5s, 1s, 2s, 4s entre intentos (tope ~7s en peor caso).
    Total worst-case por URL muerta: ~30s timeout × 2 connect + ~3s backoff ≈ 60-65s.
    """
    retry = Retry(
        total=read_retries + connect_retries,
        connect=connect_retries,
        read=read_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """``"Rhea americana"`` → ``"rhea_americana"``."""
    return "_".join(name.lower().strip().split())


def parse_length(length: str | None) -> float:
    """Convierte ``"MM:SS"`` o ``"HH:MM:SS"`` (formato XC) a segundos float."""
    if not length:
        return 0.0
    try:
        parts = [int(p) for p in length.split(":")]
    except ValueError:
        return 0.0
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return 0.0


def species_to_tags(scientific_name: str) -> str:
    """Convierte ``"Rhea americana"`` (o con subespecie) en tags v3.

    - ``"Rhea americana"``                   → ``gen:Rhea sp:americana``
    - ``"Rhea americana albescens"``         → ``gen:Rhea sp:americana ssp:albescens``
    - Una sola palabra                       → ``gen:<palabra>`` (busca el género).
    """
    parts = scientific_name.strip().split()
    if not parts:
        raise ValueError("scientific_name vacío")
    out = [f"gen:{parts[0]}"]
    if len(parts) >= 2:
        out.append(f"sp:{parts[1]}")
    if len(parts) >= 3:
        out.append(f"ssp:{parts[2]}")
    return " ".join(out)


def min_quality_to_xc(min_quality: str | None) -> str | None:
    """Traduce ``--min-quality B`` a la sintaxis v3 (operador entre comillas).

    Convención XC: A es la mejor, E la peor.
    - ``A`` → solo A          → ``q:A``
    - ``B`` → A o B           → ``q:">C"``
    - ``C`` → A, B o C        → ``q:">D"``
    - ``D`` → A, B, C o D     → ``q:">E"``
    - ``E`` o None            → sin filtro de calidad
    """
    if not min_quality:
        return None
    q = min_quality.strip().upper()
    if q == "A":
        return "q:A"
    if q == "B":
        return 'q:">C"'
    if q == "C":
        return 'q:">D"'
    if q == "D":
        return 'q:">E"'
    if q == "E":
        return None
    raise ValueError(f"min_quality inválido: {min_quality!r} (esperado A-E)")


def build_query(
    scientific_name: str,
    *,
    country: str | None = None,
    area: str | None = None,
    quality_filter: str | None = None,
) -> str:
    """Arma el ``query`` v3 a partir del nombre científico + filtros opcionales.

    Si se pasan tanto ``country`` como ``area``, ambos se incluyen (XC los combina con AND).
    """
    parts: list[str] = [species_to_tags(scientific_name)]
    if country:
        # cnt: acepta valor sin comillas si no hay espacios; con comillas si hay.
        parts.append(f'cnt:"{country}"' if " " in country else f"cnt:{country}")
    if area:
        parts.append(f"area:{area}")
    if quality_filter:
        parts.append(quality_filter)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def query_xenocanto(
    query: str,
    *,
    key: str,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
    timeout: float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> dict:
    """Llamada cruda a la API v3. Devuelve el dict JSON tal cual."""
    if not key:
        raise ValueError(
            "Xeno-canto API v3 requiere una key. Registrate en https://xeno-canto.org/account "
            "y guardá la tuya en la variable de entorno XENOCANTO_API_KEY (o pasala por --api-key)."
        )
    params = {"query": query, "key": key, "page": page, "per_page": per_page}
    s = session or requests
    resp = s.get(XENOCANTO_API, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def search_recordings(
    query: str,
    *,
    key: str,
    max_records: int | None = None,
    per_page: int = DEFAULT_PER_PAGE,
    session: requests.Session | None = None,
) -> Iterator[dict]:
    """Itera sobre todas las grabaciones de una query, paginando."""
    page = 1
    total_pages = 1
    yielded = 0
    while page <= total_pages:
        data = query_xenocanto(query, key=key, page=page, per_page=per_page, session=session)
        total_pages = int(data.get("numPages", 1) or 1)
        for rec in data.get("recordings", []):
            yield rec
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
        page += 1
        if page <= total_pages:
            time.sleep(DEFAULT_PAGE_SLEEP)


def count_recordings(
    query: str,
    *,
    key: str,
    session: requests.Session | None = None,
) -> tuple[int, int]:
    """Devuelve ``(numRecordings, numSpecies)`` consultando solo la primera página."""
    data = query_xenocanto(query, key=key, page=1, per_page=50, session=session)
    return int(data.get("numRecordings", 0) or 0), int(data.get("numSpecies", 0) or 0)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def download_recording(
    url: str,
    dest: Path,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    session: requests.Session | None = None,
) -> bool:
    """Baja una grabación a ``dest``. Devuelve True si bajó algo, False si ya existía."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    url = _normalize_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with s.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return True
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            wait = backoff * attempt
            logger.warning("download failed (%s), retry %d/%d in %.1fs", e, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}: {last_error}")


# ---------------------------------------------------------------------------
# Mapping a metadata
# ---------------------------------------------------------------------------
def recording_to_metadata(rec: dict, filepath: str) -> dict:
    """Convierte un dict de la API XC en un row para ``metadata.parquet``.

    Campos cubiertos por ``MetadataSchema`` (ver ``src/data/schemas.py``):
    ``filepath``, ``species``, ``rating``, ``source``, ``duration_seconds``.
    ``sample_rate`` queda en 0; se completa más tarde inspeccionando el audio.

    Campos extras (auditoría / cita / re-descarga):
    ``xenocanto_id``, ``country``, ``english_name``, ``recordist``, ``license``, ``url``.
    """
    full_species = f"{rec.get('gen', '').strip()} {rec.get('sp', '').strip()}".strip()
    rating = rec.get("q") or "no-score"
    return {
        "filepath": filepath,
        "species": full_species,
        "rating": str(rating).upper(),
        "source": "xenocanto",
        "duration_seconds": parse_length(rec.get("length")),
        "sample_rate": 0,
        "xenocanto_id": str(rec.get("id", "")),
        "country": rec.get("cnt", "") or "",
        "english_name": rec.get("en", "") or "",
        "recordist": rec.get("rec", "") or "",
        "license": rec.get("lic", "") or "",
        "url": rec.get("url", "") or "",
    }
