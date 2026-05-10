"""Analiza cobertura potencial de WikiAves para nuestras 23 especies.

NO descarga audios. Solo cuenta cuántos registros de sonido tiene WikiAves
por especie y compara con nuestros counts locales (data/processed/embeddings.parquet).

API endpoints no oficiales (replicados de github.com/Athospd/wikiaves):

    https://www.wikiaves.com.br/getTaxonsJSON.php?term={species}
        → busca species_id por nombre. Devuelve lista de matches con id y name.

    https://www.wikiaves.com.br/getRegistrosJSON.php?tm=s&t=s&s={species_id}&o=mp&p=1
        → lista paginada (18/página) de registros de sonido. El campo
          registros.total contiene el total disponible.

Comparación esperada: 0 duplicados con nuestro dataset actual (todos vienen
de GBIF, que NO incluye WikiAves).

Uso:
    python scripts/analyze_wikiaves.py                       # full sobre 23 sp
    python scripts/analyze_wikiaves.py --limit 2             # smoke test
    python scripts/analyze_wikiaves.py --species "Tringa flavipes"   # una sp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
OUT_DIR = ROOT / "reports" / "wikiaves_analysis"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) bird-classifier-research/0.1"
)
HEADERS = {"User-Agent": UA, "Accept": "application/json, text/javascript, */*"}
TAXONS_URL = "https://www.wikiaves.com.br/getTaxonsJSON.php"
REGISTROS_URL = "https://www.wikiaves.com.br/getRegistrosJSON.php"
SLEEP_S = 0.6  # cortesía entre requests


def get_species_id(species: str, session: requests.Session) -> tuple[int | None, str | None, list]:
    """Busca species_id por nombre científico. Devuelve (id, name, all_matches)."""
    r = session.get(TAXONS_URL, params={"term": species}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None, None, []
    # data es lista de {id, name, ...}. Buscamos match exacto case-insensitive
    matches = data if isinstance(data, list) else [data]
    target = species.lower().strip()
    exact = [m for m in matches if str(m.get("nome_c", m.get("name", ""))).lower().strip() == target]
    chosen = exact[0] if exact else matches[0]
    sid = chosen.get("id")
    name = chosen.get("nome_c") or chosen.get("name")
    try:
        sid = int(sid)
    except (TypeError, ValueError):
        return None, name, matches
    return sid, name, matches


def get_total_sounds(species_id: int, session: requests.Session) -> tuple[int | None, dict]:
    """Lee registros.total del primer page de getRegistrosJSON."""
    params = {"tm": "s", "t": "s", "s": species_id, "o": "mp", "p": 1}
    r = session.get(REGISTROS_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    # Estructura inferida del repo R: {"registros": {"total": N, "itens": {...}}}
    reg = data.get("registros", data) if isinstance(data, dict) else {}
    total = reg.get("total")
    try:
        total = int(total)
    except (TypeError, ValueError):
        total = None
    return total, data


def load_local_counts() -> pd.Series:
    """Counts no-aug por especie del embeddings.parquet."""
    df = pd.read_parquet(EMB_PATH)
    if "is_aug" in df.columns:
        df = df[~df["is_aug"]]
    return df.groupby("species").size().rename("n_local")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--species", default=None,
                        help="Una sola especie (formato científico). Si no, usa las 23 del repo.")
    parser.add_argument("--save-raw", action="store_true",
                        help="Guarda JSON crudo de la primera página de getRegistrosJSON.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    local = load_local_counts()
    if args.species:
        species_list = [args.species]
    else:
        species_list = sorted(local.index.tolist())
    if args.limit:
        species_list = species_list[: args.limit]

    print(f"Especies a consultar: {len(species_list)}")
    for s in species_list:
        print(f"  - {s}")

    session = requests.Session()
    rows = []
    for i, sp in enumerate(species_list, 1):
        n_local = int(local.get(sp, 0))
        sid: int | None = None
        name = None
        n_wa = None
        err = None
        try:
            sid, name, matches = get_species_id(sp, session)
            time.sleep(SLEEP_S)
            if sid is None:
                err = f"no species_id (matches={len(matches)})"
            else:
                n_wa, raw = get_total_sounds(sid, session)
                if args.save_raw:
                    (OUT_DIR / f"raw_{sid}_{sp.replace(' ', '_')}.json").write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                time.sleep(SLEEP_S)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        rows.append({
            "species": sp,
            "species_id": sid,
            "wa_name": name,
            "n_wikiaves": n_wa,
            "n_local": n_local,
            "ratio_wa_over_local": (n_wa / n_local) if (n_wa and n_local) else None,
            "error": err,
        })
        marker = "OK" if (n_wa is not None) else f"FAIL ({err})"
        print(f"  [{i}/{len(species_list)}] {sp:30s}  id={sid}  wa={n_wa}  local={n_local}  {marker}")

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("n_wikiaves", ascending=False, na_position="last")
    out_csv = OUT_DIR / "wikiaves_coverage.csv"
    df_sorted.to_csv(out_csv, index=False)

    print("\n=== Tabla por cobertura WikiAves (descendente) ===")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(df_sorted.to_string(index=False))

    if df["n_wikiaves"].notna().any():
        total_wa = int(df["n_wikiaves"].fillna(0).sum())
        total_local = int(df["n_local"].sum())
        print(f"\nTOTAL audios disponibles WikiAves: {total_wa}")
        print(f"TOTAL audios locales (no-aug):     {total_local}")
        print(f"RATIO global: {total_wa/max(total_local,1):.2f}x")

    print(f"\nGuardado: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
