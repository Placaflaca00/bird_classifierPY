"""Analiza data/raw/<species>/*.mp3 y reporta archivos silenciosos / corruptos.

Usa ffprobe + ffmpeg (volumedetect) — no requiere librosa/soundfile.

Modos:
    python scripts/analyze_raw_audio.py             # dry-run (solo reporta)
    python scripts/analyze_raw_audio.py --delete    # borra silenciosos/corruptos

Un archivo se considera "silencioso" si:
    - ffprobe falla al leerlo (corrupto / 0 bytes), o
    - max_volume detectado por ffmpeg < SILENCE_DB (default -50 dB), o
    - duración < MIN_DURATION_SEC (default 0.5 s).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SILENCE_DB = -50.0
MIN_DURATION_SEC = 0.5


@dataclass
class FileReport:
    path: Path
    duration: float | None = None
    max_volume: float | None = None
    mean_volume: float | None = None
    error: str | None = None

    @property
    def is_bad(self) -> bool:
        if self.error is not None:
            return True
        if self.duration is None or self.duration < MIN_DURATION_SEC:
            return True
        if self.max_volume is None or self.max_volume < SILENCE_DB:
            return True
        return False

    @property
    def reason(self) -> str:
        if self.error:
            return f"error: {self.error}"
        if self.duration is None or self.duration < MIN_DURATION_SEC:
            return f"too_short ({self.duration}s)"
        if self.max_volume is None or self.max_volume < SILENCE_DB:
            return f"silent (max={self.max_volume}dB)"
        return "ok"


def probe_duration(path: Path) -> float | None:
    """Duración en segundos vía ffprobe; None si falla."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, timeout=30, check=True,
        )
        data = json.loads(out.stdout.decode("utf-8", errors="replace"))
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, KeyError, ValueError):
        return None


_VOL_RE = re.compile(r"(mean|max)_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def probe_volume(path: Path) -> tuple[float | None, float | None]:
    """(mean_volume_dB, max_volume_dB) vía ffmpeg volumedetect; (None,None) si falla."""
    try:
        out = subprocess.run(
            [
                "ffmpeg", "-nostats", "-hide_banner",
                "-i", str(path),
                "-af", "volumedetect",
                "-vn", "-sn", "-dn",
                "-f", "null", "-",
            ],
            capture_output=True, timeout=60, check=False,
        )
    except subprocess.TimeoutExpired:
        return None, None

    blob = (out.stderr or b"").decode("utf-8", errors="replace")
    mean = max_ = None
    for kind, value in _VOL_RE.findall(blob):
        if kind == "mean":
            mean = float(value)
        elif kind == "max":
            max_ = float(value)
    return mean, max_


def analyze_file(path_str: str) -> FileReport:
    path = Path(path_str)
    if path.stat().st_size == 0:
        return FileReport(path=path, error="empty_file")
    duration = probe_duration(path)
    if duration is None:
        return FileReport(path=path, error="ffprobe_failed")
    mean, max_ = probe_volume(path)
    return FileReport(path=path, duration=duration, mean_volume=mean, max_volume=max_)


@dataclass
class SpeciesSummary:
    species: str
    total: int = 0
    ok: int = 0
    bad: list[FileReport] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    @property
    def total_duration_min(self) -> float:
        return sum(self.durations) / 60.0


def collect_files(raw_dir: Path) -> dict[str, list[Path]]:
    species_files: dict[str, list[Path]] = {}
    for sp_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = sorted(sp_dir.glob("*.mp3"))
        if files:
            species_files[sp_dir.name] = files
    return species_files


def print_report(summaries: list[SpeciesSummary], delete: bool) -> None:
    summaries = sorted(summaries, key=lambda s: s.ok)
    header = f"{'species':<28} {'total':>6} {'ok':>6} {'bad':>5} {'min':>8} {'mean_s':>7}"
    print(header)
    print("-" * len(header))
    grand_total = grand_ok = grand_bad = 0
    grand_minutes = 0.0
    for s in summaries:
        mean_s = (sum(s.durations) / len(s.durations)) if s.durations else 0.0
        print(
            f"{s.species:<28} {s.total:>6} {s.ok:>6} {len(s.bad):>5} "
            f"{s.total_duration_min:>8.1f} {mean_s:>7.1f}"
        )
        grand_total += s.total
        grand_ok += s.ok
        grand_bad += len(s.bad)
        grand_minutes += s.total_duration_min
    print("-" * len(header))
    print(
        f"{'TOTAL':<28} {grand_total:>6} {grand_ok:>6} {grand_bad:>5} "
        f"{grand_minutes:>8.1f}"
    )

    bad_files = [r for s in summaries for r in s.bad]
    if not bad_files:
        print("\nSin archivos malos.")
        return

    print(f"\n{'BORRADOS' if delete else 'A BORRAR (dry-run)'}: {len(bad_files)} archivos")
    for r in bad_files[:20]:
        print(f"  {r.path.parent.name}/{r.path.name}  -> {r.reason}")
    if len(bad_files) > 20:
        print(f"  ... y {len(bad_files) - 20} más")

    if delete:
        for r in bad_files:
            try:
                r.path.unlink()
            except OSError as e:
                print(f"  ! falló borrar {r.path}: {e}", file=sys.stderr)
        print(f"\nBorrados {len(bad_files)} archivos.")
    else:
        print("\nDry-run: no se borró nada. Re-ejecutar con --delete para aplicar.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true",
                        help="Borrar archivos silenciosos/corruptos (default: dry-run)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Procesos paralelos para ffmpeg (default: 8)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    species_files = collect_files(args.raw_dir)
    if not species_files:
        print(f"No se encontraron .mp3 en {args.raw_dir}", file=sys.stderr)
        return 1

    total_files = sum(len(v) for v in species_files.values())
    print(f"Analizando {total_files} archivos en {len(species_files)} especies...\n")

    flat: list[tuple[str, Path]] = [
        (sp, p) for sp, files in species_files.items() for p in files
    ]

    summaries = {sp: SpeciesSummary(species=sp) for sp in species_files}
    for sp in summaries:
        summaries[sp].total = len(species_files[sp])

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(analyze_file, str(p)): sp for sp, p in flat}
        for fut in as_completed(futures):
            sp = futures[fut]
            report = fut.result()
            done += 1
            if done % 50 == 0 or done == total_files:
                print(f"  {done}/{total_files}", file=sys.stderr)
            if report.is_bad:
                summaries[sp].bad.append(report)
            else:
                summaries[sp].ok += 1
                if report.duration is not None:
                    summaries[sp].durations.append(report.duration)

    print()
    print_report(list(summaries.values()), delete=args.delete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
