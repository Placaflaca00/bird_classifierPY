# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repo is **mostly scaffolding**. Files in `src/`, `lambda/`, `app/`, and `tests/` are docstring-only stubs that describe intended behavior with `TODO: implementar`. Always read a file before importing from it or recommending it — assume nothing is implemented unless verified. The phase checklist in `README.md` (Fase 0–11) tracks overall buildout. As of this writing, real working code lives in `scripts/analyze_raw_audio.py` and `scripts/build_metadata.py`; everything else is a stub.

## Common commands

```powershell
# Setup (Python 3.11; the .venv in repo uses Python 3.12 — check before troubleshooting)
.venv\Scripts\activate
pip install -e ".[dev]"          # core + dev tools (ruff, pytest, pre-commit)
pip install -e ".[data]"         # data pipeline deps (pandas, pyarrow, pandera, ...)
pre-commit install

# Lint + format (ruff handles both via pre-commit)
pre-commit run --all-files

# Tests
pytest
pytest tests/test_data.py -k <name>     # single test by keyword

# Data pipeline (run from repo root)
python scripts/analyze_raw_audio.py              # dry-run scan for silent/corrupt mp3s
python scripts/analyze_raw_audio.py --delete     # actually delete bad files
python scripts/build_metadata.py                 # dry-run consolidated metadata.parquet
python scripts/build_metadata.py --apply         # write data/raw/metadata.parquet
```

Credentials (W&B, AWS, HF) live in `.env` (copy from `.env.example`). When the user needs to add a token, **direct them to paste into `.env` directly — never into chat**. The user has a documented habit of pasting credentials in chat; flag aggressively.

**W&B project name (case-sensitive): `bird-classifierPy`** (con `P` mayúscula al final, NO `bird-classifier-py`). Aplica a `WANDB_PROJECT` env var, URLs `wandb.ai/<entity>/bird-classifierPy`, `use_artifact()`, `wandb.init(project=...)`. Los `HF_SPACE_NAME`, `ECR_REPOSITORY`, `LAMBDA_FUNCTION_NAME` en `.env.example` son nombres de OTROS servicios (no W&B) y pueden tener su propia convención.

## Architecture (big picture)

Two-stage classifier:

1. **Feature extractor** — BirdNET V2.4 TFLite produces a **1024-dim** embedding per audio clip (penúltima capa, leída con `experimental_preserve_all_tensors=True` y `embedding_idx = classifier_out_idx - 1`; ver `scripts/precompute_embeddings.py`). Frozen, not trained. Used both offline (training-time embedding pre-computation) and online (Lambda inference). Nota: docs viejos de BirdNET (V1.x / V2.1) mencionan 320 — **no aplica a V2.4**.
2. **Classification head** — MLP `1024 → 256 → 128 → num_classes` as `pl.LightningModule` trained on pre-computed embeddings. Exported to ONNX for production.

Data flow:

```
data/raw/<species>/*.mp3  →  data/raw/metadata.parquet  →  data/processed/{embeddings.npy, labels.npy}  →  ONNX  →  Lambda
                              scripts/build_metadata.py     scripts/precompute_embeddings.py              nb 05
```

Deployment topology (`docs/architecture.md`):

```
[Gradio (HF Spaces)] → [API Gateway] → [Lambda Docker] → BirdNET TFLite → ONNX classifier
                                              ↓
                              [S3 flagged audio] + [DynamoDB flag rows]  → Colab retrain
```

Key conventions:
- The folder name under `data/raw/` (snake_case `rhea_americana`) is the **canonical species label**. The `species` column in `metadata.parquet` uses scientific format (`Rhea americana`) derived from the folder, not from external metadata.
- `data/raw/`, `data/processed/`, `data/test_sets/` are gitignored. Don't expect them in CI; assume any analysis script must handle a missing/empty `data/`.
- W&B Artifacts are the source of truth for versioned datasets/models. Local `data/` is a working copy.

## Where work happens

- **Training runs in Google Colab**, not locally. Notebooks (`notebooks/01..05`) clone the repo, `pip install -e .`, then `from src.training.train import train`. Keep heavy logic in `src/`, not in notebook cells.
- **`scripts/`** are one-shot CLIs run on a workstation. Follow the **dry-run-by-default + `--apply` flag** pattern (see `analyze_raw_audio.py` and `build_metadata.py`); never destructive without an explicit flag.
- **`lambda/handler.py`** is the production inference entry point. **`app/app.py`** is the Gradio frontend on HuggingFace Spaces.
- **Architecture decisions** go in `docs/decisions/D<NN>_<slug>.md` (ADR format; D1–D30 indexed in `docs/decisions/README.md`, mostly empty). When making a non-obvious design choice, propose adding a Dnn entry instead of a code comment.

## Audio gotchas

- The system has `ffprobe` / `ffmpeg` (Gyan.FFmpeg via winget). Use them for fast metadata/silence detection — don't install librosa just for that.
- On Windows, **never use `subprocess.run(..., text=True)` against ffmpeg/ffprobe**: stderr contains MP3 metadata bytes (artist, title) that crash cp1252 decoding. Capture bytes and decode with `errors="replace"`. The same applies to `out.stderr` being `None` on subprocess failure.
- Dataset sample rates vary widely (8 kHz – 192 kHz). BirdNET resamples internally to 48 kHz; do not pre-resample.
- 23 species are in scope. Three have <20 audios (`eudromia_formosa`, `pipile_jacutinga`, `rhea_americana`); these are intentionally kept rather than dropped — see `memory/project_app_crowdsource_idea.md` for the planned crowdsourcing flow.

## Language

The codebase is Spanish-first: docstrings, commit messages, inline comments, and most docs are in Spanish (English identifiers). Match that style when adding new code or documentation. Status reports back to the user are in Spanish.
