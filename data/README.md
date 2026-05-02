# `data/`

Esta carpeta contiene los datos crudos y derivados del proyecto.
**Todo lo que está dentro de `raw/`, `processed/` y `test_sets/` está ignorado por git** (ver `.gitignore`); solo se versionan los `.gitkeep` y este README.

## `raw/`

Audios crudos descargados desde Xeno-canto (y eventualmente field-recordings propios).

Estructura esperada:

```
raw/
├── metadata.parquet                    # generado por scripts/download_xenocanto.py
└── <Genus_species>/
    ├── XC123456.mp3
    ├── XC123457.mp3
    └── ...
```

`metadata.parquet` cumple el schema definido en `src/data/schemas.py`.

## `processed/`

Salidas pre-computadas listas para entrenar el cabezal:

```
processed/
├── embeddings.npy                      # (N, 320) float32 — embeddings BirdNET
├── labels.npy                          # (N,)     int64   — índice de clase
├── index.parquet                       # mapping i → metadata original
└── label_issues.parquet                # output de scripts/find_label_issues.py
```

Generado por `scripts/precompute_embeddings.py` (o el notebook 02).

## `test_sets/`

Set fijo y versionado para **tests de regresión** (`tests/test_regression.py`).

```
test_sets/
├── audio/
│   └── <species>/clip_*.wav
├── labels.csv
└── baseline_metrics.json               # accuracy/F1 del último modelo aceptado
```

A diferencia de `raw/` y `processed/`, este set sí debería poder reconstruirse de manera determinística — pero por su tamaño se mantiene fuera de git y se sincroniza desde S3.

---

**Cómo poblar estas carpetas desde cero:**

```bash
python scripts/download_xenocanto.py        # → data/raw/
python scripts/precompute_embeddings.py     # → data/processed/
# data/test_sets/ se baja desde s3://${S3_BUCKET_MODELS}/test_sets/
```
