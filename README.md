# bird_classifierPY

![status](https://img.shields.io/badge/status-scaffolding-lightgrey)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Clasificador de aves de Paraguay por audio (20 especies).
Pipeline: BirdNET V2.4 (TFLite) extrae embeddings de 1024 dimensiones → cabezal MLP propio entrenado con PyTorch Lightning → ONNX para producción.
Deploy en AWS Lambda (Docker + ECR + API Gateway) con frontend Gradio en HuggingFace Spaces.

## Results

Benchmark `wa-drop3-v1` (modelo en producción) vs BirdNET V2.4 nativo sobre el mismo audio, mapeando el top-1 nativo a las 20 especies del scope:

| Test set | n | Baseline (BirdNET nativo) | Pipeline (mio) | Delta |
|---|--:|--:|--:|--:|
| **clean** | 351 | 70.4% top-1 / **63.6% macro** | **94.6% / 93.1% macro** | **+24.2pp / +29.6pp macro** |
| **hard (OOD)** | 208 | 38.9% top-1 / **34.5% macro** | **81.2% / 87.7% macro** | **+42.3pp / +53.2pp macro** |

- **Macro accuracy** (promedio per-especie sin pesos) es la métrica primaria — pondera todas las clases por igual y no se infla con las clases que tienen muchos audios.
- **Pipeline gana o empata en las 20 especies, en ambos folds.** Sin regresiones.
- **Pipile jacutinga** (especie paraguaya no presente en BirdNET V2.4 ni como sinónimo `Aburria`/`Penelope`): baseline 0% garantizado, pipeline 10/10 correct. Ilustra por qué el fine-tuning aporta valor sobre un modelo pretrained de cobertura global.
- Robustez OOD: gap clean→hard del pipeline -13pp, del baseline -31pp. El fine-tuning también comprime la varianza easy/hard.

Tabla completa con per-species + decisiones metodológicas: [`benchmarks/baseline_vs_finetuned_2026-05-16.md`](benchmarks/baseline_vs_finetuned_2026-05-16.md).

## Stack

- **Modelado:** PyTorch, PyTorch Lightning, timm
- **Audio:** librosa, audiomentations, BirdNET (TFLite)
- **Datos:** Pandera (schema validation), Cleanlab (label noise), Xeno-canto API
- **Tracking:** Weights & Biases
- **Config:** Pydantic Settings
- **Backend:** AWS Lambda + Docker + ECR + API Gateway + DynamoDB + S3
- **Frontend:** Gradio en HuggingFace Spaces
- **Monitoring:** EvidentlyAI
- **Calidad:** Ruff (lint + format), pre-commit, pytest
- **Entrenamiento:** Google Colab (notebooks) importando `src/` como paquete

## Estructura del repo

```
bird_classifierPY/
├── .github/workflows/        # CI: tests, deploy a ECR/Lambda, cleanup de W&B
├── src/                      # Paquete Python instalable (importable desde notebooks)
│   ├── config.py             # Pydantic Settings centralizado
│   ├── data/                 # Schemas, Dataset, augmentations, downloader
│   ├── models/               # LightningModule del clasificador
│   ├── training/             # Entry point train() llamado desde notebooks
│   └── monitoring/           # Reportes EvidentlyAI
├── notebooks/                # Notebooks Colab (01..05)
├── scripts/                  # Comandos one-shot reproducibles
├── lambda/                   # Código de la función Lambda + Dockerfile
├── app/                      # Frontend Gradio para HF Spaces
├── tests/                    # pytest (data, models, inference, regression)
├── infra/                    # Documentación del setup AWS manual
├── data/                     # raw/ processed/ test_sets/ (gitignored)
└── docs/                     # Arquitectura, setup, decisiones técnicas (D1-D30)
```

## Setup local

1. **Clonar el repo**
   ```bash
   git clone https://github.com/Placaflaca00/bird_classifierPY.git
   cd bird_classifierPY
   ```

2. **Crear y activar virtualenv (Python 3.11)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux / macOS
   .venv\Scripts\activate        # Windows
   ```

3. **Instalar el paquete en modo editable + dev tools**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Instalar hooks de pre-commit**
   ```bash
   pre-commit install
   ```

5. **Copiar y completar variables de entorno**
   ```bash
   cp .env.example .env
   # editar .env con las credenciales reales (W&B, AWS, HF, etc.)
   ```

## Workflow

El **entrenamiento corre en Google Colab** (no localmente). Los notebooks de `notebooks/` clonan este repo, hacen `pip install -e .` y luego importan desde `src/`. Esto mantiene los notebooks delgados y la lógica testeable.

```python
# Ejemplo de celda en Colab
!git clone https://github.com/Placaflaca00/bird_classifierPY.git
%cd bird_classifierPY
!pip install -e .

from src.training.train import train
train(config_overrides={"epochs": 50})
```

Los `scripts/` son one-shots reproducibles desde CLI (descargar dataset, exportar ONNX, limpiar runs de W&B). Lo que se ejecuta en producción vive en `lambda/` (inferencia) y `app/` (frontend).

## Status

- [ ] **Fase 0** — Scaffolding del repo
- [ ] **Fase 1** — Descarga + curación de datos (Xeno-canto)
- [ ] **Fase 2** — Validación con Pandera + análisis exploratorio
- [ ] **Fase 3** — Extracción de embeddings con BirdNET
- [ ] **Fase 4** — Entrenamiento baseline en Colab
- [ ] **Fase 5** — Augmentations (audiomentations + mixup)
- [ ] **Fase 6** — Detección de label noise con Cleanlab
- [ ] **Fase 7** — Export a ONNX + tests de regresión
- [ ] **Fase 8** — Lambda + Docker + ECR + API Gateway
- [ ] **Fase 9** — Frontend Gradio en HF Spaces
- [ ] **Fase 10** — Monitoring con EvidentlyAI
- [ ] **Fase 11** — CI/CD completo (GitHub Actions)

## Links

- **Demo (HF Spaces):** _pendiente_
- **API (API Gateway):** _pendiente_
- **W&B project:** _pendiente_
- **Documentación de arquitectura:** [`docs/architecture.md`](docs/architecture.md)
- **Decisiones técnicas:** [`docs/decisions/README.md`](docs/decisions/README.md)
- **Setup de infraestructura AWS:** [`infra/README.md`](infra/README.md)

## Licencia

[MIT](LICENSE)
