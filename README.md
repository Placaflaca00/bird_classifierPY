# bird_classifierPY

![status](https://img.shields.io/badge/status-scaffolding-lightgrey)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Clasificador de aves de Paraguay por audio.
Pipeline: BirdNET (TFLite) extrae embeddings de 320 dimensiones → cabezal MLP propio entrenado con PyTorch Lightning.
Deploy en AWS Lambda (Docker + ECR + API Gateway) con frontend Gradio en HuggingFace Spaces.

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
