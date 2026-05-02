# Decisiones técnicas

Registro tipo ADR (Architecture Decision Record) de las decisiones de diseño del proyecto.
Cada decisión vive en su propio archivo `D<N>_<slug>.md` (e.g. `D01_birdnet_embeddings.md`) y se referencia desde este índice.

## Formato de cada decisión

```
# D<N> — <título>

- **Estado:** propuesta | aceptada | superada por D<M>
- **Fecha:** YYYY-MM-DD
- **Contexto:** qué problema motivó la decisión.
- **Decisión:** qué se decidió.
- **Consecuencias:** trade-offs, qué se gana y qué se pierde.
- **Alternativas consideradas:** qué se descartó y por qué.
```

## Índice (D1 — D30)

> A medida que se tomen decisiones, agregar aquí el link al archivo correspondiente.

- [ ] D01 — Usar BirdNET como feature extractor (vs entrenar from scratch o fine-tuning).
- [ ] D02 — Cabezal MLP simple sobre embeddings (vs head más profundo).
- [ ] D03 — PyTorch Lightning como framework de training.
- [ ] D04 — Weights & Biases como tracker (vs MLflow / TensorBoard).
- [ ] D05 — Pydantic Settings para config (vs Hydra / dataclass + YAML).
- [ ] D06 — Pandera para validar metadata (vs schema ad-hoc).
- [ ] D07 — Audiomentations + mixup como estrategia de augmentation.
- [ ] D08 — Cleanlab para detección de label noise.
- [ ] D09 — ONNX como formato de export para producción (vs TorchScript).
- [ ] D10 — AWS Lambda + Docker + ECR como backend (vs SageMaker / EC2 / ECS).
- [ ] D11 — API Gateway HTTP API (vs REST API).
- [ ] D12 — DynamoDB para flagging (vs Postgres / S3-only).
- [ ] D13 — Gradio + HuggingFace Spaces como frontend (vs Streamlit / propia).
- [ ] D14 — EvidentlyAI para drift monitoring (vs WhyLabs / casero).
- [ ] D15 — Notebooks en Colab + código modular en `src/` (vs todo en notebooks).
- [ ] D16 — Ruff (lint + format) como única herramienta de calidad de código.
- [ ] D17 — Python 3.11 (vs 3.10 / 3.12).
- [ ] D18 — Estrategia de splits (train/val/test estratificado por especie).
- [ ] D19 — Métrica primaria: macro-F1 (vs accuracy / weighted-F1).
- [ ] D20 — Política de tests de regresión sobre set fijo en `data/test_sets/`.
- [ ] D21 — Política de versionado de modelos (semver del modelo en S3).
- [ ] D22 — Setup AWS manual (vs IaC con Terraform/CDK desde el día 1).
- [ ] D23 — Sin VPC para la Lambda (latencia vs aislamiento).
- [ ] D24 — Política de cleanup de runs W&B (cron semanal).
- [ ] D25 — Política de retención de clips flagged en S3.
- [ ] D26 — Reentrenamiento manual (vs continuous training).
- [ ] D27 — Convención de naming de runs en W&B.
- [ ] D28 — Estrategia para clases con pocos ejemplos (excluir / agrupar).
- [ ] D29 — Filtro de Xeno-canto (rating mínimo, países permitidos).
- [ ] D30 — Estrategia de fallback ante error de la Lambda en el frontend.
