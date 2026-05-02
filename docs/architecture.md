# Arquitectura

> Documento placeholder. Ver `README.md` para una descripción rápida y `infra/README.md` para el setup de AWS.

## Diagrama (TODO)

```
[Audio del usuario]
        │
        ▼
[Gradio UI (HF Spaces)] ──► [API Gateway] ──► [Lambda (Docker)]
                                                    │
                                                    ├─ BirdNET TFLite (embedding 320-dim)
                                                    └─ Classifier ONNX (top-K)
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              ▼                                           ▼
                     [Respuesta al usuario]                       [Flagging opcional]
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                          ▼                               ▼
                                            [S3: clips flagged]                 [DynamoDB: metadata]
                                                          │
                                                          ▼
                                       Re-entrenamiento periódico (Colab)
```

## Componentes (TODO desarrollar)

- Frontend
- Inference layer
- Storage layer
- Observability layer
- Re-training loop
