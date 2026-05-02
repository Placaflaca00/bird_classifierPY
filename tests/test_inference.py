"""Tests del pipeline de inferencia (Lambda handler).

Cobertura objetivo:
- ``handler(event, context)`` con un audio de prueba devuelve la estructura
  ``{"predictions": [...], "model_version": ...}``.
- Manejo de body vacío / formato inválido devuelve 400.
- Dummy mocks para ``tflite_runtime`` y ``onnxruntime`` para no requerir los
  modelos reales en CI.

TODO: implementar tests.
"""
