"""Tests del subpaquete ``src.models``.

Cobertura objetivo:
- ``BirdClassifier(num_classes=K).forward(x)`` devuelve shape (B, K).
- ``training_step`` produce un loss escalar finito.
- ``configure_optimizers`` devuelve la estructura esperada por Lightning.
- Smoke test: una época sobre 1 batch sintético no falla.

TODO: implementar tests.
"""
