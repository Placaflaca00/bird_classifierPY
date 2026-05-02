"""Tests de regresión sobre el set fijo en ``data/test_sets/``.

Filosofía:
- Se versiona un set chico de clips representativos (10-30) con sus etiquetas
  conocidas en ``data/test_sets/``.
- Cada release ejecuta este test: el modelo nuevo NO puede empeorar la
  accuracy contra ese set respecto a un baseline guardado en
  ``data/test_sets/baseline_metrics.json``.

Cobertura objetivo:
- ``test_no_accuracy_regression``  : accuracy_actual >= accuracy_baseline - tol.
- ``test_no_class_collapsed``      : el modelo predice al menos N clases distintas.
- ``test_pytorch_vs_onnx_match``   : outputs de PyTorch y ONNX coinciden ± atol.

TODO: implementar tests y commitear el ``baseline_metrics.json`` cuando exista.
"""
