"""CLI: exporta un checkpoint Lightning a ONNX (versión scriptable del notebook 05).

Uso:
    python scripts/export_model.py \\
        --ckpt path/to/best.ckpt \\
        --out models/classifier.onnx \\
        --opset 17

Verifica numéricamente que ONNX y PyTorch producen las mismas salidas
sobre un batch de prueba (tolerancia configurable).

TODO: implementar.
"""
