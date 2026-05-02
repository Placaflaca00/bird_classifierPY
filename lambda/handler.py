"""Entry point de la función AWS Lambda.

Función ``handler(event, context)`` invocada vía API Gateway.

Flujo esperado:
1. Decodificar el audio del request (base64 en el body o presigned S3 URL).
2. Resamplear a 48 kHz (librosa).
3. Extraer embedding 320-dim con BirdNET TFLite (``tflite_runtime``).
4. Inferir top-K especies con el modelo ONNX (``onnxruntime``).
5. Devolver JSON: ``{"predictions": [{"species": ..., "score": ...}], "model_version": ...}``.

Cold-start optimizado:
- Cargar BirdNET y el ONNX FUERA del handler (a módulo) para reuso entre invocaciones.

TODO: implementar handler + helpers, sin llamar a librerías pesadas en import time si la Lambda tiene memoria limitada.
"""
