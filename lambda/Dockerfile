# AWS Lambda image: clasificador de aves por audio.
# Build context: directorio `lambda/` (no la raíz del repo).
#
# Build local:
#   docker build --platform linux/amd64 -t bird-classifier:local lambda/
#
# Test local con Lambda Runtime Interface Emulator (incluido en la base image):
#   docker run --rm --platform linux/amd64 -p 9000:8080 bird-classifier:local
#   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#        -d '{"body": "{\"audio_b64\": \"...\", \"top_k\": 3}"}'

# Python 3.12 -> AL2023 (glibc 2.34): requerido para wheels manylinux_2_27 de
# ai-edge-litert y manylinux_2_28 de soundfile (libsndfile bundleado).
# Plataforma forzada a amd64 (Lambda x86_64; armaríamos arm64 aparte si vamos a Graviton).
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Instalar dependencias Python. --no-cache-dir + limpieza de .pyc reduce ~30-50MB.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
 && find /var/lang/lib/python3.12 -name "__pycache__" -type d -exec rm -rf {} + \
 && find /var/lang/lib/python3.12 -name "*.pyc" -delete \
 && rm -rf /root/.cache /tmp/*

# Verificación build-time: libsndfile bundleado debe ser 1.1.0+ (soporte MP3).
# Si esto falla, la imagen no se construye -> evitamos sorpresas en runtime.
RUN python -c "import soundfile as sf; v = sf.__libsndfile_version__; print(f'libsndfile bundled: {v}'); assert tuple(int(x) for x in v.split('.')[:2]) >= (1, 1), f'libsndfile {v} sin soporte MP3 (necesita 1.1.0+)'"

# Handler + metadata de especies (~3 KB)
COPY handler.py species_metadata.json ${LAMBDA_TASK_ROOT}/

# Modelos: BirdNET TFLite (~50 MB) + classifier ONNX (1.2 MB) + sidecar JSON (~1 KB).
# Total ~51 MB, muy por debajo del límite de 10 GB de Lambda container.
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Smoke test del import: si el handler no levanta por imports rotos o paths
# inválidos, falla en build (no en cold-start de producción).
RUN python -c "import sys; sys.path.insert(0, '${LAMBDA_TASK_ROOT}'); import handler; print('handler OK, num_classes =', handler._NUM_CLASSES)"

# Lambda RIC busca <module>.<function>
CMD ["handler.handler"]
