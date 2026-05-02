# Placeholder Dockerfile para la imagen de AWS Lambda.
#
# Base recomendada: public.ecr.aws/lambda/python:3.11
# Instalará lambda/requirements.txt y copiará handler.py + los dos modelos
# (birdnet.tflite y classifier.onnx) en /var/task.
#
# TODO: completar build una vez que handler.py esté implementado y los
# modelos estén versionados en S3.

FROM public.ecr.aws/lambda/python:3.11

# TODO: copiar requirements e instalar
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# TODO: copiar handler y modelos
# COPY handler.py ./
# COPY models/birdnet.tflite models/classifier.onnx ./models/

CMD ["handler.handler"]
