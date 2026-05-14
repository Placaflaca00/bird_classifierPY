# D10 — AWS Lambda + Docker + ECR como backend de inferencia

- **Estado:** aceptada
- **Fecha:** 2026-05-12
- **Contexto**

  Necesitamos un endpoint HTTP para inferencia del clasificador. Características de la carga:

  - Demo de portafolio, tráfico irregular (probablemente decenas/cientos de invocaciones por día, no sostenido).
  - Audio MP3 de entrada típico < 200 KB, payload base64 < 300 KB.
  - Modelo: BirdNET V2.4 TFLite (~50 MB) + ONNX MLP (~1.2 MB) + ~200 MB de deps Python (librosa, onnxruntime, ai-edge-litert, soundfile, numba).
  - Inferencia CPU, ~130 ms warm sobre 2 ventanas de 3 s. No requiere GPU.
  - Sin SLA de latencia estricto; un cold-start de varios segundos es tolerable.
  - Restricción de costo: el proyecto no debe gastar más que el AWS Free Tier en estado idle.

- **Decisión**

  Servir la inferencia desde **AWS Lambda con container image** (no zip), publicado en **ECR** en `us-east-1`. La imagen empaqueta el handler, BirdNET TFLite, ONNX classifier y todas las deps Python en un único artefacto inmutable etiquetado con la versión del artifact de W&B del modelo (`wa-drop3-v1`).

  Parámetros confirmados al cierre de Fase 6:

  - Package type: `Image` (no zip).
  - Base: `public.ecr.aws/lambda/python:3.12` (AL2023, glibc 2.34).
  - Memoria: **3008 MB** — Lambda asigna vCPU proporcional a memoria (~1.7 vCPU a 3008 MB). Empíricamente, este valor minimiza el costo por request porque el speedup de inferencia compensa el aumento de precio por GB-segundo.
  - Timeout: **90 s** — cubre cold-start post-create observado (70 s) + worst-case inference.
  - Arquitectura: **x86_64** — todos los wheels de deps (ai-edge-litert manylinux_2_27, onnxruntime manylinux_2_28, soundfile con libsndfile bundleado) están disponibles y validados en x86_64. arm64/Graviton es opción futura, deferida.
  - Ephemeral storage: 512 MB (default) — usado por `NUMBA_CACHE_DIR=/tmp` y `HF_HOME=/tmp/huggingface`.
  - Sin VPC (ver D23).
  - Sin reserved/provisioned concurrency.

- **Consecuencias**

  Trade-offs medidos:

  - **Costo:** pay-per-invoke. AWS Free Tier (1M invocaciones + 400k GB-segundos/mes) cubre con margen todo el tráfico esperado del demo. Idle cost = 0 (no hay instancias corriendo). Costo marginal por request a 3008 MB con ~1.4 s de runtime: ~$0.00007 (~$0.07 cada 1k requests).
  - **Cold start aceptable:** 23.2 s de roundtrip / 1384 ms de inferencia en cold-start medido post-`update-function-configuration` (medición limpia). Para portafolio, está dentro de lo tolerable; si en algún momento hay tráfico real con SLA, hay tres palancas conocidas: SnapStart for Python (cuando lo soporten para container images), provisioned concurrency, o warm-ping desde EventBridge cada N minutos.
  - **Sin infra que mantener:** no hay instancias que parchear, autoscaling que configurar, ni health checks. El deploy es `docker build → docker push → aws lambda update-function-code`.
  - **Forced container path:** las deps Python descomprimidas pesan ~280 MB, lo que excede el límite de 250 MB para Lambda zip. El path container no era preferencia, era requisito.
  - **Latencia geográfica desde Paraguay → us-east-1:** ~3 s adicionales en roundtrip warm (medido: 3.3-4.7 s roundtrip vs 128-131 ms de inferencia). Aceptable para demo. La región se eligió por mejor selección de servicios y mejor free tier, no por latencia (ver "Alternativas").
  - **Tamaño imagen:** 426 MB en ECR (manifest single-arch, comprimido). 4% del límite de Lambda container (10 GB), espacio sobrado.
  - **Lock-in moderado:** la imagen es estándar OCI v2, portable a Cloud Run o ECS si en el futuro hace falta cambiar de backend. El handler Python sigue el formato Lambda (`event, context`) pero la lógica de inferencia es agnóstica.

  Validación funcional al cierre de Fase 6: top-1 Chauna torquata 0.9614 sobre el mismo MP3 de prueba, **bit-idéntico** al resultado de Lambda Runtime Interface Emulator (RIE) local. No hay drift por containerización ni por la arquitectura de despliegue.

- **Alternativas consideradas**

  1. **SageMaker Inference Endpoint:** descartado. Instancias always-on con costo mínimo $50-200/mes, overkill para demo sin tráfico real. Tampoco aporta features que necesitemos (no usamos features de SageMaker como model monitoring nativo ni multi-model endpoints).
  2. **EC2 con Flask/FastAPI:** descartado. Requiere gestión de uptime, parches de SO, auto-recovery, certificate management, security groups — overhead operacional desproporcionado para una demo. Costo idle ~$5-10/mes incluso con t4g.nano.
  3. **ECS Fargate:** descartado. Mejor para cargas sostenidas, pero no escala a cero. Costo idle ~$5-15/mes según task definition.
  4. **Lambda con zip package:** **bloqueado por límite técnico**. Las deps Python descomprimidas pasan 250 MB. No viable.
  5. **SageMaker Serverless Inference:** descartado. Producto más nuevo, soporte container image menos maduro, precio por invocación mayor que Lambda, y heredamos cold-start similar sin beneficios claros.
  6. **HuggingFace Inference Endpoints:** descartado. Lock-in a HF, menos control sobre el entorno (necesitamos pinear ai-edge-litert y onnxruntime). Requeriría separar BirdNET (que correría aparte) del classifier ONNX, complicando el pipeline.
  7. **Google Cloud Run:** viable como alternativa. Descartado por consistencia con el resto del stack planeado (S3 para audio flagged en D25, DynamoDB para flags en D12, ECR ya integrado con Lambda). Si en algún momento migramos por costo o latencia, la imagen Docker es portable.

- **Notas de implementación**

  Tres gotchas no obvios que rompieron el deploy y están documentados en detalle en `memory/project_lambda_container_gotchas.md` (memoria privada del usuario):

  1. **Docker 28 con BuildKit produce OCI image index** (`application/vnd.oci.image.index.v1+json` multi-arch + provenance + sbom), que Lambda rechaza. Fix: `docker build --provenance=false --sbom=false --platform linux/amd64`. Validar con `aws ecr describe-images --query 'imageDetails[0].imageManifestMediaType'` — debe ser `application/vnd.docker.distribution.manifest.v2+json`.
  2. **librosa importa numba**, que intenta cachear JIT en `site-packages/` (read-only en Lambda). Fix vía `ENV NUMBA_CACHE_DIR=/tmp` en el `Dockerfile`. Por higiene se redirigen también `MPLCONFIGDIR` y `HF_HOME` a `/tmp` (caches de matplotlib y huggingface_hub que también pueden romper si una dep transitiva los usa).
  3. **La base `public.ecr.aws/lambda/python:3.12` es minimalista** y no incluye `find`/`rm` con globbing. Cualquier `RUN find ...` en el Dockerfile falla con exit 127. Decisión: no hacer cleanup de `.pyc` post pip-install — las layers son aditivas y los `.pyc` aceleran cold-start.

  Tag strategy: la imagen se tagea con el nombre del artifact de W&B del modelo (`wa-drop3-v1`). Trazabilidad directa: ante la pregunta "¿qué modelo está en producción?", la respuesta es el tag de la imagen actual de la función. No se usa `:latest`. Bug-fixes de runtime (como las ENV vars de cache) **sobreescriben** el tag sin bumpear versión, porque el modelo no cambió.

- **Referencias cruzadas**

  - D09 — ONNX como formato de export: habilita un runtime ligero y compatible con Lambda CPU.
  - D11 — API Gateway HTTP API: capa pública sobre esta Lambda.
  - D22 — Setup AWS manual: justifica los comandos `aws cli` explícitos en vez de IaC.
  - D23 — Sin VPC para la Lambda: decisión separada referenciada arriba.
  - `docs/architecture.md` — diagrama de despliegue (TODO: actualizar el embedding dim de 320 → 1024 cuando se redibuje).
  - `lambda/Dockerfile`, `lambda/handler.py`, `lambda/requirements.txt` — implementación.
