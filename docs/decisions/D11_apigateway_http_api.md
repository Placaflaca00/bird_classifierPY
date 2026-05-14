# D11 — API Gateway HTTP API como capa pública sobre la Lambda

- **Estado:** aceptada
- **Fecha:** 2026-05-13
- **Contexto**

  Cerrada D10 (Lambda en ECR), necesitamos exponerla por HTTPS para que el frontend Gradio en HuggingFace Spaces pueda invocarla desde el navegador del usuario. Restricciones:

  - Demo de portafolio, una sola ruta lógica (`POST /predict`).
  - Tráfico esperado: decenas/cientos de invocaciones por día, picos eventuales por compartir el link en redes/CV.
  - Request body: JSON con audio MP3 en base64, payload típico < 300 KB.
  - El frontend corre en otro dominio (`*.hf.space`), así que el endpoint **necesita CORS habilitado**.
  - Costo: lo mismo que D10 — no debe gastar fuera del Free Tier en idle.
  - Sin requisitos de autenticación en esta fase (público); auth se evaluará si se llega a Fase 4 (DynamoDB + feedback).
  - Sin requisitos de WAF/per-IP rate limit en esta fase; aceptable que un atacante pueda saturar la rate-limit global.

- **Decisión**

  Servir la Lambda detrás de **AWS API Gateway HTTP API** (no REST API, no Function URL) en `us-east-1`, con integración `AWS_PROXY` directa a la Lambda de D10.

  Parámetros confirmados al cierre de Fase 1:

  - Protocolo: **HTTP API v2** (`apigatewayv2`), no REST API.
  - Endpoint: `https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict`.
  - Ruta única: `POST /predict` con `AuthorizationType: NONE` (pública).
  - Integración: `AWS_PROXY`, `PayloadFormatVersion: 2.0`, `TimeoutInMillis: 30000` (máximo permitido por HTTP API).
  - **CORS configurado a nivel API**, no en el handler:
    ```
    AllowOrigins: ["*"]      # restringir a *.hf.space cuando exista la app
    AllowMethods: ["POST", "OPTIONS"]
    AllowHeaders: ["Content-Type"]
    MaxAge: 3600
    ```
    Esto hace que API Gateway responda automáticamente al OPTIONS preflight sin invocar a la Lambda (verificado: `204 No Content` con cero cold-starts gatillados).
  - **Throttling explícito a nivel stage**, no defaults:
    ```
    ThrottlingBurstLimit: 20
    ThrottlingRateLimit: 10   # req/s sostenido
    ```
  - Stage: `prod` con `AutoDeploy: true` (cualquier cambio futuro de ruta/integración se publica solo).
  - Lambda resource policy scopeada al ARN específico de esta API + esta ruta:
    `arn:aws:execute-api:us-east-1:863518416901:1jbbnu85e5/*/POST/predict`. Otras APIs en la misma cuenta no pueden invocar la función.

- **Consecuencias**

  Trade-offs medidos al cierre de Fase 1:

  - **Costo:** $1.00 por millón de requests (HTTP API), ~70% más barato que REST API ($3.50/M). A volumen de demo (1k req/mes) el total mensual del endpoint completo (API Gateway + Lambda + logs) es **~$0.012/mes**. Free Tier cubre con holgura.
  - **CORS nativo a nivel API**: una sola fuente de verdad. Si el handler agregara headers `Access-Control-Allow-*`, API Gateway los sobreescribiría — la regla es: CORS solo en API Gateway, jamás en `lambda/handler.py`.
  - **Latencia warm desde Paraguay: ~1.15 s roundtrip** (vs ~2.7 s del invoke directo a `lambda.us-east-1`, medido). API Gateway entra por la red edge de CloudFront que tiene POPs en Sudamérica; el endpoint `lambda.*.amazonaws.com` directo no. **Esto es contraintuitivo**: agregar el hop de API Gateway hace la latencia warm 2.3× mejor para usuarios sudamericanos, no peor. Para reportes de latencia user-facing usar siempre los números de API Gateway.
  - **Cold start: 26.8 s roundtrip** medido (vs 23.2 s de invoke directo en Fase 6). Los ~3 s extra son overhead de API Gateway en el primer hit. Sigue dentro del timeout de 30 s.
  - **Timeout de integración cap-eado a 30 s** — límite duro de HTTP API, no configurable hacia arriba. Si los audios crecen mucho (muchas windows) o el cold start drifta, el cliente verá `504 Gateway Timeout` aunque la Lambda termine. Mitigation path conocido (no implementado): mover a inferencia asíncrona con SQS/EventBridge si pasa.
  - **Comportamiento de throttling: devuelve 503, no 429.** Bajo carga ráfaga sobre HTTP API + AWS_PROXY integration, el rechazo por exceder el burst del token bucket viene como `HTTP 503 {"message":"Service Unavailable"}`, no como `429 Too Many Requests` que documenta AWS. El header `Apigw-Requestid` confirma que fue rechazado en API Gateway antes de invocar Lambda. **El cliente (Gradio app de Fase 2) debe tratar 429 y 503 como "throttled, retry"**, no solo 429. Verificado empíricamente: 50 requests paralelas → 31×400 (llegan a la Lambda con body inválido) + 19×503 (rechazadas en API Gateway).
  - **Throttling es global, no per-IP.** Un solo atacante puede consumir la rate-limit de 10 req/s para todos los usuarios legítimos. Mitigation path conocido (no implementado): AWS WAF (~$5/mes + per-request) con rate-based rule per-IP, solo si aparece abuso real.
  - **Sin custom domain.** La URL es `*.execute-api.us-east-1.amazonaws.com/prod/predict`. Aceptable para portafolio; un dominio propio agregaría Route53 + ACM y costos marginales sin valor para CV.
  - **Auto-deploy ON simplifica el workflow.** No hay `create-deployment` manual ni stages multiple. Para single-stage HTTP API es lo estándar.

  Validación funcional al cierre de Fase 1: predicción `Chauna torquata 0.9614324569702148` sobre el mismo MP3 de prueba, **bit-idéntica** al resultado del invoke directo a la Lambda y al RIE local. Tres puntos de verificación, cero drift introducido por la capa API Gateway.

- **Alternativas consideradas**

  1. **API Gateway REST API:** descartado. ~3.5× más caro ($3.50/M vs $1.00/M), latencia un poco mayor, CORS requiere configurarse por método y duplicar en el handler. Aporta features que no necesitamos (usage plans, API keys, request validation, modelos JSON Schema). Migrar de HTTP a REST después es trivial si en algún momento aparece la necesidad.
  2. **Lambda Function URL:** descartado. Técnicamente válido para "ML inference backend": Lambda directamente expone un endpoint HTTPS, sin API Gateway. Más simple, más barato, similar para el demo. Razones para preferir HTTP API:
     - **Valor de portafolio**: API Gateway es vocabulario reconocido en JDs y entrevistas; Function URL es feature relativamente nuevo y menos central en discusiones de arquitectura serverless.
     - **Integración de dos servicios AWS** (API Gateway + Lambda) es señal de competencia más clara que un servicio solo.
     - **Extensibilidad**: si se agregan más Lambdas (p. ej. una para feedback en Fase 4), comparten la misma API en lugar de tener N Function URLs sueltas.
     - **CORS nativo a nivel servicio** (Function URL también lo tiene, pero la config es per-function en vez de centralizada).
  3. **Application Load Balancer con Lambda como target:** descartado. ALB cobra por hora ($16-22/mes fijo) además del costo por request — anti-económico para una sola ruta y tráfico irregular. ALB tiene sentido para mezclar muchas Lambdas + targets EC2/ECS, no acá.
  4. **CloudFront + Lambda@Edge:** descartado. La inferencia es CPU-bound y stateful (carga BirdNET + ONNX en memoria); Lambda@Edge tiene límites estrictos de memoria (10 GB en Origin, pero 128 MB en Viewer) y de tamaño de deployment package, incompatibles con la imagen container de D10.
  5. **AppSync (GraphQL):** descartado. Una sola operación POST sin schema relacional no justifica GraphQL. Curva de aprendizaje y costos mayores sin ganancia.
  6. **Nginx/Caddy delante de una EC2 + Lambda invoke:** descartado por las mismas razones que descartaron EC2 en D10 (overhead operacional).

- **Notas de implementación**

  Detalles operacionales y gotchas documentados en `memory/project_phase1_executed_2026-05-13.md` y `memory/project_phase1_api_gateway_plan.md` (memoria privada del usuario):

  1. **CORS shorthand en AWS CLI rompe con valores de lista** (`AllowMethods=POST,OPTIONS` se parsea mal porque las comas conflicten con el separador outer del shorthand). El path robusto en PowerShell es pasar la config como JSON file: `--cors-configuration file://$env:TEMP/cors-policy.json`.
  2. **El handler `lambda/handler.py` NO setea headers CORS.** Solo `Content-Type: application/json`. Esto es deliberado: si el handler devolviera `Access-Control-Allow-*`, API Gateway los reemplazaría. Si algún PR futuro agrega CORS desde el handler, borrarlo.
  3. **`PayloadFormatVersion: 2.0` vs `1.0`:** elegimos 2.0 (default y recomendado para HTTP API). El handler hace `event.get("body", event)` que funciona idéntico con ambos formatos — el `body` sigue siendo un string JSON en los dos. 2.0 es más simple (sin campos REST-legacy).
  4. **El default del throttling de HTTP API es la cuota regional completa de la cuenta** (10k burst / 5k sustained). Sin throttling explícito a nivel stage, un atacante apuntando a este endpoint puede agotar la cuota regional de toda la cuenta AWS. **Configurar throttling explícito es obligatorio**, no opcional, para cualquier HTTP API público en esta cuenta.
  5. **Resource policy de la Lambda scopeada al SourceArn exacto** (`<api-id>/*/POST/predict`), no a `apigateway.amazonaws.com` genérico. Esto previene que otra API en la misma cuenta pueda invocar accidentalmente esta función.
  6. **Tags consistentes los 5** con el resto de los recursos del proyecto: `Project=bird-classifier-py`, `ModelVersion=wa-drop3-v1`, `Environment=dev`, `ManagedBy=manual`, `CostCenter=portfolio`. `ModelVersion` aplica al API Gateway aunque el servicio no tenga relación semántica con el modelo — la convención del proyecto es **mismo set de 5 tags en todos los recursos** para cost allocation consistente.

- **Referencias cruzadas**

  - D10 — AWS Lambda + Docker + ECR: el target detrás de esta API.
  - D13 — Gradio + HuggingFace Spaces como frontend (pendiente): el cliente que va a consumir este endpoint con CORS.
  - D22 — Setup AWS manual (pendiente): justifica los comandos `aws cli` explícitos en vez de IaC para el setup del API.
  - D30 — Fallback ante error de la Lambda en el frontend (pendiente): debe contemplar el caso 503-como-throttled descripto arriba, además de 429, 504 (timeout 30s) y 5xx genéricos.
  - `memory/project_phase1_executed_2026-05-13.md` — ARNs, IDs y números reales medidos.
