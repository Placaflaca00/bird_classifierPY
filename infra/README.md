# Infraestructura AWS — setup manual

> Documentación step-by-step del setup de infra AWS.
> Todo se configura **a mano** por la consola/CLI (no IaC todavía).
> Cuando estabilice se migrará a Terraform o CDK.

---

## 1. IAM

### 1.1 Usuario IAM para CI/CD

- TODO: documentar política mínima (push a ECR, update Lambda, R/W en S3 buckets, R/W en DynamoDB).

### 1.2 Rol de ejecución de la Lambda

- TODO: política con permisos para escribir en S3 (clips flagged), DynamoDB y CloudWatch logs.

### 1.3 Rol OIDC para GitHub Actions

- TODO: configurar trust policy para `repo:Placaflaca00/bird_classifierPY:*`.

---

## 2. S3

### 2.1 Bucket de modelos (`${S3_BUCKET_MODELS}`)

- TODO: versioning ON, lifecycle policy para transicionar a IA después de 30 días.

### 2.2 Bucket de audios flagged (`${S3_BUCKET_FLAGGED_AUDIO}`)

- TODO: versioning ON, encryption SSE-S3, bloqueo de acceso público.

---

## 3. DynamoDB

### 3.1 Tabla de flags (`${DYNAMODB_TABLE_FLAGS}`)

- TODO: definir partition key (`recording_id`), sort key (`timestamp`), atributos (predicted_label, user_label, model_version).

---

## 4. ECR

### 4.1 Repositorio (`${ECR_REPOSITORY}`)

- TODO: scan-on-push ON, lifecycle policy para mantener solo las últimas N imágenes.

---

## 5. Lambda

### 5.1 Función (`${LAMBDA_FUNCTION_NAME}`)

- TODO: container image desde ECR, memoria 2048 MB, timeout 30s, env vars desde el bloque AWS de `.env.example`.

---

## 6. API Gateway

### 6.1 HTTP API

- TODO: integración Lambda proxy, throttling 10 req/s burst, CORS para el dominio de HF Spaces.

---

## 7. Verificación end-to-end

- TODO: lista de comandos curl para validar que el endpoint responde correctamente.
