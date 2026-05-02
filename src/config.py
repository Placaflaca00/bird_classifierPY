"""Configuración centralizada del proyecto (Pydantic Settings).

Define una clase ``Settings`` (BaseSettings) que carga variables desde:
1. Variables de entorno del proceso.
2. Archivo ``.env`` en la raíz del repo.

Campos esperados (ver ``.env.example`` para la lista completa):

- W&B           : ``WANDB_API_KEY``, ``WANDB_PROJECT``, ``WANDB_ENTITY``
- HuggingFace   : ``HF_TOKEN``, ``HF_USERNAME``, ``HF_SPACE_NAME``
- AWS           : ``AWS_REGION``, ``AWS_ACCOUNT_ID``, credenciales,
                  ``S3_BUCKET_MODELS``, ``S3_BUCKET_FLAGGED_AUDIO``,
                  ``DYNAMODB_TABLE_FLAGS``, ``ECR_REPOSITORY``,
                  ``LAMBDA_FUNCTION_NAME``, ``API_GATEWAY_URL``
- Xeno-canto    : ``XENOCANTO_API_KEY`` (opcional)
- Paths locales : ``DATA_DIR``, ``RAW_AUDIO_DIR``, ``PROCESSED_DIR``,
                  ``TEST_SETS_DIR``, ``MODELS_DIR``

Patrón de uso:

    from src.config import settings
    print(settings.wandb_project)

TODO: implementar la clase Settings con sus validators y un singleton ``settings``.
"""
