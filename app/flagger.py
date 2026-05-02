"""Custom flagging callback para Gradio.

``APIGatewayFlagger(gr.FlaggingCallback)`` que en lugar de escribir un CSV local
(comportamiento default de Gradio):
- Hace POST al endpoint ``${API_GATEWAY_URL}/flag`` con:
    * el audio (multipart),
    * la predicción del modelo,
    * la corrección del usuario (label real, opcional),
    * timestamp y user-agent.
- El backend persiste el clip en ``s3://${S3_BUCKET_FLAGGED_AUDIO}/`` y
  escribe metadata en ``${DYNAMODB_TABLE_FLAGS}``.

Estos clips alimentan el siguiente ciclo de re-entrenamiento.

TODO: implementar la subclase y registrarla en ``app.py``.
"""
