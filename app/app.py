"""Frontend Gradio para HuggingFace Spaces.

Construye un ``gr.Interface`` (o ``gr.Blocks``) que:
- Recibe un archivo de audio (gr.Audio).
- Lo envía al endpoint AWS API Gateway (URL en ``API_GATEWAY_URL``).
- Renderiza top-K predicciones con probabilidades.
- Permite al usuario marcar la predicción como correcta/incorrecta y, en ese caso,
  delega a ``app.flagger.APIGatewayFlagger`` que sube el clip a S3 + escribe
  fila en DynamoDB.

Pensado para ser ejecutado por HF Spaces como ``app.py``.

TODO: implementar la UI y el cliente HTTP.
"""
