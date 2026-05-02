"""Generador de reportes EvidentlyAI.

Compara la distribución de embeddings/predicciones del set de referencia
(training) contra las queries reales que llegan al endpoint de Lambda
(loggeadas a S3 + DynamoDB).

Reportes a generar:
- **Data drift** : drift por dimensión sobre los 320-dim del embedding.
- **Prediction drift** : drift en la distribución de clases predichas.
- **Performance** : si llegan labels via flagging del usuario, accuracy en producción.

API esperada:
    generate_report(reference_df, current_df, out_html_path)

TODO: implementar usando ``evidently.Report`` con los presets correspondientes.
"""
