"""bird_classifier_py — paquete principal.

Expone subpaquetes:
- ``src.config``       : configuración centralizada (Pydantic Settings).
- ``src.data``         : schemas, Dataset, augmentations, descarga.
- ``src.models``       : LightningModule del clasificador sobre embeddings BirdNET.
- ``src.training``     : función ``train()`` invocada desde notebooks de Colab.
- ``src.monitoring``   : generación de reportes EvidentlyAI.
"""

__version__ = "0.0.1"
