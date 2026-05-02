"""Función ``train()`` invocada desde notebooks de Colab.

API esperada:

    from src.training.train import train
    train(config_overrides={"epochs": 50, "lr": 3e-4})

Responsabilidades:
1. Cargar config (Pydantic Settings) y aplicar ``config_overrides``.
2. Construir DataLoaders (train/val) vía ``src.data.dataset``.
3. Instanciar ``BirdClassifier`` (``src.models.classifier``).
4. Configurar logger W&B (``WandbLogger``) y callbacks:
   - ``ModelCheckpoint`` (monitor val_macro_f1).
   - ``EarlyStopping`` (patience configurable).
   - ``LearningRateMonitor``.
5. ``pl.Trainer(...).fit(model, dm)`` y luego ``.test(...)``.
6. Devolver el path del mejor checkpoint y el ``run.id`` de W&B.

TODO: implementar.
"""
