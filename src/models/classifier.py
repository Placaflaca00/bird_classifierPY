"""Cabezal clasificador sobre embeddings BirdNET (320-dim).

Define ``BirdClassifier(pl.LightningModule)``:

Arquitectura propuesta (MLP):
    Linear(320 → 256) → ReLU → Dropout(p)
    Linear(256 → 128) → ReLU → Dropout(p)
    Linear(128 → num_classes)

Hooks de Lightning a implementar:
- ``__init__(num_classes, lr, weight_decay, dropout)``
- ``forward(x)``
- ``training_step``  : CrossEntropy + log loss/acc.
- ``validation_step``: log val_loss / val_acc / macro-F1.
- ``test_step``      : idem, separado por split.
- ``configure_optimizers``: AdamW + ReduceLROnPlateau (o cosine schedule).

Métricas con ``torchmetrics``:
- Accuracy, MacroF1, ConfusionMatrix (loggeada a W&B al final del val epoch).

TODO: implementar la clase y exportarla.
"""
