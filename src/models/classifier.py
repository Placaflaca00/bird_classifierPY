"""Cabezal MLP sobre embeddings BirdNET (1024-dim).

Arquitectura:
    Linear(1024 -> 256) -> ReLU -> Dropout
    Linear(256  -> 128) -> ReLU -> Dropout
    Linear(128  -> num_classes)

Decisiones:
- AdamW + ReduceLROnPlateau sobre val_loss (paciencia 3, factor 0.5).
- CrossEntropyLoss directo sobre logits.
- Métricas: Accuracy y MacroF1 (torchmetrics) — macro-F1 es la métrica primaria
  porque el dataset está desbalanceado.
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class BirdClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 23,
        embedding_dim: int = 1024,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_macro_f1", self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_macro_f1", self.test_f1)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
