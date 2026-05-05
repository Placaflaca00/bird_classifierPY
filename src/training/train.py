"""Entry point de entrenamiento. Llamado desde notebooks o CLI.

Uso desde notebook (Colab):
    from src.training.train import train
    train()                             # baseline default
    train({"max_epochs": 100, "lr": 5e-4})

Uso CLI:
    python -m src.training.train

Pipeline:
    1. Cargar embeddings:v0 + splits:v0 (lineage W&B).
    2. Construir DataLoaders (train/val/test_clean/test_hard).
    3. Instanciar BirdClassifier.
    4. Trainer + WandbLogger + callbacks (Checkpoint, EarlyStopping, LR monitor).
    5. trainer.fit().
    6. trainer.test() sobre val + test_clean + test_hard, reportar a W&B summary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders, load_species_mapping  # noqa: E402
from src.models.classifier import BirdClassifier  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "batch_size": 64,
    "max_epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "hidden_dims": (256, 128),
    "mixup_alpha": 0.0,
    "use_class_weights": False,
    "early_stopping_patience": 8,
    "seed": 42,
    "run_name": "baseline-v0",
    "tags": ["baseline", "no-augmentation"],
    # Refs a artifacts de input (lineage). Override desde el notebook para
    # apuntar a versiones aumentadas u otras splits.
    "embeddings_artifact": "embeddings:v0",
    "splits_artifact": "splits:v0",
}


def train(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config_overrides or {})}
    L.seed_everything(cfg["seed"], workers=True)

    load_dotenv(ROOT / ".env")
    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()

    species_to_idx = load_species_mapping()
    num_classes = len(species_to_idx)
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    class_names = [idx_to_species[i] for i in range(num_classes)]

    loaders = build_dataloaders(batch_size=cfg["batch_size"])
    print(
        f"Sizes: train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} "
        f"test_clean={len(loaders['test_clean'].dataset)} "
        f"test_hard={len(loaders['test_hard'].dataset)}"
    )
    print(f"num_classes: {num_classes}")

    # Class weights estilo sklearn-balanced: w_c = N / (C * n_c)
    # Solo se aplican a train_loss (val/test quedan sin pesar para mantener
    # las métricas comparables con runs anteriores).
    class_weights_t: torch.Tensor | None = None
    if cfg["use_class_weights"]:
        train_labels = loaders["train"].dataset.labels.numpy()
        counts = np.bincount(train_labels, minlength=num_classes)
        if (counts == 0).any():
            missing = [class_names[i] for i, c in enumerate(counts) if c == 0]
            raise RuntimeError(f"Clases sin ejemplos en train: {missing}")
        weights = train_labels.shape[0] / (num_classes * counts)
        class_weights_t = torch.tensor(weights, dtype=torch.float32)
        print(
            f"Class weights: min={weights.min():.3f}  max={weights.max():.3f}  "
            f"mean={weights.mean():.3f}"
        )

    model = BirdClassifier(
        num_classes=num_classes,
        embedding_dim=1024,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mixup_alpha=cfg["mixup_alpha"],
        class_weights=class_weights_t,
    )

    logger = WandbLogger(
        project=project,
        entity=entity,
        name=cfg["run_name"],
        job_type="train",
        tags=cfg["tags"],
        config=cfg,
        save_dir=str(ROOT / "wandb"),
    )
    # Lineage: declarar inputs ANTES de fit
    logger.experiment.use_artifact(cfg["embeddings_artifact"])
    logger.experiment.use_artifact(cfg["splits_artifact"])

    ckpt_dir = ROOT / "checkpoints" / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_macro_f1:.3f}",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_cb = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=cfg["early_stopping_patience"],
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="cpu",
        logger=logger,
        callbacks=[ckpt_cb, early_cb, lr_cb],
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best val_macro_f1: {ckpt_cb.best_model_score:.4f}")

    # Eval final sobre los 3 sets, usando el mejor checkpoint
    print("\n=== Evaluación final con best checkpoint ===")
    final_results: dict[str, dict[str, float]] = {}
    for fold in ("val", "test_clean", "test_hard"):
        if len(loaders[fold].dataset) == 0:
            continue
        out = trainer.test(
            model,
            dataloaders=loaders[fold],
            ckpt_path=ckpt_cb.best_model_path,
            verbose=False,
        )
        final_results[fold] = {k: float(v) for k, v in out[0].items()}
        print(
            f"  {fold:<11} loss={final_results[fold]['test_loss']:.4f}  "
            f"acc={final_results[fold]['test_acc']:.4f}  "
            f"macro_f1={final_results[fold]['test_macro_f1']:.4f}"
        )

    # Resumen en W&B summary (con prefijo claro)
    for fold, metrics in final_results.items():
        for k, v in metrics.items():
            clean_k = k.replace("test_", "")
            logger.experiment.summary[f"final/{fold}_{clean_k}"] = v

    # Matrices de confusión para test_clean y test_hard usando el best ckpt.
    # Recargo el modelo desde el checkpoint (los pesos en memoria pueden no
    # ser los del best — trainer.test no muta el modelo en memoria).
    best_model = BirdClassifier.load_from_checkpoint(ckpt_cb.best_model_path)
    best_model.eval()
    for fold in ("test_clean", "test_hard"):
        if len(loaders[fold].dataset) == 0:
            continue
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in loaders[fold]:
                logits = best_model(x)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        logger.experiment.log(
            {
                f"confmat/{fold}": wandb.plot.confusion_matrix(
                    y_true=labels.tolist(),
                    preds=preds.tolist(),
                    class_names=class_names,
                )
            }
        )
        print(f"  confmat/{fold} loggeada a W&B")

    return {
        "best_ckpt": ckpt_cb.best_model_path,
        "best_val_macro_f1": float(ckpt_cb.best_model_score),
        "final_results": final_results,
    }


if __name__ == "__main__":
    train()
