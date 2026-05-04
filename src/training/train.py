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
    "early_stopping_patience": 8,
    "seed": 42,
    "run_name": "baseline-v0",
    "tags": ["baseline", "no-augmentation"],
}


def train(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config_overrides or {})}
    L.seed_everything(cfg["seed"], workers=True)

    load_dotenv(ROOT / ".env")
    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()

    species_to_idx = load_species_mapping()
    num_classes = len(species_to_idx)

    loaders = build_dataloaders(batch_size=cfg["batch_size"])
    print(
        f"Sizes: train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} "
        f"test_clean={len(loaders['test_clean'].dataset)} "
        f"test_hard={len(loaders['test_hard'].dataset)}"
    )
    print(f"num_classes: {num_classes}")

    model = BirdClassifier(
        num_classes=num_classes,
        embedding_dim=1024,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mixup_alpha=cfg["mixup_alpha"],
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
    logger.experiment.use_artifact("embeddings:v0")
    logger.experiment.use_artifact("splits:v0")

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

    return {
        "best_ckpt": ckpt_cb.best_model_path,
        "best_val_macro_f1": float(ckpt_cb.best_model_score),
        "final_results": final_results,
    }


if __name__ == "__main__":
    train()
