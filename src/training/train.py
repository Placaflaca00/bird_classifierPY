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

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
    # Especies a excluir del mapping y de los datasets (sin tocar los parquets).
    # Útil para descartar clases con poca data o problemáticas. Tupla, no lista,
    # para que sea hashable y serialice limpio en W&B config.
    "drop_species": (),
    "early_stopping_patience": 8,
    "seed": 42,
    "run_name": "baseline-v0",
    "tags": ["baseline", "no-augmentation"],
    # Refs a artifacts de input (lineage). Override desde el notebook para
    # apuntar a versiones aumentadas u otras splits.
    "embeddings_artifact": "embeddings:v0",
    "splits_artifact": "splits:v0",
    # Modo local: usa CSVLogger en vez de WandbLogger. No llama use_artifact
    # ni loggea a W&B summary. Las confmat se guardan como PNG en checkpoints/<run>/.
    "local": False,
}


def train(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config_overrides or {})}
    L.seed_everything(cfg["seed"], workers=True)

    load_dotenv(ROOT / ".env")
    if not cfg["local"]:
        project = os.environ["WANDB_PROJECT"].strip()
        entity = os.environ["WANDB_ENTITY"].strip()
    else:
        project = entity = None

    drop_species = list(cfg["drop_species"])
    species_to_idx = load_species_mapping(drop_species=drop_species)
    num_classes = len(species_to_idx)
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    class_names = [idx_to_species[i] for i in range(num_classes)]
    if drop_species:
        print(f"Especies excluidas: {drop_species}")

    loaders = build_dataloaders(
        batch_size=cfg["batch_size"], drop_species=drop_species
    )
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

    if cfg["local"]:
        logger = CSVLogger(
            save_dir=str(ROOT / "logs"),
            name=cfg["run_name"],
        )
    else:
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

    # Resumen: si W&B, va a summary; si local, sólo print (ya impreso arriba).
    if not cfg["local"]:
        for fold, metrics in final_results.items():
            for k, v in metrics.items():
                clean_k = k.replace("test_", "")
                logger.experiment.summary[f"final/{fold}_{clean_k}"] = v

    # Matrices de confusión para test_clean y test_hard usando el best ckpt.
    # Recargo el modelo desde el checkpoint (los pesos en memoria pueden no
    # ser los del best — trainer.test no muta el modelo en memoria).
    # strict=False: tolera checkpoints viejos que guardaron class_weights como
    # buffer persistente (ya no lo hacemos, pero el flag es defensivo).
    best_model = BirdClassifier.load_from_checkpoint(
        ckpt_cb.best_model_path, strict=False
    )
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
        if cfg["local"]:
            cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(12, 12))
            disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title(f"{cfg['run_name']} — confmat {fold}")
            fig.tight_layout()
            png_path = ckpt_dir / f"confmat_{fold}.png"
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"  confmat/{fold} -> {png_path}")
        else:
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


def _parse_cli() -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", action="store_true",
                   help="No usa W&B. CSVLogger + confmat PNG local.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=None)
    p.add_argument("--drop-species", nargs="*", default=None,
                   help="Lista de species a excluir (formato científico).")
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--embeddings-artifact", default=None,
                   help="Artifact W&B de embeddings (formato 'name:version'). Solo aplica sin --local.")
    p.add_argument("--splits-artifact", default=None,
                   help="Artifact W&B de splits.")
    args = p.parse_args()
    overrides: dict[str, Any] = {}
    if args.local: overrides["local"] = True
    if args.run_name: overrides["run_name"] = args.run_name
    if args.epochs is not None: overrides["max_epochs"] = args.epochs
    if args.lr is not None: overrides["lr"] = args.lr
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.dropout is not None: overrides["dropout"] = args.dropout
    if args.use_class_weights: overrides["use_class_weights"] = True
    if args.mixup_alpha is not None: overrides["mixup_alpha"] = args.mixup_alpha
    if args.drop_species is not None: overrides["drop_species"] = tuple(args.drop_species)
    if args.tags is not None: overrides["tags"] = list(args.tags)
    if args.embeddings_artifact: overrides["embeddings_artifact"] = args.embeddings_artifact
    if args.splits_artifact: overrides["splits_artifact"] = args.splits_artifact
    return overrides


if __name__ == "__main__":
    train(_parse_cli())
