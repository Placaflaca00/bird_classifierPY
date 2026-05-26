"""Entrena el MLP head para candidato de Fase 6 retraining loop.

Consume el parquet self-contained generado por ``scripts/export_dataset.py``
(con columnas filepath, species, fold, embedding, source, ddb_pk, ...) y
entrena un ``BirdClassifier`` desde scratch (Enfoque C: from scratch + replay
balanceado). Output: checkpoint del best model + metricas JSON + confusion
matrices PNG. NO exporta a ONNX (eso es trabajo de ``promote.py``).

Decisiones (consensuadas 2026-05-25)
====================================
1. **Standalone**, no wrapper sobre ``src/training/train.py``. Reusa solo el
   modelo ``BirdClassifier`` de ``src/models/classifier.py``. Esto desacopla
   Fase 6 del legacy y da un entrypoint claro para el workflow GH Actions.
   El test de equivalencia (``tests/test_train_equivalence.py``) garantiza
   que ambos paths producen modelos identicos bit-a-bit.

2. **Hyperparams hardcoded** a defaults de ``wa-drop3-v1`` (sin override
   por default; flag ``--config`` opcional para sweep futuro). Razon: el
   loop de Fase 6 cambia el DATASET (agrega approved del annotator) para
   ver si mejora performance. Si tambien cambias hyperparams, los efectos
   se confunden (confounding variable) — no se puede atribuir la mejora al
   nuevo dataset. Aislar la variable nueva es metodologia experimental
   basica.

3. **Output a `checkpoints/retrain-<ts>/`** (cada run su carpeta). No toca
   `models/classifier.onnx` que ES wa-drop3-v1 en produccion. Solo
   ``promote.py`` mueve un candidato a produccion.

4. **No ONNX export**. Eso es trabajo de ``promote.py`` solo si el
   candidato gana en McNemar. Si no gana, no se exporta nada.

Mejoras adicionales
===================
- **Determinismo robusto**: ``torch.use_deterministic_algorithms(True)`` +
  cudnn flags + CUBLAS_WORKSPACE_CONFIG. Hoy corre en CPU pero futureproof
  para GPU runs.
- **Sanity checks pre-train**: verifica counts por especie por fold,
  invariante D20 (cero items annotator en test_clean/test_hard), shape de
  embedding (1024,), sin NaN. Si falla, aborta antes del fit.
- **W&B run naming**: ``retrain-<timestamp>-<git_sha[:7]>`` para tracking
  cross-run.

Usage
=====
    # Dry-run: reporta sanity checks + config, no entrena
    python scripts/train.py --dataset data/processed/embeddings_retrain_<ts>.parquet

    # Smoke: 2 epochs para verificar pipeline
    python scripts/train.py --dataset <path> --apply --max-epochs 2

    # Run completo (defaults wa-drop3-v1)
    python scripts/train.py --dataset <path> --apply

    # Con W&B
    python scripts/train.py --dataset <path> --apply --upload-wandb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Determinismo: setear ANTES de importar torch para que cuBLAS lo respete.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import lightning as L  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

# Defaults consistentes con wa-drop3-v1 (Fase 5). Ver Etapa 4 de
# RETRAINING_STRATEGY.txt para justificacion. CAMBIAR ESTOS DEFAULTS rompe
# la comparabilidad con modelos anteriores.
DEFAULTS = {
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
    "embedding_dim": 1024,
}

VALID_FOLDS = {"train", "val", "test_clean", "test_hard"}
MIN_PER_SPECIES_TRAIN_WARN = 10  # warn si una especie tiene <N en train


# ---------------------------------------------------------------------------
# Args + reproducibility setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True,
                   help="Path al parquet self-contained de export_dataset.py")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: checkpoints/retrain-<ts>/")
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="Default: reports/retrain-<ts>/")
    p.add_argument("--apply", action="store_true",
                   help="Ejecuta entrenamiento. Default es dry-run (solo sanity checks).")
    p.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"],
                   help=f"Default {DEFAULTS['max_epochs']}.")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help=f"Default {DEFAULTS['seed']}, mismo que wa-drop3-v1.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Default: retrain-<ts>-<git_sha[:7]>.")
    p.add_argument("--upload-wandb", action="store_true",
                   help="Sube modelo + metricas como W&B Artifact.")
    return p.parse_args()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def setup_determinism(seed: int) -> None:
    """Determinismo robusto. Hoy corre en CPU pero futureproof para GPU runs.

    Combinacion completa: Lightning ``seed_everything`` + cuDNN deterministic
    + CUBLAS workspace fijo + torch use_deterministic_algorithms(True).
    """
    L.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUBLAS_WORKSPACE_CONFIG ya seteado a nivel modulo antes de importar torch.


# ---------------------------------------------------------------------------
# Dataset + DataLoader (lee parquet self-contained)
# ---------------------------------------------------------------------------

class ParquetDataset(Dataset):
    """Lee fold especifico del parquet self-contained.

    El parquet ya tiene columna ``fold`` autoritativa (no hace merge con un
    splits.parquet separado). Esto difiere de ``src/data/dataset.py`` legacy
    que mergea embeddings + splits.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fold: str,
        species_to_idx: dict[str, int],
    ) -> None:
        sub = df[df["fold"] == fold].reset_index(drop=True)
        # Filtrar augmentations para non-train (legacy de embeddings.parquet).
        if fold != "train" and "is_aug" in sub.columns:
            sub = sub[~sub["is_aug"]].reset_index(drop=True)

        self.fold = fold
        if len(sub) == 0:
            self.embeddings = torch.empty((0, DEFAULTS["embedding_dim"]), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.long)
            return

        emb_arr = np.stack(sub["embedding"].to_numpy()).astype(np.float32)
        self.embeddings = torch.from_numpy(emb_arr)
        self.labels = torch.tensor(
            sub["species"].map(species_to_idx).to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def build_loaders(
    df: pd.DataFrame,
    species_to_idx: dict[str, int],
    batch_size: int,
    seed: int,  # kept for signature symmetry; usado via seed_everything global
) -> dict[str, DataLoader]:
    """4 DataLoaders, shuffle solo en train. NO pasa ``generator`` explicito —
    usa el global RNG state seteado por ``L.seed_everything(seed)``. Esto
    garantiza equivalencia con ``src/training/train.py`` legacy que tampoco
    pasa generator (ver tests/test_train_equivalence.py).
    """
    loaders = {}
    for fold in ("train", "val", "test_clean", "test_hard"):
        ds = ParquetDataset(df, fold, species_to_idx)
        loaders[fold] = DataLoader(
            ds, batch_size=batch_size, shuffle=(fold == "train"),
            num_workers=0, drop_last=False,
        )
    return loaders


# ---------------------------------------------------------------------------
# Sanity checks pre-train (aborta antes de gastar compute)
# ---------------------------------------------------------------------------

def sanity_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Valida el dataset antes del fit. Levanta RuntimeError si algo
    invariante esta roto. Devuelve dict con counts + warnings para report.
    """
    problems: list[str] = []
    warnings: list[str] = []

    # 1. Folds validos
    invalid_folds = set(df["fold"].unique()) - VALID_FOLDS
    if invalid_folds:
        problems.append(f"folds invalidos: {invalid_folds}")

    # 2. Shape del embedding
    bad_shapes = df["embedding"].apply(
        lambda x: not (isinstance(x, np.ndarray) and x.shape == (1024,))
    ).sum()
    if bad_shapes > 0:
        problems.append(f"{bad_shapes} filas con embedding shape != (1024,)")

    # 3. NaN en embeddings
    nan_count = df["embedding"].apply(
        lambda x: bool(np.isnan(x).any()) if isinstance(x, np.ndarray) else True
    ).sum()
    if nan_count > 0:
        problems.append(f"{nan_count} filas con NaN en embedding")

    # 4. INVARIANTE D20: cero items annotator en hold-out
    if "source" in df.columns:
        for hold_fold in ("test_clean", "test_hard"):
            n_annot_in_hold = (
                (df["fold"] == hold_fold) & (df["source"] == "annotator")
            ).sum()
            if n_annot_in_hold > 0:
                problems.append(
                    f"VIOLACION D20: {n_annot_in_hold} items annotator en "
                    f"fold={hold_fold}. El hold-out debe ser solo originales."
                )

    # 5. Counts por especie en train (warn si < 10)
    train_counts = df[df["fold"] == "train"]["species"].value_counts()
    for sp, n in train_counts.items():
        if n < MIN_PER_SPECIES_TRAIN_WARN:
            warnings.append(f"  WARN: {sp} tiene solo {n} samples en train (< {MIN_PER_SPECIES_TRAIN_WARN})")

    if problems:
        raise RuntimeError(
            "Sanity checks FAILED:\n  - " + "\n  - ".join(problems)
            + "\n\nDataset corrupto o invariantes rotas. Fix antes de entrenar."
        )

    fold_counts = df["fold"].value_counts().to_dict()
    source_counts_train = {}
    if "source" in df.columns:
        train_only = df[df["fold"] == "train"]
        source_counts_train = train_only["source"].value_counts().to_dict()

    return {
        "fold_counts": fold_counts,
        "source_counts_train": source_counts_train,
        "n_species": len(df["species"].unique()),
        "species_counts_train": train_counts.to_dict(),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    species_to_idx: dict[str, int],
    output_dir: Path,
    reports_dir: Path,
    run_name: str,
    max_epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Entrena. Devuelve dict con best_ckpt + metrics. Es lo que evaluate.py
    y promote.py consumen.
    """
    num_classes = len(species_to_idx)
    class_names = [
        s for s, _ in sorted(species_to_idx.items(), key=lambda kv: kv[1])
    ]

    loaders = build_loaders(df, species_to_idx, DEFAULTS["batch_size"], seed)
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"DataLoader sizes: {sizes}")

    model = BirdClassifier(
        num_classes=num_classes,
        embedding_dim=DEFAULTS["embedding_dim"],
        hidden_dims=DEFAULTS["hidden_dims"],
        dropout=DEFAULTS["dropout"],
        lr=DEFAULTS["lr"],
        weight_decay=DEFAULTS["weight_decay"],
        mixup_alpha=DEFAULTS["mixup_alpha"],
        class_weights=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:02d}-{val_macro_f1:.3f}",
        monitor="val_macro_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    early_cb = EarlyStopping(
        monitor="val_macro_f1", mode="max",
        patience=DEFAULTS["early_stopping_patience"],
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir=str(reports_dir), name=run_name)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=logger,
        callbacks=[ckpt_cb, early_cb, lr_cb],
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"])

    print(f"\nBest ckpt: {ckpt_cb.best_model_path}")
    print(f"Best val_macro_f1: {ckpt_cb.best_model_score:.4f}")

    # Eval final sobre los 3 folds con best checkpoint
    final_metrics: dict[str, dict[str, float]] = {}
    for fold in ("val", "test_clean", "test_hard"):
        if sizes[fold] == 0:
            continue
        out = trainer.test(
            model, dataloaders=loaders[fold],
            ckpt_path=ckpt_cb.best_model_path, verbose=False,
        )
        final_metrics[fold] = {k: float(v) for k, v in out[0].items()}
        m = final_metrics[fold]
        print(f"  {fold:<11s}  loss={m['test_loss']:.4f}  "
              f"acc={m['test_acc']:.4f}  macro_f1={m['test_macro_f1']:.4f}")

    # Confusion matrices con best ckpt (reload de disco para asegurar pesos)
    best = BirdClassifier.load_from_checkpoint(ckpt_cb.best_model_path, strict=False)
    best.eval()
    for fold in ("val", "test_clean", "test_hard"):
        if sizes[fold] == 0:
            continue
        preds: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in loaders[fold]:
                logits = best(x)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(y.cpu().numpy())
        p = np.concatenate(preds)
        l = np.concatenate(labels)
        cm = confusion_matrix(l, p, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Blues",
                  values_format="d", colorbar=False)
        ax.set_title(f"{run_name} — confmat {fold}")
        fig.tight_layout()
        png_path = reports_dir / f"confmat_{fold}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  confmat {fold} -> {png_path}")

    return {
        "best_ckpt": ckpt_cb.best_model_path,
        "best_val_macro_f1": float(ckpt_cb.best_model_score),
        "final_metrics": final_metrics,
        "sizes": sizes,
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# State dict SHA-256 (para test de equivalencia y audit)
# ---------------------------------------------------------------------------

def state_dict_sha256(ckpt_path: Path) -> str:
    """SHA-256 estable del state_dict (sin metadata Lightning). Usado por
    tests/test_train_equivalence.py.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode("utf-8"))
        h.update(state[key].cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# W&B upload opcional
# ---------------------------------------------------------------------------

def upload_to_wandb(
    ckpt_path: Path, metrics: dict, alias: str, run_name: str, dataset_path: Path,
) -> None:
    import wandb
    project = os.environ["WANDB_PROJECT"].strip()
    entity = os.environ["WANDB_ENTITY"].strip()
    run = wandb.init(
        entity=entity, project=project, job_type="train-retrain",
        name=run_name,
        notes="Candidato Fase 6 retraining loop.",
    )
    artifact = wandb.Artifact(
        name="model_retrain", type="model",
        metadata={
            "dataset_path": str(dataset_path.name),
            "metrics": metrics,
            "git_sha": git_sha(),
            "state_dict_sha256": state_dict_sha256(ckpt_path),
        },
    )
    artifact.add_file(str(ckpt_path))
    run.log_artifact(artifact, aliases=[alias])
    run.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sha = git_sha()
    run_name = args.run_name or f"retrain-{ts}-{sha[:7]}"
    output_dir = args.output_dir or (ROOT / "checkpoints" / run_name)
    reports_dir = args.reports_dir or (ROOT / "reports" / run_name)

    print("=== Train (Fase 6.3) ===")
    print(f"  run_name:     {run_name}")
    print(f"  dataset:      {args.dataset}")
    print(f"  max_epochs:   {args.max_epochs}")
    print(f"  seed:         {args.seed}")
    print(f"  output_dir:   {output_dir}")
    print(f"  reports_dir:  {reports_dir}")
    print(f"  git_sha:      {sha}")
    print()

    if not args.dataset.exists():
        print(f"! Dataset no existe: {args.dataset}", file=sys.stderr)
        return 1

    print("Cargando dataset...")
    df = pd.read_parquet(args.dataset)
    print(f"  filas: {len(df)}")

    print("\nSanity checks...")
    info = sanity_checks(df)
    print(f"  folds: {info['fold_counts']}")
    print(f"  source en train: {info['source_counts_train']}")
    print(f"  n_species: {info['n_species']}")
    if info["warnings"]:
        print("  Warnings:")
        for w in info["warnings"]:
            print(w)
    print("  OK (D20 + shape + NaN validados)")

    species_to_idx = {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}
    print(f"\nspecies_to_idx (alfabetico, {len(species_to_idx)} clases):")
    for sp, i in species_to_idx.items():
        print(f"  {i:2d}  {sp}")

    if not args.apply:
        print("\nDRY-RUN: agregar --apply para entrenar.")
        return 0

    # Setup determinismo DESPUES de sanity (sanity no es estocastico).
    setup_determinism(args.seed)

    print(f"\nEntrenando (max_epochs={args.max_epochs})...")
    result = train_model(
        df, species_to_idx, output_dir, reports_dir, run_name,
        args.max_epochs, args.seed,
    )

    # Persistir metrics.json + sha del state dict
    sd_sha = state_dict_sha256(Path(result["best_ckpt"]))
    metrics_blob = {
        "run_name": run_name,
        "git_sha": sha,
        "dataset_path": str(args.dataset),
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "hyperparams": {k: v for k, v in DEFAULTS.items() if k != "embedding_dim"},
        "best_ckpt": result["best_ckpt"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "final_metrics": result["final_metrics"],
        "sizes": result["sizes"],
        "state_dict_sha256": sd_sha,
    }
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_blob, f, indent=2)
    print(f"\nMetrics: {metrics_path}")
    print(f"state_dict SHA-256: {sd_sha}")

    if args.upload_wandb:
        alias = run_name  # mismo nombre que el local
        print(f"\nSubiendo a W&B como model_retrain:{alias}...")
        upload_to_wandb(
            Path(result["best_ckpt"]), metrics_blob, alias, run_name, args.dataset
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
