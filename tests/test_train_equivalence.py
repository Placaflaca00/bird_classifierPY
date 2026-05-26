"""Test de equivalencia entre scripts/train.py y src/training/train.py.

Mitiga drift entre el path nuevo de Fase 6 (``scripts/train.py``, lee parquet
self-contained de export_dataset.py) y el path legacy (``src/training/train.train()``,
lee ``embeddings.parquet`` + ``splits.parquet`` mergeados).

Si los scripts divergen en data loading, hyperparam handling, o orden de seed
setup, este test detecta el drift loud — antes de que un retrain de Fase 6
produzca un modelo distinto del que producirian los notebooks de Colab.

Estrategia
==========
1. Construye mini dataset (50 samples) en tmp_path en ambos formatos.
2. ``monkeypatch`` los paths globales de ``src.data.dataset`` para que legacy
   lea de los mini.
3. Llama ambas funciones de training directamente (no subprocess) con misma
   seed y misma config.
4. Compara los state_dicts. Tolerancia inicial muy laxa (atol=1e-4) porque
   bit-perfect entre dos paths de Lightning con orden distinto de
   seed_everything es dificil. Si pasa, sube a SHA-256 estricto.

Si el test falla, NO es un fix mecanico — significa que las semánticas de
entrenamiento divergieron. Hay que investigar QUE cambio antes de modificar
el assertion.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def _build_mini_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Construye mini dataset en ambos formatos a partir del embeddings.parquet
    original. Returns (mini_combined, mini_emb_legacy, mini_splits_legacy).

    Sampling: 50 audios estratificados, balanceados entre folds (35/5/5/5)
    para que ambos scripts tengan val/test no vacios.
    """
    src_emb = ROOT / "data" / "processed" / "embeddings.parquet"
    src_splits = ROOT / "data" / "processed" / "splits.parquet"
    if not src_emb.exists() or not src_splits.exists():
        pytest.skip(
            "Necesita embeddings.parquet + splits.parquet locales. "
            "Skipea en CI si no estan."
        )

    emb = pd.read_parquet(src_emb)
    splits = pd.read_parquet(src_splits)
    merged = emb.merge(splits, on="filepath", how="inner")
    merged = merged[~merged["is_aug"]].reset_index(drop=True)

    # Sampling DIVERSO: para cada (fold, especie), toma las primeras N filas.
    # Asi el mini dataset cubre TODAS las especies, no solo las primeras
    # alfabeticamente. Sin esto, num_classes=1 (un solo folder).
    per_fold_per_species = {"train": 2, "val": 1, "test_clean": 1, "test_hard": 1}
    parts: list[pd.DataFrame] = []
    for fold, n in per_fold_per_species.items():
        fold_df = merged[merged["fold"] == fold]
        for sp, group in fold_df.groupby("species"):
            parts.append(group.head(n))
    mini = pd.concat(parts, ignore_index=True)
    # Sanidad: todas las especies presentes en cada fold (necesario para que
    # torchmetrics no falle con num_classes=1 en algun fold).
    for fold in per_fold_per_species:
        n_sp = mini[mini["fold"] == fold]["species"].nunique()
        assert n_sp >= 2, f"mini fold={fold} tiene solo {n_sp} especie(s)"

    # Formato legacy: embeddings.parquet (sin fold) + splits.parquet (con fold)
    mini_emb = tmp_path / "embeddings.parquet"
    mini_splits = tmp_path / "splits.parquet"
    mini[["filepath", "species", "split", "embedding", "is_aug", "aug_id"]].to_parquet(
        mini_emb, index=False
    )
    mini[["filepath", "fold"]].to_parquet(mini_splits, index=False)

    # Formato self-contained: parquet con fold + source + audit columns
    mini_combined = tmp_path / "embeddings_mini_combined.parquet"
    mini_self = mini.copy()
    mini_self["source"] = "original"
    mini_self["ddb_pk"] = pd.NA
    mini_self["generated_at"] = "2026-05-25T00:00:00Z"
    mini_self["git_sha"] = "test-fixed"
    # Asegurar embedding dtype float32 (igual que produce export_dataset).
    mini_self["embedding"] = mini_self["embedding"].apply(
        lambda x: x.astype(np.float32) if x.dtype != np.float32 else x
    )
    mini_self.to_parquet(mini_combined, index=False)

    return mini_combined, mini_emb, mini_splits


def _state_dict_close(ckpt_a: Path, ckpt_b: Path, atol: float, rtol: float) -> dict:
    """Compara dos state_dicts. Returns info dict con max_diff per layer
    y bool overall_close.
    """
    a = torch.load(ckpt_a, map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(ckpt_b, map_location="cpu", weights_only=False)["state_dict"]
    assert set(a.keys()) == set(b.keys()), (
        f"Keys distintos: a={set(a.keys())} vs b={set(b.keys())}"
    )
    diffs: dict[str, float] = {}
    all_close = True
    for k in a:
        ta, tb = a[k], b[k]
        if ta.dtype != tb.dtype:
            return {"all_close": False, "reason": f"dtype mismatch on {k}"}
        if ta.shape != tb.shape:
            return {"all_close": False, "reason": f"shape mismatch on {k}"}
        max_diff = (ta - tb).abs().max().item() if ta.numel() else 0.0
        diffs[k] = max_diff
        if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
            all_close = False
    return {"all_close": all_close, "diffs": diffs}


def test_scripts_train_matches_legacy(tmp_path, monkeypatch):
    """Verifica que scripts/train.py y src.training.train.train() producen
    modelos equivalentes cuando entrenan sobre los mismos datos con misma
    seed. Tolerancia atol=1e-4: detecta drift semántico, tolera no-determinismo
    de orden entre dos paths de Lightning.
    """
    mini_combined, mini_emb, mini_splits = _build_mini_dataset(tmp_path)

    # --- LEGACY: monkeypatch paths globales antes de invocar ---
    from src.data import dataset as ds_module
    monkeypatch.setattr(ds_module, "EMB_PATH", mini_emb)
    monkeypatch.setattr(ds_module, "SPLITS_PATH", mini_splits)

    legacy_run_name = f"test_equiv_legacy_{uuid4().hex[:8]}"
    legacy_ckpt_dir = ROOT / "checkpoints" / legacy_run_name
    legacy_log_dir = ROOT / "wandb"  # legacy escribe ahi cuando local=False

    try:
        from src.training.train import train as legacy_train_fn
        legacy_result = legacy_train_fn({
            "local": True,
            "run_name": legacy_run_name,
            "max_epochs": 3,
            "seed": 42,
            "batch_size": 16,  # smaller batch para mini dataset
            "early_stopping_patience": 99,  # disable early stopping
        })
        legacy_ckpt = Path(legacy_result["best_ckpt"])
        assert legacy_ckpt.exists(), f"legacy ckpt missing: {legacy_ckpt}"

        # --- NEW: import y llamar train_model directamente ---
        # IMPORTANTE: NO usamos setup_determinism(42) del nuevo, solo
        # seed_everything (igual que legacy). Razon: setup_determinism agrega
        # torch.use_deterministic_algorithms + cudnn flags que legacy no tiene.
        # El test valida la LOGICA core de training (data loading, optimizer,
        # loss), no el stack extra de determinismo del nuevo (que es mejora
        # opcional, no break-equivalencia).
        from scripts.train import train_model
        import lightning as L

        L.seed_everything(42, workers=True)
        df = pd.read_parquet(mini_combined)
        species_to_idx = {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}

        new_out = tmp_path / "new_ckpts"
        new_reports = tmp_path / "new_reports"
        new_run_name = f"test_equiv_new_{uuid4().hex[:8]}"

        # Monkeypatch DEFAULTS para que coincidan con legacy_train_fn args
        from scripts import train as new_train_module
        original_defaults = new_train_module.DEFAULTS.copy()
        monkeypatch.setattr(
            new_train_module, "DEFAULTS",
            {**original_defaults, "batch_size": 16, "early_stopping_patience": 99},
        )

        new_result = train_model(
            df, species_to_idx, new_out, new_reports, new_run_name,
            max_epochs=3, seed=42,
        )
        new_ckpt = Path(new_result["best_ckpt"])
        assert new_ckpt.exists(), f"new ckpt missing: {new_ckpt}"

        # --- COMPARAR ---
        cmp = _state_dict_close(legacy_ckpt, new_ckpt, atol=1e-4, rtol=1e-3)
        if not cmp["all_close"]:
            # Falla loud con info de qué layers divergieron
            top_diffs = sorted(
                cmp["diffs"].items(), key=lambda kv: kv[1], reverse=True
            )[:5]
            msg = (
                "DRIFT detectado entre scripts/train.py y src.training.train.\n"
                f"Top 5 layers con max_diff:\n"
            ) + "\n".join(f"  {k}: {v:.6e}" for k, v in top_diffs)
            pytest.fail(msg)

    finally:
        # Cleanup: legacy escribió a ROOT/checkpoints/<run_name>. Borrar.
        if legacy_ckpt_dir.exists():
            shutil.rmtree(legacy_ckpt_dir, ignore_errors=True)
        # CSVLogger del legacy crea logs/<run_name>
        legacy_logs = ROOT / "logs" / legacy_run_name
        if legacy_logs.exists():
            shutil.rmtree(legacy_logs, ignore_errors=True)
