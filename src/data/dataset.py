"""Dataset PyTorch para embeddings BirdNET pre-computados.

Carga ``data/processed/embeddings.parquet`` + ``data/processed/splits.parquet``,
los mergea y filtra por fold (``train``, ``val``, ``test_clean``, ``test_hard``).

Devuelve tuplas ``(embedding[1024], label_idx)``.

El mapping ``species -> class_idx`` es determinístico (alfabético sobre el set
completo de especies en embeddings.parquet), así todos los splits usan la
misma codificación de clases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = ROOT / "data" / "processed" / "embeddings.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.parquet"

VALID_FOLDS = {"train", "val", "test_clean", "test_hard"}


def load_species_mapping() -> dict[str, int]:
    """species -> class_idx. Alfabético, determinístico."""
    df = pd.read_parquet(EMB_PATH, columns=["species"])
    species = sorted(df["species"].unique())
    return {sp: i for i, sp in enumerate(species)}


class BirdEmbeddingsDataset(Dataset):
    """Carga embeddings de un fold específico en memoria como tensores."""

    def __init__(self, fold: str, species_to_idx: dict[str, int] | None = None) -> None:
        if fold not in VALID_FOLDS:
            raise ValueError(f"fold inválido: {fold!r}. Debe ser uno de {VALID_FOLDS}")

        emb = pd.read_parquet(EMB_PATH)
        splits = pd.read_parquet(SPLITS_PATH)
        df = emb.merge(splits, on="filepath", how="inner")
        df = df[df["fold"] == fold].reset_index(drop=True)

        self.species_to_idx = species_to_idx or load_species_mapping()
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        self.fold = fold
        self.filepaths: list[str] = df["filepath"].tolist()

        if len(df) == 0:
            self.embeddings = torch.empty((0, 1024), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.long)
            return

        emb_arr = np.stack(df["embedding"].to_numpy()).astype(np.float32)
        self.embeddings = torch.from_numpy(emb_arr)
        self.labels = torch.tensor(
            df["species"].map(self.species_to_idx).to_numpy(),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def build_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Crea DataLoaders para los 4 folds. Usa el mismo species->idx en todos."""
    species_to_idx = load_species_mapping()
    loaders: dict[str, DataLoader] = {}
    for fold in ("train", "val", "test_clean", "test_hard"):
        ds = BirdEmbeddingsDataset(fold, species_to_idx)
        loaders[fold] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(fold == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders
