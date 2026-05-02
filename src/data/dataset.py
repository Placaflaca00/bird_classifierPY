"""PyTorch Dataset para embeddings BirdNET pre-computados.

Define ``BirdEmbeddingsDataset(torch.utils.data.Dataset)`` que:
- Lee ``embeddings.npy`` (shape ``(N, 320)``, float32) y ``labels.npy`` (shape ``(N,)``, int64).
- Devuelve tuplas ``(embedding: torch.Tensor[320], label: torch.Tensor[scalar])``.
- Soporta opcionalmente un ``transform`` (mixup u otras augmentations en el espacio de embeddings).

Variantes a contemplar:
- ``train`` vs ``val`` vs ``test`` (sin augmentations en val/test).
- Carga lazy desde disco vs in-memory (para Colab con datasets chicos, in-memory).

TODO: implementar la clase y un helper ``build_dataloaders(cfg)``.
"""
