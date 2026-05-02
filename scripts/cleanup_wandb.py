"""CLI: limpieza de runs viejos de Weights & Biases.

Borra del proyecto W&B aquellos runs que cumplan TODOS los criterios:
- estado ∈ {"crashed", "failed"} o sin tag,
- antigüedad > N días,
- no tienen artifacts referenciados por otros runs.

Pensado para correr semanalmente desde GitHub Actions
(ver ``.github/workflows/cleanup_wandb.yml``).

Uso:
    python scripts/cleanup_wandb.py --days 30 --dry-run

TODO: implementar usando ``wandb.Api()``.
"""
