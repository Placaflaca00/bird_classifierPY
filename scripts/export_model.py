"""CLI: exporta un checkpoint Lightning a ONNX y verifica numéricamente.

Uso:
    python scripts/export_model.py
    python scripts/export_model.py --ckpt path/al/best.ckpt --out models/classifier.onnx
    python scripts/export_model.py --opset 17 --tol 1e-5

El export es para servir desde AWS Lambda con `onnxruntime`. Genera además un
sidecar JSON con metadata útil para inferencia (idx_to_species, embedding_dim,
checkpoint_source, métricas de validación numérica).
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

DEFAULT_CKPT_GLOB = "checkpoints/wa-drop3-v1/best-*.ckpt"
DEFAULT_OUT = ROOT / "models" / "classifier.onnx"
EMB_PARQUET = ROOT / "data" / "processed" / "embeddings.parquet"


def species_mapping_from(emb_path: Path) -> dict[str, int]:
    import pandas as pd
    df = pd.read_parquet(emb_path, columns=["species"])
    return {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help=f"Default: glob {DEFAULT_CKPT_GLOB}")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Path al archivo .onnx de salida")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch del input para verificación numérica.")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Tolerancia atol para np.allclose.")
    parser.add_argument("--dynamic-batch", action="store_true", default=True,
                        help="Eje 0 dinámico (batch variable en runtime).")
    args = parser.parse_args()

    if args.ckpt is None:
        cands = sorted(glob(str(ROOT / DEFAULT_CKPT_GLOB)))
        if not cands:
            print(f"No checkpoint en {DEFAULT_CKPT_GLOB}", file=sys.stderr)
            return 2
        ckpt_path = cands[-1]
    else:
        ckpt_path = args.ckpt
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output:     {out_path}")

    species_to_idx = species_mapping_from(EMB_PARQUET)
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    num_classes = len(species_to_idx)
    print(f"num_classes: {num_classes}")

    model = BirdClassifier.load_from_checkpoint(ckpt_path, strict=False, map_location="cpu")
    model.eval()
    out_features = model.net[-1].out_features
    if out_features != num_classes:
        print(f"! ckpt num_classes ({out_features}) != mapping ({num_classes})", file=sys.stderr)
        return 3

    embedding_dim = model.net[0].in_features  # 1024 esperado
    print(f"embedding_dim: {embedding_dim}")

    # Input dummy del tamaño correcto
    dummy = torch.randn(args.batch_size, embedding_dim, dtype=torch.float32)

    dynamic_axes = {"embedding": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None
    print(f"Exportando ONNX (opset={args.opset}, dynamic_batch={args.dynamic_batch})...")
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["embedding"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )

    # El exporter por default puede dejar pesos en archivo .onnx.data separado.
    # Para deploy en Lambda preferimos single-file: merge los external data
    # y reescribimos inline.
    import onnx
    external_data = out_path.with_suffix(".onnx.data")
    if external_data.exists():
        print(f"Mergeando external data inline ({external_data.stat().st_size/1024:.1f} KB)...")
        m = onnx.load(str(out_path))
        onnx.save(m, str(out_path), save_as_external_data=False)
        external_data.unlink()
    size_kb = out_path.stat().st_size / 1024
    print(f"OK: {out_path} ({size_kb:.1f} KB)")

    # Verificación numérica
    print("\nVerificación numérica (PyTorch vs ONNX Runtime)...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    test_inputs = [
        rng.standard_normal((args.batch_size, embedding_dim)).astype(np.float32),
        rng.standard_normal((1, embedding_dim)).astype(np.float32),  # single-sample
        rng.standard_normal((32, embedding_dim)).astype(np.float32),  # batch grande
    ]
    max_diff = 0.0
    for x_np in test_inputs:
        with torch.no_grad():
            y_torch = model(torch.from_numpy(x_np)).numpy()
        y_onnx = sess.run(["logits"], {"embedding": x_np})[0]
        diff = float(np.abs(y_torch - y_onnx).max())
        max_diff = max(max_diff, diff)
        ok = np.allclose(y_torch, y_onnx, atol=args.tol)
        print(f"  batch_size={x_np.shape[0]:>3d}  max_diff={diff:.2e}  allclose(atol={args.tol:.0e})={ok}")
    print(f"Max diff global: {max_diff:.2e}")

    if max_diff > args.tol:
        print(f"! Verificación FALLÓ (max_diff > tol).", file=sys.stderr)
        return 4

    # Sidecar JSON con metadata de inferencia
    sidecar = out_path.with_suffix(".json")
    metadata = {
        "checkpoint_source": str(ckpt_path),
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "opset": args.opset,
        "dynamic_batch": args.dynamic_batch,
        "max_numerical_diff": max_diff,
        "input": {"name": "embedding", "shape": ["batch", embedding_dim], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch", num_classes], "dtype": "float32"},
        "idx_to_species": idx_to_species,
        "inference_notes": (
            "Apply softmax over axis=1 to convert logits to class probabilities. "
            "argmax(probs) gives predicted class index; map to species via idx_to_species. "
            "Embedding (1024-d) viene de BirdNET V2.4 mean-pool sobre ventanas de 3s @ 48kHz."
        ),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSidecar metadata: {sidecar} ({sidecar.stat().st_size} bytes)")
    print(f"\nListo. Modelo ONNX listo para servir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
