"""Compara candidato (output de ``scripts/train.py``) vs modelo en produccion
(``wa-drop3-v1``) sobre test_hard inmutable (politica D20). Decide si el
candidato merece ser promovido al loop de deploy.

Output: JSON consumido por ``scripts/promote.py`` (que solo deploya si
``should_promote=true``).

Reglas de decision (acordadas)
==============================
Para que ``should_promote=true`` deben cumplirse las TRES:

1. ``delta_acc_pp >= 1.0`` sobre test_hard.
2. ``mcnemar.p_value < 0.05`` (test estadistico de pareados).
3. ``delta_macro_f1_pp > -0.5`` (guarda anti-regresion sigilosa: previene
   caso donde accuracy gana por una clase grande pero el macro F1
   colapsa por regresion en clases minoritarias).

Cualquier condicion que falle -> ``should_promote=false``, con ``reason``
explicito en el JSON output.

Mid-p McNemar (vs Edwards continuity correction + exact binomial fallback)
==========================================================================
Usamos **mid-p McNemar** (Fagerland, Lydersen & Laake 2013, "The McNemar
test for binary matched-pairs data: mid-p and asymptotic are better than
exact conditional", BMC Medical Research Methodology), citada como
default moderno para tablas 2x2 pareadas con muestras pequenas-medianas
(n discordantes < 100).

Justificacion: Edwards continuity correction sobre-conservadora y exact
binomial tambien. Mid-p tiene mejor power, mantiene level nominal, y en
los 9595 escenarios simulados del paper nunca violo alfa=0.05. La
implementacion son ~5 lineas:

    k = min(b, c)
    n = b + c
    p_exact_one_sided = scipy.stats.binom.cdf(k, n, 0.5)
    p_mid_one_sided = p_exact_one_sided - 0.5 * scipy.stats.binom.pmf(k, n, 0.5)
    p_value = min(1.0, 2 * p_mid_one_sided)

Edge case b+c==0 (modelos producen predicciones identicas): McNemar
undefined. Comportamiento defensivo: ``p_value=1.0``, ``should_promote=false``,
``reason="modelos producen predicciones identicas sobre test_hard"``.

Bootstrap CI para deltas
========================
Punto-estimate de delta_acc sin CI es enganoso. 1000 resamples con
reemplazo sobre los pares pareados (pred_candidato, pred_prod, true) ->
percentile 95% para delta_acc_pp y delta_macro_f1_pp. Si el CI cruza
cero, **segunda senal independiente** de McNemar: aunque McNemar diga
p<0.05, si el CI cruza cero, hay duda sobre direccion real del efecto.

CONTRATO CON promote.py (CRITICO — train-serving skew)
======================================================
``evaluate.py`` decide con el ``.ckpt`` PyTorch (comparacion pareada limpia
sin diff numerica). Pero el deploy es ONNX. La conversion .ckpt->ONNX
tiene ``max_numerical_diff ~ 1.9e-06`` (sidecar models/classifier.json
de wa-drop3-v1). Sobre 208 muestras + argmax, ese diff puede flipear
0-2 predicciones, suficiente para invalidar McNemar en casos borderline.

**promote.py DEBE**:
1. Convertir el .ckpt candidato a ONNX.
2. Re-evaluar el ONNX contra test_hard.
3. Abortar el deploy si ``delta_acc(onnx_candidate, ckpt_candidate) > 0.5pp``.

Sin ese check, evaluate.py puede dar luz verde a un candidato cuya version
desplegada (ONNX) regresa silenciosamente vs su version evaluada (ckpt).

Future work documentado
=======================
Per-class F1 floor: ademas de macro_f1 floor (-0.5pp), considerar que
ninguna especie individual pierda >5pp de F1. Previene caso "macro F1
estable porque mejoras grandes en algunas clases compensan colapso en
2 clases". No implementado en MVP. Si se agrega, ir a la regla 4 de
``should_promote`` y reportar peores 3 clases en JSON.

Usage
=====
    python scripts/evaluate.py \\
        --candidate-ckpt checkpoints/retrain-<ts>-<sha>/best-*.ckpt \\
        --dataset data/processed/embeddings_retrain_<ts>.parquet

    # Con prod custom y thresholds tunneable
    python scripts/evaluate.py \\
        --candidate-ckpt ... \\
        --prod-ckpt checkpoints/wa-drop3-v1/best-epoch=12-val_macro_f1=0.909.ckpt \\
        --dataset ... \\
        --threshold-pp 2.0 --threshold-pvalue 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

DEFAULT_PROD_GLOB = str(ROOT / "checkpoints" / "wa-drop3-v1" / "best-*.ckpt")

# Defaults de threshold (alineados con plan Fase 6)
DEFAULT_THRESHOLD_DELTA_PP = 1.0
DEFAULT_THRESHOLD_PVALUE = 0.05
DEFAULT_MACRO_F1_FLOOR_PP = -0.5

# Bootstrap CI
N_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidate-ckpt", type=Path, required=True,
                   help="Path al .ckpt del candidato (output de train.py).")
    p.add_argument("--prod-ckpt", type=Path, default=None,
                   help="Path al .ckpt de prod. Default: wa-drop3-v1.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Parquet self-contained con fold=test_hard (D20).")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <candidate-dir>/../reports/<run>/evaluation_vs_prod.json")
    p.add_argument("--threshold-pp", type=float, default=DEFAULT_THRESHOLD_DELTA_PP)
    p.add_argument("--threshold-pvalue", type=float, default=DEFAULT_THRESHOLD_PVALUE)
    p.add_argument("--macro-f1-floor-pp", type=float, default=DEFAULT_MACRO_F1_FLOOR_PP,
                   help="Si delta_macro_f1_pp < floor, NO promover. Default -0.5.")
    p.add_argument("--apply", action="store_true",
                   help="Default: dry-run (carga modelos, reporta plan, no escribe JSON).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

def resolve_prod_ckpt(arg: Path | None) -> Path:
    if arg is not None:
        if not arg.exists():
            raise FileNotFoundError(f"prod-ckpt no existe: {arg}")
        return arg
    matches = sorted(glob(DEFAULT_PROD_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No encuentro prod ckpt default: {DEFAULT_PROD_GLOB}. "
            "Pasa --prod-ckpt explicito."
        )
    return Path(matches[-1])  # mas reciente alfabeticamente


def load_classifier(ckpt: Path) -> BirdClassifier:
    """Carga BirdClassifier desde .ckpt. strict=False tolera checkpoints
    viejos con buffers extra (class_weights persistente, deprecated).
    """
    model = BirdClassifier.load_from_checkpoint(str(ckpt), strict=False, map_location="cpu")
    model.eval()
    return model


def run_inference(
    model: BirdClassifier,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Devuelve top-1 predictions (int64 array, shape=(N,))."""
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        logits = model(x)
        return logits.argmax(dim=1).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy + macro F1 sobre predictions pareadas con labels."""
    if len(labels) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "n": 0}
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"acc": acc, "macro_f1": f1, "n": int(len(labels))}


# ---------------------------------------------------------------------------
# Mid-p McNemar (Fagerland et al. 2013)
# ---------------------------------------------------------------------------

def mid_p_mcnemar(
    preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    """McNemar mid-p para 2 modelos sobre el mismo test set.

    Convencion de la tabla 2x2:
        a (both correct):                   correct_a & correct_b
        b (A correct, B wrong):             correct_a & ~correct_b
        c (A wrong, B correct):             ~correct_a & correct_b
        d (both wrong):                     ~correct_a & ~correct_b

    Para "candidato vs prod" con A=candidato, B=prod:
        b = candidato acerto y prod fallo -> mejoras del candidato
        c = prod acerto y candidato fallo -> regresiones

    El test es two-sided: H0 = b == c. p_value chico = diferencia
    significativa en alguna direccion (no dice cual).
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())

    if b + c == 0:
        return {
            "b_cand_correct_prod_wrong": b,
            "c_prod_correct_cand_wrong": c,
            "method": "undefined (no discordant pairs)",
            "p_value": 1.0,
            "mid_p_one_sided": None,
        }

    k = min(b, c)
    n = b + c
    p_exact_one_sided = scipy.stats.binom.cdf(k, n, 0.5)
    p_mid_one_sided = p_exact_one_sided - 0.5 * scipy.stats.binom.pmf(k, n, 0.5)
    p_value = float(min(1.0, 2 * p_mid_one_sided))

    return {
        "b_cand_correct_prod_wrong": b,
        "c_prod_correct_cand_wrong": c,
        "method": "mid_p_mcnemar (Fagerland et al. 2013)",
        "p_value": p_value,
        "mid_p_one_sided": float(p_mid_one_sided),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI para deltas (segunda señal independiente)
# ---------------------------------------------------------------------------

def bootstrap_delta_ci(
    preds_cand: np.ndarray,
    preds_prod: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """1000 resamples con reemplazo sobre los pares (cand, prod, true).
    Devuelve CI 95% (percentile method) para delta_acc_pp y delta_macro_f1_pp.
    Si el CI cruza cero, ``crosses_zero=true`` (segunda senal vs McNemar).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n == 0:
        return {
            "delta_acc_pp": {"ci_low": 0.0, "ci_high": 0.0, "crosses_zero": True},
            "delta_macro_f1_pp": {"ci_low": 0.0, "ci_high": 0.0, "crosses_zero": True},
        }

    deltas_acc = np.empty(n_resamples)
    deltas_f1 = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s_cand, s_prod, s_true = preds_cand[idx], preds_prod[idx], labels[idx]
        deltas_acc[i] = ((s_cand == s_true).mean() - (s_prod == s_true).mean()) * 100
        f1_c = f1_score(s_true, s_cand, average="macro", zero_division=0)
        f1_p = f1_score(s_true, s_prod, average="macro", zero_division=0)
        deltas_f1[i] = (f1_c - f1_p) * 100

    def ci(arr: np.ndarray) -> dict[str, float]:
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        return {
            "ci_low": low,
            "ci_high": high,
            "crosses_zero": bool(low < 0.0 < high),
        }

    return {
        "delta_acc_pp": ci(deltas_acc),
        "delta_macro_f1_pp": ci(deltas_f1),
    }


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def decide_promotion(
    delta_acc_pp: float,
    delta_macro_f1_pp: float,
    p_value: float,
    threshold_pp: float,
    threshold_pvalue: float,
    macro_f1_floor_pp: float,
) -> tuple[bool, str]:
    """Aplica las 3 condiciones. Devuelve (should_promote, reason)."""
    reasons: list[str] = []
    if delta_acc_pp < threshold_pp:
        reasons.append(
            f"delta_acc_pp ({delta_acc_pp:+.2f}) < threshold ({threshold_pp:+.2f})"
        )
    if p_value >= threshold_pvalue:
        reasons.append(
            f"p_value ({p_value:.4f}) >= threshold ({threshold_pvalue:.2f})"
        )
    if delta_macro_f1_pp < macro_f1_floor_pp:
        reasons.append(
            f"delta_macro_f1_pp ({delta_macro_f1_pp:+.2f}) < floor ({macro_f1_floor_pp:+.2f}) — "
            "regresion sigilosa en clases minoritarias"
        )

    if not reasons:
        return True, (
            f"OK: delta_acc {delta_acc_pp:+.2f}pp >= {threshold_pp:+.2f}, "
            f"p {p_value:.4f} < {threshold_pvalue}, "
            f"delta_macro_f1 {delta_macro_f1_pp:+.2f}pp > {macro_f1_floor_pp:+.2f}"
        )
    return False, " AND ".join(reasons)


# ---------------------------------------------------------------------------
# Dataset loading (parquet self-contained)
# ---------------------------------------------------------------------------

def load_fold(df: pd.DataFrame, fold: str, species_to_idx: dict[str, int]
              ) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["fold"] == fold]
    if fold != "train" and "is_aug" in sub.columns:
        sub = sub[~sub["is_aug"]]
    sub = sub.reset_index(drop=True)
    if len(sub) == 0:
        return np.empty((0, 1024), dtype=np.float32), np.empty((0,), dtype=np.int64)
    emb = np.stack(sub["embedding"].to_numpy()).astype(np.float32)
    labels = sub["species"].map(species_to_idx).to_numpy().astype(np.int64)
    return emb, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Resolver paths
    candidate_ckpt = args.candidate_ckpt
    if not candidate_ckpt.exists():
        # tolerar glob (best-*.ckpt)
        matches = sorted(glob(str(candidate_ckpt)))
        if not matches:
            print(f"! Candidate ckpt no existe: {candidate_ckpt}", file=sys.stderr)
            return 1
        candidate_ckpt = Path(matches[-1])
    prod_ckpt = resolve_prod_ckpt(args.prod_ckpt)
    dataset = args.dataset
    if not dataset.exists():
        print(f"! Dataset no existe: {dataset}", file=sys.stderr)
        return 1

    # Output default
    if args.output:
        out_path = args.output
    else:
        # candidate_ckpt es checkpoints/<run>/best-*.ckpt
        run_name = candidate_ckpt.parent.name
        out_path = ROOT / "reports" / run_name / "evaluation_vs_prod.json"

    print("=== Evaluate (Fase 6.3) ===")
    print(f"  candidate:  {candidate_ckpt}")
    print(f"  prod:       {prod_ckpt}")
    print(f"  dataset:    {dataset}")
    print(f"  output:     {out_path}")
    print(f"  thresholds: delta>={args.threshold_pp}pp AND p<{args.threshold_pvalue} "
          f"AND delta_macro_f1>{args.macro_f1_floor_pp}pp")
    print()

    # Cargar dataset y construir species->idx (alfabetico, consistente con train.py)
    print("Cargando dataset...")
    df = pd.read_parquet(dataset)
    species_to_idx = {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}
    n_classes = len(species_to_idx)
    print(f"  filas: {len(df)}  clases: {n_classes}")

    folds_data = {fold: load_fold(df, fold, species_to_idx)
                  for fold in ("val", "test_clean", "test_hard")}
    for fold, (emb, labels) in folds_data.items():
        print(f"  {fold:<12s} {len(labels)} muestras")
    print()

    if not args.apply:
        print("DRY-RUN: cargo modelos pero no escribo output JSON.")
        print("Para ejecutar real: agregar --apply")
        print()

    # Cargar modelos
    print("Cargando candidato...")
    cand_model = load_classifier(candidate_ckpt)
    print("Cargando prod...")
    prod_model = load_classifier(prod_ckpt)
    print()

    # Inferencia + metricas por fold
    metrics: dict[str, dict[str, Any]] = {"candidate": {}, "prod": {}}
    preds: dict[str, dict[str, np.ndarray]] = {"candidate": {}, "prod": {}}
    for fold, (emb, labels) in folds_data.items():
        if len(labels) == 0:
            continue
        preds["candidate"][fold] = run_inference(cand_model, emb)
        preds["prod"][fold] = run_inference(prod_model, emb)
        metrics["candidate"][fold] = compute_metrics(preds["candidate"][fold], labels)
        metrics["prod"][fold] = compute_metrics(preds["prod"][fold], labels)

    # Reporte por fold
    print("Metricas por fold:")
    print(f"{'fold':<12s} {'cand_acc':>9s} {'prod_acc':>9s} {'delta_pp':>9s}  "
          f"{'cand_f1':>9s} {'prod_f1':>9s} {'delta_pp':>9s}")
    for fold in ("val", "test_clean", "test_hard"):
        if fold not in metrics["candidate"]:
            continue
        mc, mp = metrics["candidate"][fold], metrics["prod"][fold]
        d_acc = (mc["acc"] - mp["acc"]) * 100
        d_f1 = (mc["macro_f1"] - mp["macro_f1"]) * 100
        print(f"{fold:<12s} {mc['acc']:>9.4f} {mp['acc']:>9.4f} {d_acc:>+9.2f}  "
              f"{mc['macro_f1']:>9.4f} {mp['macro_f1']:>9.4f} {d_f1:>+9.2f}")
    print()

    # === DECISION sobre test_hard ===
    hard_labels = folds_data["test_hard"][1]
    hard_cand = preds["candidate"]["test_hard"]
    hard_prod = preds["prod"]["test_hard"]

    mc_hard = metrics["candidate"]["test_hard"]
    mp_hard = metrics["prod"]["test_hard"]
    delta_acc_pp = (mc_hard["acc"] - mp_hard["acc"]) * 100
    delta_macro_f1_pp = (mc_hard["macro_f1"] - mp_hard["macro_f1"]) * 100

    print("McNemar mid-p sobre test_hard (Fagerland et al. 2013)...")
    mcnemar = mid_p_mcnemar(hard_cand, hard_prod, hard_labels)
    print(f"  b (cand correct, prod wrong):  {mcnemar['b_cand_correct_prod_wrong']}")
    print(f"  c (prod correct, cand wrong):  {mcnemar['c_prod_correct_cand_wrong']}")
    print(f"  method: {mcnemar['method']}")
    print(f"  p_value: {mcnemar['p_value']:.4f}")
    print()

    print(f"Bootstrap CI ({N_BOOTSTRAP_RESAMPLES} resamples)...")
    boot = bootstrap_delta_ci(hard_cand, hard_prod, hard_labels)
    da, df1 = boot["delta_acc_pp"], boot["delta_macro_f1_pp"]
    print(f"  delta_acc_pp:      [{da['ci_low']:+.2f}, {da['ci_high']:+.2f}]  "
          f"crosses_zero={da['crosses_zero']}")
    print(f"  delta_macro_f1_pp: [{df1['ci_low']:+.2f}, {df1['ci_high']:+.2f}]  "
          f"crosses_zero={df1['crosses_zero']}")
    print()

    should_promote, reason = decide_promotion(
        delta_acc_pp, delta_macro_f1_pp, mcnemar["p_value"],
        args.threshold_pp, args.threshold_pvalue, args.macro_f1_floor_pp,
    )
    print(f"DECISION: should_promote = {should_promote}")
    print(f"  reason: {reason}")
    print()

    # Construir JSON output
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "candidate_ckpt": str(candidate_ckpt),
        "prod_ckpt": str(prod_ckpt),
        "dataset": str(dataset),
        "test_hard_size": int(len(hard_labels)),
        "n_classes": n_classes,
        "candidate": metrics["candidate"],
        "prod": metrics["prod"],
        "delta_test_hard": {
            "acc_pp": float(delta_acc_pp),
            "macro_f1_pp": float(delta_macro_f1_pp),
        },
        "mcnemar": mcnemar,
        "bootstrap_ci_95": boot,
        "threshold": {
            "delta_pp": args.threshold_pp,
            "p_value": args.threshold_pvalue,
            "macro_f1_floor_pp": args.macro_f1_floor_pp,
        },
        "should_promote": should_promote,
        "reason": reason,
    }

    if not args.apply:
        print("DRY-RUN: NO escribo JSON. Re-correr con --apply.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Escrito: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
