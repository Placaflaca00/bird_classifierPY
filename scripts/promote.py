"""Promueve un candidato de Fase 6 a produccion: ONNX export -> skew check ->
Docker build -> ECR push -> Lambda update -> smoke test -> local sync.

4 etapas con risk creciente, gateadas por flags explicitos:

    --export-only        Etapa 1 (ONNX + skew check, sin AWS).
    --build              Etapas 1+2 (incluye docker build local).
    --apply-deploy       Etapas 1+2+3+4 (FULL deploy, toca prod).
    --rollback-to <tag>  Re-deploya tag ECR previo (operacional aparte).

Contrato consumido de evaluate.py
=================================
Lee ``evaluation_vs_prod.json`` y verifica ``should_promote=true``. Si es
false, exit 0 sin tocar nada. Bypass solo con ``--bypass-promotion-gate``
(naming explicito anti-accidente, y aborta si ``BIRD_CLASSIFIER_ENV=prod``).

Contrato hacia abajo (train-serving skew)
=========================================
``evaluate.py`` decide con el ``.ckpt`` PyTorch (comparacion pareada limpia).
``promote.py`` DEBE verificar que el ONNX deployado replica las decisiones
del ckpt antes de deployar:
  1. Convertir .ckpt -> ONNX.
  2. Re-evaluar ONNX contra test_hard del parquet self-contained.
  3. Abortar si:
     - ``delta_acc(onnx, ckpt) > 0.5pp``
     - ``disagreement_rate(onnx_preds, ckpt_preds) > 1%``
       (el delta_acc agregado puede enmascarar flips locales que se
       compensan entre clases; disagreement_rate los detecta).

Decision: $LATEST + update-function-code, NO aliases
====================================================
El sistema actual usa $LATEST con ``update-function-code``, sin Lambda
aliases. Verificado 2026-05-25 contra AWS real: ``list-aliases`` devuelve
``[]`` y el API Gateway integration apunta directo al function ARN sin
sufijo de alias.

Considered and rejected: migrating to PROD/CANARY aliases with
``update-alias`` for instant rollback. Rationale:
  - Volume: 1-2 users/day with active warming (EventBridge rate(7 min))
    makes millisecond rollback irrelevant.
  - Audience: zoos/NGOs PY without SLA requirements (donation, not
    commercial product).
  - Cost: alias migration requires API Gateway IntegrationUri change +
    resource policy update + version lifecycle management — ~2-3hrs of
    prod-touching work.
  - Risk: not the technical risk of update-function-code itself (which
    AWS handles gracefully — old container serves until new is Active;
    ~3.8% inconsistency window only with >=50 concurrent reqs per AWS
    docs) but the complexity tax for a single admin.

Decision revisable if SLA requirements emerge or if we hit the
inconsistency window in practice (unlikely at current volume).

Rollback strategy en este modelo
================================
Auto-rollback en falla del smoke test: capturar ``digest_previo`` ANTES
del ``update-function-code``, hacer el cambio, smoke test, si falla
hacer ``update-function-code`` de vuelta al digest previo.

Manual rollback: ``--rollback-to <ecr_tag>`` re-deploya un tag ECR
existente (e.g., ``--rollback-to fase5b-retention``).

ONNX export disciplinado (defaults)
===================================
  - model.eval() antes del export (BatchNorm/Dropout off, critico).
  - do_constant_folding=True (optimizacion segura).
  - onnx.checker.check_model() post-export para validar estructura.
  - opset=17 (matchea models/classifier.json sidecar de wa-drop3-v1).

Smoke test con baseline persistido
==================================
``scripts/smoke_baseline.json`` tiene 5 audios fijos + predicciones
esperadas (top1, confidence) del modelo en produccion al momento de
generarlo. Smoke pasa si TODAS:
  - top1 == esperado en >=80% de samples (4/5).
  - confidence dentro de +-15pp del baseline.
  - latencia p95 < 2s sobre warm Lambda.

Si baseline no existe: ``--generate-baseline`` lo regenera con el
modelo en produccion ACTUAL (no el candidato — ese es el cuyo
comportamiento queremos preservar dentro de tolerancia).

Usage
=====
    # Antes del primer promote, generar baseline:
    python scripts/promote.py --generate-baseline

    # Workflow normal (consume evaluation JSON):
    python scripts/promote.py --evaluation reports/<run>/evaluation_vs_prod.json --export-only
    python scripts/promote.py --evaluation reports/<run>/evaluation_vs_prod.json --build
    python scripts/promote.py --evaluation reports/<run>/evaluation_vs_prod.json --apply-deploy

    # Testing E2E con candidato actual que falla should_promote:
    python scripts/promote.py --evaluation ... --apply-deploy --bypass-promotion-gate

    # Rollback explicito:
    python scripts/promote.py --rollback-to fase5b-retention
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any

# Windows: torch.onnx.export emite emojis (✅/❌) a stdout que cp1252 default
# no puede codificar (UnicodeEncodeError). Forzar utf-8 ANTES de importar torch.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import BirdClassifier  # noqa: E402

# AWS resources
ECR_REPOSITORY = "bird-classifier-py"
LAMBDA_FUNCTION_NAME = "bird-classifier-py-inference"
AWS_REGION = "us-east-1"
AWS_ACCOUNT_ID = "863518416901"
ECR_REPO_FULL = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}"

# Local paths
MODELS_DIR = ROOT / "models"
SMOKE_BASELINE_PATH = ROOT / "scripts" / "smoke_baseline.json"

# Thresholds (skew check + smoke test)
DISAGREEMENT_THRESHOLD = 0.01  # 1%
SKEW_DELTA_PP_MAX = 0.5
SMOKE_TOP1_THRESHOLD = 0.80  # 4/5
SMOKE_CONFIDENCE_TOLERANCE_PP = 15.0
SMOKE_LATENCY_P95_MAX_S = 2.0
LAMBDA_ACTIVE_TIMEOUT_S = 300
SMOKE_LAMBDA_WARMUP_INVOKES = 2

# Audios del smoke baseline (rutas relativas a data/raw/)
SMOKE_BASELINE_AUDIOS = [
    "chauna_torquata",
    "ramphastos_toco",
    "columba_livia",
    "jabiru_mycteria",
    "rhea_americana",
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--evaluation", type=Path, default=None,
                   help="Path a evaluation_vs_prod.json. Requerido salvo "
                        "--rollback-to o --generate-baseline.")
    p.add_argument("--candidate-ckpt", type=Path, default=None,
                   help="Path al .ckpt candidato. Default: leido de --evaluation.")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Parquet self-contained para skew check. "
                        "Default: leido de --evaluation.")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--export-only", action="store_true",
                     help="Solo Etapa 1: ONNX + skew check. Default si sin flags.")
    grp.add_argument("--build", action="store_true",
                     help="Etapas 1+2: incluye docker build local. Sin push.")
    grp.add_argument("--apply-deploy", action="store_true",
                     help="Etapas 1+2+3+4: FULL deploy. Toca prod.")
    grp.add_argument("--rollback-to", type=str, default=None, metavar="TAG",
                     help="Re-deploya un tag ECR existente. Bypass de evaluation.")
    grp.add_argument("--generate-baseline", action="store_true",
                     help="Regenera scripts/smoke_baseline.json con modelo actual.")

    p.add_argument("--bypass-promotion-gate", action="store_true",
                   help="TESTING ONLY: ignora should_promote del evaluation. "
                        "Aborta si BIRD_CLASSIFIER_ENV=prod.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env guards
# ---------------------------------------------------------------------------

def check_env_guard(bypass: bool) -> None:
    """Aborta el bypass si BIRD_CLASSIFIER_ENV=prod. Defensa en profundidad."""
    if bypass and os.environ.get("BIRD_CLASSIFIER_ENV") == "prod":
        raise RuntimeError(
            "BIRD_CLASSIFIER_ENV=prod detectado y se pidio --bypass-promotion-gate. "
            "ABORTANDO. Si esto es intencional, unset BIRD_CLASSIFIER_ENV primero."
        )


def _boto3_clients():
    import boto3
    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError(
            "boto3 no encontro credenciales AWS. Configurar via `aws configure`."
        )
    return (
        session.client("ecr", region_name=AWS_REGION),
        session.client("lambda", region_name=AWS_REGION),
    )


# ---------------------------------------------------------------------------
# ETAPA 1 — ONNX export + skew check
# ---------------------------------------------------------------------------

def stage1_export_onnx(
    candidate_ckpt: Path, output_path: Path, sidecar_path: Path,
    idx_to_species: dict[int, str],
) -> tuple[Path, Path]:
    """Export disciplinado: model.eval(), do_constant_folding, checker.
    El sidecar incluye ``idx_to_species`` que handler.py requiere para
    mapear logits.argmax -> nombre de especie en /predict.

    Returns (onnx_path, sidecar_path).
    """
    print(f"[Etapa 1.a] Cargando {candidate_ckpt.name}...")
    model = BirdClassifier.load_from_checkpoint(
        str(candidate_ckpt), strict=False, map_location="cpu",
    )
    model.eval()  # CRITICO: BatchNorm/Dropout off

    num_classes = model.hparams.num_classes
    embedding_dim = model.hparams.embedding_dim

    dummy_input = torch.randn(1, embedding_dim, dtype=torch.float32)

    print(f"[Etapa 1.a] Exportando ONNX -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.net,  # solo el nn.Sequential, no el LightningModule wrapper
        dummy_input,
        str(output_path),
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={
            "embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Re-save inline (single-file). El nuevo torch.onnx (dynamo=True, default)
    # exporta external data en classifier.onnx.data, pero el ONNX en produccion
    # de wa-drop3-v1 es single-file. Lambda + ONNX Runtime fallan si el .data
    # no existe (ValidateExternalDataPath). Forzar single-file para paridad
    # con prod y simplicidad de deploy.
    import onnx
    loaded = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(
        loaded, str(output_path),
        save_as_external_data=False,
    )
    # Limpiar el .data huerfano
    data_file = output_path.parent / (output_path.name + ".data")
    data_file.unlink(missing_ok=True)

    # Validacion estructural (sobre el modelo single-file final)
    print(f"[Etapa 1.a] onnx.checker.check_model...")
    onnx.checker.check_model(onnx.load(str(output_path)))

    # Sidecar JSON (mismo schema que models/classifier.json existente).
    # `idx_to_species` es CRITICO para handler.py (linea 245 lo usa para
    # mapear logits.argmax a nombre de especie en /predict).
    if len(idx_to_species) != num_classes:
        raise RuntimeError(
            f"idx_to_species len ({len(idx_to_species)}) != num_classes "
            f"({num_classes}). Sidecar quedaria inconsistente con el ONNX."
        )
    sidecar = {
        "checkpoint_source": str(candidate_ckpt),
        "num_classes": int(num_classes),
        "embedding_dim": int(embedding_dim),
        "opset": 17,
        "dynamic_batch": True,
        "input": {"name": "embedding", "shape": ["batch", embedding_dim], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch", num_classes], "dtype": "float32"},
        "idx_to_species": {str(i): sp for i, sp in idx_to_species.items()},
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[Etapa 1.a] OK: {output_path.name} + {sidecar_path.name}")
    return output_path, sidecar_path


def stage1_skew_check(
    candidate_ckpt: Path, candidate_onnx: Path, dataset_path: Path,
) -> dict[str, Any]:
    """Compara predicciones ckpt vs ONNX sobre test_hard. Abort si:
      - delta_acc > 0.5pp
      - disagreement_rate > 1%

    Returns dict con resultados para JSON output.
    """
    import onnxruntime as ort

    print(f"[Etapa 1.b] Skew check ckpt vs ONNX sobre test_hard...")
    df = pd.read_parquet(dataset_path)
    test_hard = df[df["fold"] == "test_hard"].reset_index(drop=True)
    if "is_aug" in test_hard.columns:
        test_hard = test_hard[~test_hard["is_aug"]].reset_index(drop=True)
    embeddings = np.stack(test_hard["embedding"].to_numpy()).astype(np.float32)
    species_to_idx = {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}
    labels = test_hard["species"].map(species_to_idx).to_numpy().astype(np.int64)

    # Predicciones ckpt
    model = BirdClassifier.load_from_checkpoint(
        str(candidate_ckpt), strict=False, map_location="cpu",
    )
    model.eval()
    with torch.no_grad():
        preds_ckpt = model(torch.from_numpy(embeddings)).argmax(dim=1).numpy()

    # Predicciones ONNX
    sess = ort.InferenceSession(str(candidate_onnx), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"embedding": embeddings})[0]
    preds_onnx = onnx_logits.argmax(axis=1)

    acc_ckpt = float((preds_ckpt == labels).mean())
    acc_onnx = float((preds_onnx == labels).mean())
    delta_acc_pp = (acc_onnx - acc_ckpt) * 100
    disagreement_rate = float((preds_ckpt != preds_onnx).mean())

    info = {
        "n_samples": int(len(labels)),
        "acc_ckpt": acc_ckpt,
        "acc_onnx": acc_onnx,
        "delta_acc_pp": delta_acc_pp,
        "disagreement_rate": disagreement_rate,
        "thresholds": {
            "delta_pp_max": SKEW_DELTA_PP_MAX,
            "disagreement_max": DISAGREEMENT_THRESHOLD,
        },
    }
    print(f"  acc_ckpt={acc_ckpt:.4f}  acc_onnx={acc_onnx:.4f}  "
          f"delta_pp={delta_acc_pp:+.4f}  disagreement={disagreement_rate:.4%}")

    if abs(delta_acc_pp) > SKEW_DELTA_PP_MAX:
        raise RuntimeError(
            f"SKEW: delta_acc_pp ({delta_acc_pp:+.4f}) > {SKEW_DELTA_PP_MAX}pp. "
            "ABORTANDO promote — ONNX diverge del ckpt en acc agregada."
        )
    if disagreement_rate > DISAGREEMENT_THRESHOLD:
        raise RuntimeError(
            f"SKEW: disagreement_rate ({disagreement_rate:.4%}) > "
            f"{DISAGREEMENT_THRESHOLD:.0%}. ABORTANDO promote — predicciones "
            "ckpt/ONNX divergen sample-a-sample (compensacion entre clases)."
        )
    print(f"[Etapa 1.b] OK skew check")
    return info


# ---------------------------------------------------------------------------
# ETAPA 2 — Docker build local
# ---------------------------------------------------------------------------

def stage2_docker_build(
    candidate_onnx: Path, candidate_sidecar: Path, run_name: str,
) -> str:
    """Override temporal de lambda/models/classifier.{onnx,json} con candidato,
    docker build con build context = lambda/, restaurar originales en finally.

    El Dockerfile (linea 1-2) asume build context = lambda/. Los modelos viven
    en DOS lugares en el repo: ``models/`` (raiz, source-of-truth canonical
    usado para tests locales) y ``lambda/models/`` (copia deployable usada por
    docker COPY). Este stage solo toca la copia deployable; la sync de la
    canonical se hace en update_local_models() post-smoke.
    """
    lambda_models_dir = ROOT / "lambda" / "models"
    canonical_onnx = lambda_models_dir / "classifier.onnx"
    canonical_json = lambda_models_dir / "classifier.json"
    backup_onnx = lambda_models_dir / ".classifier.onnx.bak_promote"
    backup_json = lambda_models_dir / ".classifier.json.bak_promote"

    print(f"[Etapa 2] Override temporal de lambda/models/classifier.{{onnx,json}}...")
    shutil.copy2(canonical_onnx, backup_onnx)
    shutil.copy2(canonical_json, backup_json)

    local_tag = f"{ECR_REPOSITORY}:{run_name}"
    try:
        shutil.copy2(candidate_onnx, canonical_onnx)
        shutil.copy2(candidate_sidecar, canonical_json)

        # Build context = lambda/ (matchea el Dockerfile)
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "--provenance=false", "--sbom=false",
            "-t", local_tag,
            "-f", "Dockerfile",
            "--load",
            ".",
        ]
        print(f"  cwd: {ROOT / 'lambda'}")
        print(f"  cmd: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(ROOT / "lambda"), capture_output=True,
        )
        # Stream stdout para que el usuario vea progreso post-mortem
        sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
        if result.returncode != 0:
            sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
            raise RuntimeError(f"docker build fallo con codigo {result.returncode}")
    finally:
        # Restaurar originales — el repo queda limpio
        shutil.copy2(backup_onnx, canonical_onnx)
        shutil.copy2(backup_json, canonical_json)
        backup_onnx.unlink(missing_ok=True)
        backup_json.unlink(missing_ok=True)

    print(f"[Etapa 2] OK: imagen local '{local_tag}'")
    return local_tag


# ---------------------------------------------------------------------------
# ETAPA 3 — ECR push + Lambda update
# ---------------------------------------------------------------------------

def stage3_ecr_login(ecr_client) -> None:
    print("[Etapa 3.a] ECR login...")
    auth = ecr_client.get_authorization_token()["authorizationData"][0]
    token = base64.b64decode(auth["authorizationToken"]).decode("utf-8")
    user, password = token.split(":", 1)
    proxy = auth["proxyEndpoint"]
    result = subprocess.run(
        ["docker", "login", "--username", user, "--password-stdin", proxy],
        input=password.encode("utf-8"), capture_output=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError("docker login a ECR fallo")
    print("[Etapa 3.a] OK")


def stage3_docker_push(local_tag: str, run_name: str) -> str:
    """Push a ECR. Returns image digest (sha256:...) post-push."""
    remote_tag = f"{ECR_REPO_FULL}:{run_name}"
    print(f"[Etapa 3.b] Re-tag + push: {local_tag} -> {remote_tag}")

    subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)
    result = subprocess.run(
        ["docker", "push", remote_tag], capture_output=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError("docker push fallo")

    # Extraer digest del output
    out = result.stdout.decode("utf-8", errors="replace")
    digest = None
    for line in out.splitlines():
        if "digest:" in line and "sha256:" in line:
            digest = line.split("digest:")[1].strip().split()[0]
            break
    if not digest:
        raise RuntimeError(f"No pude extraer digest del push output:\n{out}")
    print(f"[Etapa 3.b] OK digest={digest}")
    return digest


def stage3_lambda_update(
    lambda_client, new_digest: str
) -> tuple[str, str]:
    """Captura prev digest, hace update-function-code, espera Active.

    Returns (new_image_uri_with_digest, prev_image_uri_with_digest).
    El prev_image_uri es usado para auto-rollback si smoke falla.
    """
    print(f"[Etapa 3.c] Capturando digest previo + update-function-code...")
    fn_info = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
    prev_image_uri = fn_info["Code"]["ImageUri"]
    print(f"  prev:  {prev_image_uri}")

    new_image_uri = f"{ECR_REPO_FULL}@{new_digest}"
    print(f"  new:   {new_image_uri}")

    lambda_client.update_function_code(
        FunctionName=LAMBDA_FUNCTION_NAME,
        ImageUri=new_image_uri,
    )

    # Poll hasta LastUpdateStatus = Successful
    t0 = time.time()
    while True:
        info = lambda_client.get_function_configuration(FunctionName=LAMBDA_FUNCTION_NAME)
        status = info.get("LastUpdateStatus", "InProgress")
        if status == "Successful":
            break
        if status == "Failed":
            reason = info.get("LastUpdateStatusReason", "no reason")
            raise RuntimeError(f"Lambda update Failed: {reason}")
        if time.time() - t0 > LAMBDA_ACTIVE_TIMEOUT_S:
            raise RuntimeError(
                f"Lambda update no llego a Successful en {LAMBDA_ACTIVE_TIMEOUT_S}s"
            )
        time.sleep(5)
    elapsed = time.time() - t0
    print(f"[Etapa 3.c] OK Active en {elapsed:.0f}s")
    return new_image_uri, prev_image_uri


def rollback_lambda(lambda_client, prev_image_uri: str) -> None:
    """Re-deploya el digest previo. Misma logica de poll-Active."""
    print(f"[ROLLBACK] update-function-code -> {prev_image_uri}")
    lambda_client.update_function_code(
        FunctionName=LAMBDA_FUNCTION_NAME,
        ImageUri=prev_image_uri,
    )
    t0 = time.time()
    while True:
        info = lambda_client.get_function_configuration(FunctionName=LAMBDA_FUNCTION_NAME)
        status = info.get("LastUpdateStatus", "InProgress")
        if status == "Successful":
            break
        if status == "Failed":
            raise RuntimeError("Lambda rollback Failed — intervencion manual")
        if time.time() - t0 > LAMBDA_ACTIVE_TIMEOUT_S:
            raise RuntimeError("Lambda rollback timeout — intervencion manual")
        time.sleep(5)
    print(f"[ROLLBACK] OK en {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# ETAPA 4 — Smoke test contra baseline persistido
# ---------------------------------------------------------------------------

def _select_baseline_audios() -> list[Path]:
    """Selecciona 1 audio fijo por cada especie del baseline. Toma el primer
    archivo alfabeticamente para determinismo.

    Path: scripts/smoke_audios/<species>/ (tracked en git, ~3MB total).
    Movido desde data/raw/ en commit 6.6.a paso post-validacion. Razon:
    data/raw/ esta gitignored -> CI runner no tiene los audios -> el
    workflow rollback.yml tiraba 'baseline audio missing' (config_error,
    exit 2 de smoke_lambda.py), generando smoke_passed=false vacuous en
    DDB que no diferenciaba "smoke no pudo correr" de "smoke fallo".
    """
    selected: list[Path] = []
    for species_dir in SMOKE_BASELINE_AUDIOS:
        d = ROOT / "scripts" / "smoke_audios" / species_dir
        if not d.exists():
            raise FileNotFoundError(f"baseline audio dir missing: {d}")
        files = sorted(d.glob("*.mp3"))
        if not files:
            raise FileNotFoundError(f"no MP3s en {d}")
        selected.append(files[0])
    return selected


def generate_baseline() -> None:
    """Corre el ONNX en produccion ACTUAL (models/classifier.onnx) sobre los
    5 audios fijos, guarda predicciones esperadas a scripts/smoke_baseline.json.

    NOTA: usa el modelo CANONICO (no el candidato). El baseline representa
    "comportamiento que el nuevo modelo debe preservar dentro de tolerancia".
    """
    import onnxruntime as ort
    print("Generando smoke_baseline.json con models/classifier.onnx actual...")

    # Cargar models actual
    onnx_path = MODELS_DIR / "classifier.onnx"
    sidecar_path = MODELS_DIR / "classifier.json"
    if not onnx_path.exists() or not sidecar_path.exists():
        raise FileNotFoundError(
            f"Falta {onnx_path} o {sidecar_path}. Generar baseline primero "
            "requiere modelo deployado en disco."
        )
    sidecar = json.loads(sidecar_path.read_text())
    idx_to_species = {int(k): v for k, v in sidecar["idx_to_species"].items()}

    # Reusar pipeline de extraccion de embeddings desde audios
    from scripts.precompute_embeddings import build_interpreter, embed_waveform

    # Decoder COMPARTIDO con Lambda — Fase 6.6.a paso 5. Antes usabamos
    # load_waveform(path) de precompute_embeddings, que via librosa+audioread
    # podia decodificar audios que Lambda (BytesIO+soundfile sin audioread)
    # rechazaba. Resultado: entries falsos-exitosos en el baseline que
    # daban MISS deterministico en smoke_test contra prod sin ser regresion.
    # Ahora ambos paths comparten la misma funcion; cualquier audio que
    # Lambda no decodifique tampoco entra al baseline (falla aca con error
    # ruidoso explicito, perfecto: el baseline es reproducible-by-construction).
    sys.path.insert(0, str(ROOT / "lambda"))
    from audio_io import decode_audio_bytes  # noqa: E402

    interpreter, input_idx, emb_idx = build_interpreter()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    audios = _select_baseline_audios()
    baseline_entries = []
    skipped_undecodable = []
    for audio_path in audios:
        try:
            y = decode_audio_bytes(audio_path.read_bytes())
        except Exception as e:  # noqa: BLE001
            # No enmascarar: log explicito + skip. El operador ve cual fixture
            # rechaza libsndfile y debe arreglarlo (re-encodear a WAV/MP3-CBR
            # limpio) antes de regenerar baseline.
            print(
                f"  ! SKIP {audio_path.name}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            skipped_undecodable.append({
                "audio_path": str(audio_path.relative_to(ROOT).as_posix()),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        emb = embed_waveform(y, interpreter, input_idx, emb_idx).reshape(1, -1).astype(np.float32)
        logits = sess.run(["logits"], {"embedding": emb})[0]
        # Softmax para confidence
        ex = np.exp(logits[0] - logits[0].max())
        probs = ex / ex.sum()
        top1_idx = int(probs.argmax())
        baseline_entries.append({
            "audio_path": str(audio_path.relative_to(ROOT).as_posix()),
            "species_dir": audio_path.parent.name,
            "expected_top1_species": idx_to_species[top1_idx],
            "expected_top1_confidence": float(probs[top1_idx]),
            "generated_against_checkpoint": sidecar.get("checkpoint_source", "unknown"),
        })
        print(f"  {audio_path.name}: top1={idx_to_species[top1_idx]} "
              f"conf={probs[top1_idx]:.4f}")

    if skipped_undecodable:
        print(
            f"\n! WARNING: {len(skipped_undecodable)} fixture(s) skipped "
            "por no ser decodificables por libsndfile/soundfile. "
            "Re-encodear offline (ffmpeg -i in.mp3 -c:a libmp3lame -b:a 128k -ar 48000 out.mp3) "
            "y re-correr --generate-baseline.",
            file=sys.stderr,
        )
        for s in skipped_undecodable:
            print(f"    - {s['audio_path']}: {s['error']}", file=sys.stderr)

    blob = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tolerances": {
            "min_top1_match_rate": SMOKE_TOP1_THRESHOLD,
            "confidence_tolerance_pp": SMOKE_CONFIDENCE_TOLERANCE_PP,
            "latency_p95_max_s": SMOKE_LATENCY_P95_MAX_S,
        },
        "entries": baseline_entries,
    }
    SMOKE_BASELINE_PATH.write_text(json.dumps(blob, indent=2))
    print(f"\nEscrito: {SMOKE_BASELINE_PATH}")


# Smoke test extraido a scripts/smoke_lambda.py en Fase 6.6.a — usable por
# rollback.yml sin tener que correr todo el flujo de promote. Mantenemos
# el nombre stage4_smoke_test como re-export para no romper imports
# internos ni el flow numbered de "Etapa N" del script.
from scripts.smoke_lambda import smoke_test as _smoke_test_impl  # noqa: E402


def stage4_smoke_test(lambda_client) -> dict[str, Any]:
    """Compara invocacion Lambda contra baseline persistido. Returns dict
    con detalle. ``passed`` field indica overall result.

    Delega a ``scripts.smoke_lambda.smoke_test`` (refactor Fase 6.6.a). El
    print prefix ``[smoke]`` viene del modulo extraido — no replicamos el
    ``[Etapa 4]`` aca para no duplicar lineas. Equivalencia funcional 100%.
    """
    return _smoke_test_impl(lambda_client=lambda_client)


# ---------------------------------------------------------------------------
# Local sync (post-smoke OK)
# ---------------------------------------------------------------------------

def update_local_models(candidate_onnx: Path, candidate_sidecar: Path) -> None:
    """Backup viejo + sync ambas copias del modelo (canonical raiz + deployable
    lambda/models/). Las DOS deben estar sincronizadas para que tests locales
    (que cargan ``models/classifier.onnx``) midan lo mismo que produccion.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    lambda_models_dir = ROOT / "lambda" / "models"

    # Backup canonical (models/)
    backup_onnx = MODELS_DIR / f"classifier_prev_{ts}.onnx"
    backup_json = MODELS_DIR / f"classifier_prev_{ts}.json"
    shutil.copy2(MODELS_DIR / "classifier.onnx", backup_onnx)
    shutil.copy2(MODELS_DIR / "classifier.json", backup_json)

    # Sync canonical
    shutil.copy2(candidate_onnx, MODELS_DIR / "classifier.onnx")
    shutil.copy2(candidate_sidecar, MODELS_DIR / "classifier.json")

    # Sync deployable copy
    shutil.copy2(candidate_onnx, lambda_models_dir / "classifier.onnx")
    shutil.copy2(candidate_sidecar, lambda_models_dir / "classifier.json")

    print(f"[Local sync] backup canonical: {backup_onnx.name} + {backup_json.name}")
    print(f"[Local sync] models/classifier.onnx + lambda/models/classifier.onnx updated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    check_env_guard(args.bypass_promotion_gate)

    # --- Subcomandos especiales (sin evaluation) ---
    if args.generate_baseline:
        generate_baseline()
        return 0

    if args.rollback_to:
        print(f"=== ROLLBACK MANUAL a tag '{args.rollback_to}' ===")
        _, lambda_client = _boto3_clients()
        target_uri = f"{ECR_REPO_FULL}:{args.rollback_to}"
        rollback_lambda(lambda_client, target_uri)
        return 0

    # --- Workflow normal ---
    if args.evaluation is None:
        print("! Falta --evaluation (o usar --rollback-to / --generate-baseline)", file=sys.stderr)
        return 1
    if not args.evaluation.exists():
        print(f"! evaluation no existe: {args.evaluation}", file=sys.stderr)
        return 1

    evaluation = json.loads(args.evaluation.read_text())
    should = bool(evaluation.get("should_promote"))
    reason = evaluation.get("reason", "no reason given")

    print("=== Promote (Fase 6.3) ===")
    print(f"  evaluation:    {args.evaluation}")
    print(f"  should_promote: {should}")
    print(f"  reason:        {reason}")
    print(f"  bypass:        {args.bypass_promotion_gate}")
    print()

    if not should and not args.bypass_promotion_gate:
        print("should_promote=false. NO HAY NADA QUE HACER. Exit 0.")
        return 0
    if not should and args.bypass_promotion_gate:
        print("WARNING: --bypass-promotion-gate activo. Procediendo igualmente.")
        print()

    # Resolver paths del candidato + dataset
    candidate_ckpt = args.candidate_ckpt or Path(evaluation["candidate_ckpt"])
    dataset_path = args.dataset or Path(evaluation["dataset"])
    if not candidate_ckpt.exists():
        # Glob tolerante
        matches = sorted(glob(str(candidate_ckpt)))
        if not matches:
            print(f"! candidate_ckpt no existe: {candidate_ckpt}", file=sys.stderr)
            return 1
        candidate_ckpt = Path(matches[-1])
    if not dataset_path.exists():
        print(f"! dataset no existe: {dataset_path}", file=sys.stderr)
        return 1

    run_name = candidate_ckpt.parent.name
    output_dir = ROOT / "checkpoints" / run_name
    candidate_onnx = output_dir / "classifier.onnx"
    candidate_sidecar = output_dir / "classifier.json"

    # Construir species_to_idx alfabetico (mismo orden que train.py y
    # evaluate.py — consistente con el modelo entrenado).
    df = pd.read_parquet(dataset_path, columns=["species"])
    species_to_idx = {sp: i for i, sp in enumerate(sorted(df["species"].unique()))}
    idx_to_species = {i: sp for sp, i in species_to_idx.items()}

    # ETAPA 1
    stage1_export_onnx(candidate_ckpt, candidate_onnx, candidate_sidecar, idx_to_species)
    skew_info = stage1_skew_check(candidate_ckpt, candidate_onnx, dataset_path)

    # Por default sin flag, comportamiento es export-only
    if not (args.build or args.apply_deploy):
        print("\nNo se paso --build ni --apply-deploy. Termino en Etapa 1.")
        return 0

    # ETAPA 2
    local_tag = stage2_docker_build(candidate_onnx, candidate_sidecar, run_name)
    if args.build and not args.apply_deploy:
        print("\n--build OK. Imagen local lista, no se sube a ECR.")
        return 0

    # ETAPA 3 + 4 (full deploy)
    ecr_client, lambda_client = _boto3_clients()
    stage3_ecr_login(ecr_client)
    new_digest = stage3_docker_push(local_tag, run_name)
    new_uri, prev_uri = stage3_lambda_update(lambda_client, new_digest)
    print()

    try:
        smoke = stage4_smoke_test(lambda_client)
    except Exception as e:
        print(f"\n! Smoke test crasheo: {type(e).__name__}: {e}", file=sys.stderr)
        print("Disparando rollback automatico...", file=sys.stderr)
        rollback_lambda(lambda_client, prev_uri)
        return 2

    if not smoke["passed"]:
        print("\n! Smoke FAIL. Disparando rollback automatico...", file=sys.stderr)
        rollback_lambda(lambda_client, prev_uri)
        return 2

    # Smoke pasa -> sync local
    print()
    update_local_models(candidate_onnx, candidate_sidecar)

    # Resumen final
    summary_path = ROOT / "reports" / run_name / "promotion_result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "candidate_ckpt": str(candidate_ckpt),
        "run_name": run_name,
        "image_uri": new_uri,
        "prev_image_uri": prev_uri,
        "skew_check": skew_info,
        "smoke_test": smoke,
    }, indent=2))
    print(f"\n=== PROMOTION OK ===")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
