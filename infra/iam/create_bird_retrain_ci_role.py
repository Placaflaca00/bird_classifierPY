"""Crea OIDC provider + IAM role ``bird-retrain-ci-role`` para sub-fase 6.4.

Pattern: federated identity con OIDC en lugar de IAM user con access keys.
Este es el patron AWS-recommended 2025 para GitHub Actions:
  - Cero AWS secrets en GitHub (vs el anti-pattern de access keys long-lived).
  - Repo publico + leak imposible (no hay keys que robar).
  - Trust policy scopea a branch especifico — un PR de stranger NO puede asumir
    el role aunque trigger un workflow.

Recursos creados (idempotente: skip silencioso si ya existen):

1. **OIDC provider** ``token.actions.githubusercontent.com``
   - One-shot por cuenta AWS, compartido entre repos.
   - Thumbprint gestionado por AWS (campo vacio cuando se usa GitHub).

2. **IAM role** ``bird-retrain-ci-role``
   - Trust policy: ``infra/iam/bird_retrain_ci_trust_policy.json``
     restringe ``token.actions.githubusercontent.com:sub`` a refs/heads/main +
     refs/heads/feature/retrain-workflow (CRITICO en repo publico: sin esto
     cualquier branch puede asumir el role).
   - Inline policy ``bird-retrain-ci-access``:
     ``infra/iam/bird_retrain_ci_policy.json`` — least-privilege para
     export/train/evaluate/promote.

Cuando el workflow en GH Actions corre, ``aws-actions/configure-aws-credentials``
hace ``sts:AssumeRoleWithWebIdentity`` contra este role usando el ID token
emitido por GitHub. AWS valida que el sub matchea la condition y emite
credenciales temporales (default TTL 1 hora).

Usage
=====
    python infra/iam/create_bird_retrain_ci_role.py             # dry-run
    python infra/iam/create_bird_retrain_ci_role.py --apply     # crea todo
    python infra/iam/create_bird_retrain_ci_role.py --describe  # estado actual
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
ROLE_NAME = "bird-retrain-ci-role"
INLINE_POLICY_NAME = "bird-retrain-ci-access"
TRUST_POLICY_PATH = INFRA_DIR / "bird_retrain_ci_trust_policy.json"
INLINE_POLICY_PATH = INFRA_DIR / "bird_retrain_ci_policy.json"

OIDC_PROVIDER_URL = "token.actions.githubusercontent.com"
OIDC_AUDIENCE = "sts.amazonaws.com"
# Thumbprint placeholder. AWS docs (2023+): "you can leave thumbprints empty
# para GitHub" pero IAM API requiere al menos 1 entry. El thumbprint historico
# de GitHub que sigue siendo aceptado:
OIDC_THUMBPRINT = "6938fd4d98bab03faadb97b34396831e3780aea1"

AWS_ACCOUNT_ID = "863518416901"
AWS_REGION = "us-east-1"
PROJECT_TAGS = [
    {"Key": "Project", "Value": "bird-classifier-py"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "ManagedBy", "Value": "manual"},
    {"Key": "CostCenter", "Value": "portfolio"},
    {"Key": "Phase", "Value": "6"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="Crea OIDC provider + role + inline policy. Idempotente.")
    g.add_argument("--describe", action="store_true",
                   help="Muestra estado actual de OIDC provider + role.")
    return p.parse_args()


def _boto3_client():
    import boto3
    session = boto3.Session()
    if session.get_credentials() is None:
        raise RuntimeError("boto3 no encontro credenciales AWS.")
    return session.client("iam", region_name=AWS_REGION)


def ensure_oidc_provider(iam) -> str:
    """Crea el OIDC provider si no existe. Returns ARN."""
    expected_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:oidc-provider/{OIDC_PROVIDER_URL}"
    existing = iam.list_open_id_connect_providers()["OpenIDConnectProviderList"]
    for p in existing:
        if p["Arn"] == expected_arn:
            print(f"  OIDC provider ya existe: {expected_arn}")
            return expected_arn

    print(f"  Creando OIDC provider {OIDC_PROVIDER_URL}...")
    resp = iam.create_open_id_connect_provider(
        Url=f"https://{OIDC_PROVIDER_URL}",
        ClientIDList=[OIDC_AUDIENCE],
        ThumbprintList=[OIDC_THUMBPRINT],
        Tags=PROJECT_TAGS,
    )
    arn = resp["OpenIDConnectProviderArn"]
    print(f"  OIDC provider creado: {arn}")
    return arn


def ensure_role(iam, trust_policy: dict) -> str:
    """Crea el role si no existe. Returns ARN."""
    try:
        existing = iam.get_role(RoleName=ROLE_NAME)
        print(f"  Role ya existe: {existing['Role']['Arn']}")
        return existing["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"  Creando role {ROLE_NAME}...")
    resp = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=(
            "GitHub Actions CI role para Fase 6 retraining workflow. "
            "Federated via OIDC, no long-lived credentials."
        ),
        MaxSessionDuration=3600,  # 1 hora, default
        Tags=PROJECT_TAGS,
    )
    arn = resp["Role"]["Arn"]
    print(f"  Role creado: {arn}")
    return arn


def ensure_inline_policy(iam, policy_doc: dict) -> None:
    """Pone (overwrites) el inline policy. Idempotente."""
    print(f"  Setting inline policy {INLINE_POLICY_NAME} (overwrite si existe)...")
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=INLINE_POLICY_NAME,
        PolicyDocument=json.dumps(policy_doc),
    )
    print(f"  Inline policy aplicada")


def report_plan(trust_policy: dict, inline_policy: dict) -> None:
    print("=" * 70)
    print("PLAN — OIDC provider + IAM role + inline policy")
    print("=" * 70)
    print()
    print("Recurso 1: OIDC provider")
    print(f"  URL:      https://{OIDC_PROVIDER_URL}")
    print(f"  Audience: {OIDC_AUDIENCE}")
    print(f"  ARN esperado: arn:aws:iam::{AWS_ACCOUNT_ID}:oidc-provider/{OIDC_PROVIDER_URL}")
    print()
    print("Recurso 2: IAM role")
    print(f"  Nombre: {ROLE_NAME}")
    print(f"  Trust policy:")
    print(f"    Principal: Federated -> OIDC provider arriba")
    print(f"    Condition: token.actions.githubusercontent.com:sub IN [")
    for sub in trust_policy["Statement"][0]["Condition"]["StringLike"][
        "token.actions.githubusercontent.com:sub"
    ]:
        print(f"      {sub}")
    print(f"    ]")
    print()
    print("Recurso 3: Inline policy (least-privilege)")
    print(f"  Nombre: {INLINE_POLICY_NAME}")
    print(f"  Statements: {len(inline_policy['Statement'])}")
    for stmt in inline_policy["Statement"]:
        actions = stmt["Action"]
        if isinstance(actions, list):
            n = len(actions)
            preview = ", ".join(a.split(":")[0] for a in actions[:3])
            actions_str = f"{n} actions ({preview}...)"
        else:
            actions_str = actions
        sid = stmt.get("Sid", "<no-sid>")
        eff = stmt["Effect"]
        print(f"    [{eff:5s}] {sid:<32s} {actions_str}")
    print()
    print("Re-ejecutar con --apply para crear todo.")


def describe_state(iam) -> None:
    print("=== Estado actual ===")
    # OIDC provider
    expected_oidc = f"arn:aws:iam::{AWS_ACCOUNT_ID}:oidc-provider/{OIDC_PROVIDER_URL}"
    providers = iam.list_open_id_connect_providers()["OpenIDConnectProviderList"]
    oidc_exists = any(p["Arn"] == expected_oidc for p in providers)
    print(f"OIDC provider:  {'EXISTS' if oidc_exists else 'MISSING'}  {expected_oidc}")

    # Role
    try:
        role_info = iam.get_role(RoleName=ROLE_NAME)["Role"]
        print(f"Role:           EXISTS  {role_info['Arn']}")
        print(f"  Created:      {role_info['CreateDate']}")
        # Trust policy
        trust = role_info.get("AssumeRolePolicyDocument", {})
        stmts = trust.get("Statement", [])
        if stmts:
            cond = stmts[0].get("Condition", {})
            subs = cond.get("StringLike", {}).get(
                "token.actions.githubusercontent.com:sub", []
            )
            print(f"  Sub conditions:")
            for s in subs:
                print(f"    - {s}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role:           MISSING  {ROLE_NAME}")
        return

    # Inline policies
    policies = iam.list_role_policies(RoleName=ROLE_NAME)["PolicyNames"]
    print(f"  Inline policies: {policies}")
    if INLINE_POLICY_NAME in policies:
        pol = iam.get_role_policy(RoleName=ROLE_NAME, PolicyName=INLINE_POLICY_NAME)
        n_stmts = len(pol["PolicyDocument"]["Statement"])
        print(f"  {INLINE_POLICY_NAME}: {n_stmts} statements")


def main() -> int:
    args = parse_args()
    trust_policy = json.loads(TRUST_POLICY_PATH.read_text(encoding="utf-8"))
    inline_policy = json.loads(INLINE_POLICY_PATH.read_text(encoding="utf-8"))

    if args.describe:
        iam = _boto3_client()
        describe_state(iam)
        return 0

    if args.apply:
        iam = _boto3_client()
        print("Aplicando recursos...\n")
        oidc_arn = ensure_oidc_provider(iam)
        # Trust policy ya tiene el OIDC ARN hardcoded; verificar match
        federated = trust_policy["Statement"][0]["Principal"]["Federated"]
        if federated != oidc_arn:
            raise RuntimeError(
                f"trust policy.Federated ({federated}) != OIDC provider ARN "
                f"creado ({oidc_arn}). Actualizar trust_policy JSON."
            )
        ensure_role(iam, trust_policy)
        ensure_inline_policy(iam, inline_policy)
        print()
        describe_state(iam)
        print()
        print("=== OK ===")
        print(f"Role ARN para GitHub Actions:")
        print(f"  arn:aws:iam::{AWS_ACCOUNT_ID}:role/{ROLE_NAME}")
        return 0

    # Dry-run default
    report_plan(trust_policy, inline_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
