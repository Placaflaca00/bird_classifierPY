"""Tests de app/fingerprint.py — ``get_or_create_fingerprint`` (Fase 4d).

``fingerprint.py`` es un módulo puro (``re`` + ``uuid``, sin Gradio) → import
instantáneo, a diferencia de ``import app`` que arrastra todo Gradio (~7-14 s).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

import fingerprint  # noqa: E402

_FP_RE = re.compile(r"^fp_[0-9a-f]{32}$")


class TestGetOrCreateFingerprint:
    def test_none_genera_fingerprint_nuevo(self) -> None:
        """stored_fp None (primera visita) -> genera un fp_<32hex> valido."""
        result = fingerprint.get_or_create_fingerprint(None)
        assert _FP_RE.match(result)

    def test_fingerprint_valido_se_reusa(self) -> None:
        """stored_fp valido -> se devuelve identico, sin regenerar."""
        existing = "fp_" + "a" * 32
        assert fingerprint.get_or_create_fingerprint(existing) == existing

    def test_fingerprint_malformado_regenera(self) -> None:
        """stored_fp que no matchea fp_<32hex> -> genera uno nuevo valido."""
        for bad in ("", "garbage", "fp_xyz", "fp_" + "a" * 31,
                    "fp_" + "A" * 32, "FP_" + "a" * 32):
            result = fingerprint.get_or_create_fingerprint(bad)
            assert result != bad, f"no regeneró para {bad!r}"
            assert _FP_RE.match(result), f"fp inválido para input {bad!r}"
