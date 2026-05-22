"""Fingerprint anónimo por dispositivo (Fase 4d).

Módulo puro — solo ``re`` + ``uuid``, sin Gradio. Separado de ``app.py`` a
propósito: ``import app`` arrastra todo Gradio (~7 s warm, ~14 s cold); este
módulo se importa al instante, así los tests de la lógica del fingerprint no
pagan ese costo. ``app.py`` importa ``get_or_create_fingerprint`` de acá y lo
usa como callback de ``demo.load``.
"""
from __future__ import annotations

import re
import uuid

# Formato canónico del fingerprint — mismo regex que el backend (handler.py).
FINGERPRINT_PATTERN = re.compile(r"^fp_[0-9a-f]{32}$")


def get_or_create_fingerprint(stored_fp: str | None) -> str:
    """Devuelve el fingerprint del usuario; lo crea si no existe o es inválido.

    ``stored_fp`` viene del gr.BrowserState (localStorage). Si es None (primera
    visita, modo incógnito, localStorage limpio) o no matchea el formato
    canónico ``fp_<32hex>``, se genera uno nuevo. Corre como callback de
    ``demo.load`` — per-sesión, lazy — NO como default eager de gr.State
    (Gradio issue #4558: el default eager se comparte entre todos los users).
    """
    if stored_fp and FINGERPRINT_PATTERN.match(stored_fp):
        return stored_fp
    return f"fp_{uuid.uuid4().hex}"
