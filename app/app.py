"""Frontend Gradio para HuggingFace Spaces.

``gr.Blocks`` que toma un audio, lo manda a /predict vía ``client.predict()`` y
renderiza top-3 + ficha de la especie más probable.

Alcance v0 (esta fase): solo predicción. Sin flagging (Fase 4), sin validación
formal del input (fase posterior). Único filtro acá: aviso inline de baja
confianza cuando el top-1 < 0.5.

No depende de ``src/`` — en HF Spaces este archivo y ``client.py`` van al root
del Space, así que el import es directo (sibling).

Test local (exportar la URL primero, igual que client.py):
    $env:API_GATEWAY_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
    python app/app.py
"""
from __future__ import annotations

import gradio as gr

from client import predict

TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.5

# error_kind (de client.PredictResult) -> mensaje user-facing.
_ERROR_MESSAGES = {
    "throttled": (
        "El servicio está recibiendo muchas solicitudes. Esperá unos segundos "
        "y probá de nuevo."
    ),
    "timeout": (
        "El servidor tardó demasiado en responder. Puede ser un audio muy "
        "largo, o el servidor recién despertando — probá de nuevo."
    ),
    "network": (
        "No se pudo conectar con el servidor. Revisá tu conexión y probá de "
        "nuevo."
    ),
    "server": "Error interno del servidor. Probá de nuevo en un rato.",
    "bad_response": (
        "El servidor respondió algo inesperado. Probá de nuevo en un rato."
    ),
    "config": (
        "El servicio está mal configurado (es un problema nuestro, no de tu "
        "audio). Avisá al administrador."
    ),
}


def _format_species_key(pred: dict) -> str:
    """Clave para gr.Label: 'Común (Científico)'.

    El nombre científico siempre está => la clave es única. Si no hay nombre
    común, queda solo el científico.
    """
    species = pred.get("species") or "especie desconocida"
    common = pred.get("common_name")
    return f"{common} ({species})" if common else species


def _build_details(top: dict) -> str:
    """Ficha markdown del top-1, con aviso de baja confianza si corresponde."""
    species = top.get("species") or "especie desconocida"
    common = top.get("common_name") or species
    common_en = top.get("common_name_en")
    conf = top.get("confidence")
    conf_pct = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "?"

    parts: list[str] = []
    if isinstance(conf, (int, float)) and conf < LOW_CONFIDENCE_THRESHOLD:
        parts.append(
            f"**Aviso — confianza baja ({conf_pct}).** El modelo no está "
            "seguro. El audio puede tener mucho ruido, no contener un ave, o "
            "ser de una especie fuera de las que el modelo conoce. Tomá el "
            "resultado con pinzas."
        )

    ficha = [
        f"**Especie más probable:** {common} (*{species}*)",
        f"- Confianza: {conf_pct}",
    ]
    if common_en:
        ficha.append(f"- Nombre común (EN): {common_en}")
    parts.append("\n".join(ficha))
    return "\n\n".join(parts)


def classify(audio_path: str | None):
    """Callback del botón. Devuelve (valor para gr.Label, markdown de detalle).

    En cualquier caso de error devuelve ``gr.update(value=None)`` para el Label:
    limpia lo que hubiera de una clasificación anterior, así no queda un estado
    inconsistente (predicción vieja arriba, error abajo). ``gr.update(value=None)``
    es la forma confiable de resetear el componente.
    """
    if not audio_path:
        return gr.update(value=None), "Subí o grabá un audio primero."

    result = predict(audio_path, top_k=TOP_K)

    if not result.ok:
        if result.error_kind == "bad_request":
            # El detalle del backend ('Format not recognised', etc.) ayuda al
            # usuario a entender qué pasó con su archivo.
            detail = result.error_message or "el archivo no pudo procesarse"
            msg = (
                f"No se pudo procesar el audio: {detail}. "
                "Probá con otro archivo (mp3 o wav, que contenga sonido)."
            )
        else:
            msg = _ERROR_MESSAGES.get(
                result.error_kind, "Ocurrió un error inesperado."
            )
        return gr.update(value=None), msg

    if not result.predictions:
        return gr.update(value=None), "El servidor no devolvió ninguna predicción."

    label = {
        _format_species_key(p): float(p.get("confidence") or 0.0)
        for p in result.predictions
    }
    details = _build_details(result.predictions[0])
    return label, details


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_DESCRIPTION = (
    "Subí o grabá un audio de un ave de Paraguay y el modelo te dice las 3 "
    "especies más probables.\n\n"
    "_Nota: la primera consulta puede tardar ~30 s mientras el servidor "
    "despierta (cold start). Las siguientes son casi instantáneas._"
)

with gr.Blocks(title="Clasificador de aves de Paraguay") as demo:
    gr.Markdown("# Clasificador de aves de Paraguay")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                type="filepath",
                sources=["upload", "microphone"],
                label="Audio del ave",
            )
            submit = gr.Button("Clasificar", variant="primary")
        with gr.Column():
            label_out = gr.Label(num_top_classes=TOP_K, label="Predicción (top 3)")
            details_out = gr.Markdown()

    submit.click(fn=classify, inputs=audio_in, outputs=[label_out, details_out])


if __name__ == "__main__":
    demo.launch()
