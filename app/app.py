"""Frontend Gradio para HuggingFace Spaces — conocetuave.com.py.

``gr.Blocks`` con tema "bosque calido" (verde pastel) y dos tabs:

- **Clasificar**: el flujo principal. Audio in -> /predict -> panel con foto +
  ficha del top-1 + top-3.
- **Las 20 aves**: galeria estatica con las especies que el modelo conoce.

Sub-fase A del rediseno: estructura visual lista; el contenido (fotos en
``assets/birds/`` y descripciones en ``species_info.json``) se llena despues
sin tocar el codigo. Si una foto no existe en disco, el componente se oculta
en lugar de mostrar un broken image; si una descripcion esta vacia, se omite.

Alcance: solo predicir. Sin flagging (Fase 4), sin validacion formal de input
(fase posterior). Unico filtro inline: aviso si top-1 < 0.5.

Test local (exportar la URL primero, igual que client.py):
    $env:API_GATEWAY_URL = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
    python app/app.py
"""
from __future__ import annotations

import html as html_mod
import json
import threading
from pathlib import Path

import gradio as gr

from client import predict, send_feedback, warm_lambda
from fingerprint import get_or_create_fingerprint
from validation import validate_audio

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APP_TITLE = "Conoce Tu Ave Py"
APP_SUBTITLE = "Clasificador de aves de Paraguay"
TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.5

# Caption estatica debajo del audio: gestiona la expectativa de formatos. El
# selector de archivos del navegador a veces griser los .opus segun el SO, y
# el usuario reporto confusion al subir notas de voz de WhatsApp (Ogg/Opus).
# Mencionamos WhatsApp explicito porque es el caso de uso mas comun aca.
FORMATS_HINT = (
    "Formatos: mp3, wav, ogg, flac y notas de voz de WhatsApp (.opus). "
    "Hasta 90 s."
)

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
SPECIES_INFO_PATH = HERE / "species_info.json"

# Fase 4d — fingerprint anónimo por dispositivo para el rate limit del backend.
# gr.BrowserState lo persiste en localStorage; el secret debe ser FIJO o el
# valor no sobrevive un restart del Space (con secret=None Gradio genera uno
# random por arranque → localStorage indescifrable post-restart). Hardcodear es
# seguro acá: encripta un UUID anónimo (no info sensible), HF Spaces aísla cada
# Space en subdominio único (cross-origin previene leakage entre apps), y el
# sufijo -v1 es palanca de reset deliberado. Verificado en 4d.0 (smoke real).
BROWSER_STATE_SECRET = "bird-classifier-fp-v1"

# Fase 4d — consentimiento opt-in. Default DESMARCADO; el "(Opcional)"
# explícito evita que el usuario dude si marcar es requisito (GDPR UX
# best practice — ver Econsultancy "GDPR best practice UX for obtaining
# marketing consent"). GDPR "freely given": el checkbox NO condiciona la
# clasificación — el botón Clasificar funciona marcado o no, y el checkbox
# no se wirea a ningún evento. Su valor se leerá como input de classify()
# recién en 4d.2c. Disclosure inline simple; la versión layered (label
# corto + accordion) se difiere a 4e.
CONSENT_LABEL = (
    "(Opcional) Permito usar este audio para mejorar el modelo, lo que puede "
    "incluir revisión manual por colaboradores del proyecto. Audio licenciado "
    "bajo Creative Commons BY 4.0."
)

# Paleta "bosque calido" — verde oliva, sensacion organica.
BG_DEEP = "#2f3a28"      # fondo de la pagina
BG_CARD = "#3d4933"      # bloques / paneles
PRIMARY = "#5a7d4a"      # botones, links
SECONDARY = "#b8d4a8"    # highlights, acentos
TEXT = "#ecf0e4"
TEXT_DIM = "#c0c8b8"

# error_kind (de client.PredictResult) -> mensaje user-facing.
_ERROR_MESSAGES = {
    "throttled": (
        "El servicio esta recibiendo muchas solicitudes. Espera unos segundos "
        "y proba de nuevo."
    ),
    "timeout": (
        "El servidor tardo demasiado en responder. Puede ser un audio muy "
        "largo, o el servidor recien despertando — proba de nuevo."
    ),
    "network": (
        "No se pudo conectar con el servidor. Revisa tu conexion y proba de "
        "nuevo."
    ),
    "server": "Error interno del servidor. Proba de nuevo en un rato.",
    "bad_response": (
        "El servidor respondio algo inesperado. Proba de nuevo en un rato."
    ),
    "config": (
        "El servicio esta mal configurado (es un problema nuestro, no de tu "
        "audio). Avisa al administrador."
    ),
    # Fase 3: error en el PUT directo a S3 (presigned URL). Normalmente es
    # conexion inestable; el usuario reintenta y suele resolverse.
    "upload_failed": (
        "No se pudo subir el audio al servidor. Revisa tu conexion y proba "
        "de nuevo (puede ser un audio muy grande, o conexion inestable)."
    ),
    # Fase 4 — 429 del backend: cuota diaria agotada. Distinto de "throttled"
    # (503, throttle transitorio del API Gateway): "rate_limited" es terminal,
    # no se reintenta. El conteo "(30 audios)" es literal — la version dinamica
    # (leer rate_info["limit"]) es future work (ver phase4d-plan, seccion 4e).
    "rate_limited": (
        "Llegaste al límite diario (30 audios). Volvé mañana para "
        "clasificar más."
    ),
}

# Fase 2 - Nivel 2 — mensajes user-facing cuando el backend rechaza el audio
# por gating (no por error tecnico). reject_reason -> texto al usuario.
# Estilo sin acentos para matchear el resto del codebase (cf. _ERROR_MESSAGES).
_REJECT_MESSAGES = {
    "not_a_bird": (
        "No detecte un ave en el audio. Proba con una grabacion donde el "
        "canto del ave sea claro."
    ),
    "white_noise": (
        "El audio parece ruido sintetico. Proba con una grabacion de campo "
        "de un pajaro."
    ),
    "pure_tone": (
        "El audio es un tono puro (sin armonicos de canto). Proba con una "
        "grabacion de un pajaro real."
    ),
}


# ---------------------------------------------------------------------------
# Carga module-level de species_info
# ---------------------------------------------------------------------------
def _load_species_info() -> dict[str, dict]:
    """species_info.json -> {species_sci: {es, en, description, photo}}. {} si falla."""
    try:
        with open(SPECIES_INFO_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("species", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


_SPECIES_INFO = _load_species_info()


# ---------------------------------------------------------------------------
# Helpers de presentacion
# ---------------------------------------------------------------------------
def _format_species_key(pred: dict) -> str:
    """Clave para gr.Label: 'Comun (Cientifico)'. Cientifico siempre presente => unica."""
    species = pred.get("species") or "especie desconocida"
    common = pred.get("common_name")
    return f"{common} ({species})" if common else species


def _resolve_bird_photo(species: str) -> str | None:
    """Path absoluto de la foto si existe en disco, None si no."""
    info = _SPECIES_INFO.get(species)
    if not info:
        return None
    rel = info.get("photo") or ""
    if not rel:
        return None
    abspath = HERE / rel
    return str(abspath) if abspath.exists() else None


def _photo_credit_html(info: dict) -> str:
    """HTML chico con el credito + licencia de la foto, o '' si no hay credito.

    La atribucion es legalmente requerida para fotos CC-BY*. El texto viene
    formateado tal cual iNaturalist lo provee.
    """
    credit = (info.get("photo_credit") or "").strip()
    if not credit:
        return ""
    return f'<div class="photo-credit">Foto: {credit}</div>'


def _build_details(top: dict) -> str:
    """Ficha markdown del top-1 con aviso de baja confianza si corresponde."""
    species = top.get("species") or "especie desconocida"
    common = top.get("common_name") or species
    common_en = top.get("common_name_en")
    conf = top.get("confidence")
    conf_pct = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "?"

    parts: list[str] = []
    if isinstance(conf, (int, float)) and conf < LOW_CONFIDENCE_THRESHOLD:
        parts.append(
            f"**Confianza baja ({conf_pct}).** El modelo no esta seguro. "
            "El audio puede tener ruido, no contener un ave, o ser de una "
            "especie fuera de las 20 que conoce."
        )

    ficha = [f"### {common}", f"*{species}*", "", f"**Confianza:** {conf_pct}"]
    if common_en:
        ficha.append(f"**En ingles:** {common_en}")

    info = _SPECIES_INFO.get(species, {})
    desc = (info.get("description") or "").strip()
    if desc:
        ficha.extend(["", desc])

    credit_html = _photo_credit_html(info)
    if credit_html:
        ficha.extend(["", credit_html])

    parts.append("\n".join(ficha))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# State machine UI (Fase 2 - Nivel 1 UX)
# ---------------------------------------------------------------------------
# Cuatro estados gobiernan el par (boton submit, hint debajo del audio):
#
#   IDLE        — sin audio cargado. Submit disabled. Hint pide audio.
#   RECORDING   — usuario grabando con mic. Submit disabled. Hint avisa.
#   READY       — hay audio + no se esta grabando + no se esta clasificando.
#                 Submit enabled. Hint vacio.
#   PROCESSING  — classify() en curso. Submit disabled. Hint avisa.
#
# Eventos que disparan transiciones:
#   audio_in.start_recording  -> RECORDING (override de cualquier estado)
#   audio_in.change           -> READY si hay value, IDLE si no
#                                (cubre upload, clear, y stop_recording)
#   submit.click chain:       READY -> PROCESSING -> (READY | IDLE segun audio)
#
# El boton arranca disabled (IDLE) en el constructor. Prevention > recovery:
# es preferible que el usuario no pueda clickear en momento invalido a tener
# que manejarlo en classify().

_HINT_IDLE = "Subi o graba un audio primero."
_HINT_RECORDING = "Grabando... terminá la grabación para clasificar."
_HINT_PROCESSING = "Clasificando..."
_HINT_READY = ""


def _state_idle() -> tuple:
    return gr.update(interactive=False), gr.update(value=_HINT_IDLE)


def _state_recording() -> tuple:
    return gr.update(interactive=False), gr.update(value=_HINT_RECORDING)


def _state_ready() -> tuple:
    return gr.update(interactive=True), gr.update(value=_HINT_READY)


def _state_processing() -> tuple:
    return gr.update(interactive=False), gr.update(value=_HINT_PROCESSING)


def _on_audio_change(audio_path: str | None) -> tuple:
    """audio_in.change: READY si hay audio, IDLE si no."""
    return _state_ready() if audio_path else _state_idle()


# ---------------------------------------------------------------------------
# Callback del clasificador
# ---------------------------------------------------------------------------
def _hidden_result() -> tuple:
    """Tupla de updates que oculta el panel de resultado y limpia sus campos.

    Outputs: (result_group, bird_photo, bird_info, label_out)
    """
    return (
        gr.update(visible=False),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=None),
    )


# Fase 4d.2d — quota indicator (escalating visibility). Thresholds 10/5/1:
# >=10 oculto (uso normal; ~33% es el corte generico de Moesif), 5-9 heads-up
# suave, 1-4 alerta con emoji. A 0 (y negativo) oculto: el mensaje 429 ya
# cubre el limite alcanzado, y "Te quedan 0" sonaria a advertencia cuando la
# realidad es "ya no hay".
def _format_quota_indicator(rate_info: dict | None):
    """gr.update para quota_indicator segun escalating visibility.

    ``rate_info`` (de ``PredictResult``) trae ``remaining``. Oculto si no hay
    info utilizable — no mostrar un valor posiblemente stale es mejor que
    mentir.
    """
    if not rate_info or "remaining" not in rate_info:
        return gr.update(visible=False, value="")

    remaining = rate_info["remaining"]

    # Oculto en uso normal (>=10) y tambien en 0/negativo: a 0 el mensaje 429
    # ya cubre; un remaining negativo (backend bug) no debe renderizarse.
    if remaining >= 10 or remaining <= 0:
        return gr.update(visible=False, value="")

    # Concordancia singular/plural (verbo + sustantivo) cuando remaining == 1.
    verbo = "queda" if remaining == 1 else "quedan"
    noun = "clasificación" if remaining == 1 else "clasificaciones"
    texto = f"Te {verbo} {remaining} {noun} hoy."

    if remaining >= 5:
        # Heads-up suave (5-9 restantes).
        return gr.update(visible=True, value=texto)

    # Alerta (1-4 restantes) — el emoji refuerza la escalation visual.
    return gr.update(visible=True, value=f"⚠️ {texto}")


# Fase 4d.3 — 5-tupla de feedback para el return de classify(). Resetea el
# estado del feedback en cada clasificacion (correction group + status
# ocultos), decide si mostrar los botones 👍/👎 (solo en deteccion exitosa),
# y actualiza el grid de correccion (HTML sin la especie ya predicha).
def _feedback_reset(prediction_id: str | None, show_buttons: bool,
                    grid_update) -> tuple:
    """Outputs (prediction_id_state, feedback_buttons_group,
    feedback_correction_group, feedback_status, feedback_grid) para el return
    de classify(). ``grid_update`` = HTML str del grid, o gr.update() no-op.
    """
    return (
        prediction_id,
        gr.update(visible=show_buttons),
        gr.update(visible=False),
        gr.update(value=""),  # feedback_status: always-mounted, solo se limpia
        grid_update,
    )


# Fase 5A.1 — outputs no-op para el bloque de feedback en error paths de
# classify(). Preserva visibility y contenido de los 5 componentes (incluido
# el prediction_id_state) — un error transitorio NO debe ocultar los botones
# de feedback que vienen de una clasificacion exitosa anterior.
#
# Bug observable que arregla (Fase 5A.1): tras un primer intento que erra por
# cold start del Lambda (API GW corta a 30s → "No se pudo conectar"), el
# usuario reintenta y clasifica OK, pero los botones de feedback no aparecen.
# Causa: el error path llamaba `_feedback_reset(None, False, gr.update())`,
# que mandaba `gr.update(visible=False)` sobre feedback_buttons_group. El fix
# es preservar el estado actual del bloque en error paths — los botones
# solo deben aparecer/ocultarse en transiciones exitosas/iniciales.
#
# Si los botones quedan visibles tras un error, refieren a la ULTIMA
# prediccion exitosa (cuyo prediction_id se preserva en el State); un click
# en 👍/👎 manda feedback sobre esa, no sobre el intento fallido — lo cual
# es semanticamente correcto.
def _feedback_preserve_all() -> tuple:
    """5-tupla de gr.update() no-op: preserva el estado actual de los 5
    componentes de feedback. Usado en TODOS los error paths de classify().
    """
    return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())


def classify(audio_path: str | None, fingerprint: str | None, training_consent: bool):
    """Callback del boton Clasificar.

    Inputs (orden importa, matchea inputs=[...] del .click() abajo):
        0. audio_path        -> ruta del audio cargado
        1. fingerprint       -> valor del gr.BrowserState (puede ser None/invalido)
        2. training_consent  -> bool del consent_checkbox

    Outputs (orden importa, matchea el .click() abajo):
        0. result_group              -> visible True/False
        1. error_box                 -> visible + value (texto del error)
        2. bird_photo                -> Image, path o None
        3. bird_info                 -> Markdown con ficha
        4. label_out                 -> dict {label: prob} para gr.Label
        5. quota_indicator           -> gr.update (escalating visibility)
        6. prediction_id_state       -> str (prediction_id) o None
        7. feedback_buttons_group    -> visible (True solo en deteccion ok)
        8. feedback_correction_group -> visible False (reset)
        9. feedback_status           -> visible False, value "" (reset)
       10. feedback_grid             -> HTML del grid sin la especie predicha
    Los outputs 6-10 los arma _feedback_reset() (Fase 4d.3).

    Asimetria posicional/keyword: Gradio pasa los inputs POSICIONALES; se
    reenvian a ``predict()`` como KEYWORD-ONLY (``fingerprint=...``,
    ``training_consent=...``) porque ``predict()`` los declara kw-only a
    proposito (explicit dependency declaration en el cliente HTTP).

    Fase 4d (A'): ``fingerprint`` se normaliza in-memory con
    ``get_or_create_fingerprint`` apenas entra — si el BrowserState devolvio
    None/invalido se regenera, asi ``predict()`` nunca recibe un fp corrupto
    (un 400 por fingerprint daria un mensaje que culpa al audio). NO se
    persiste de vuelta a localStorage; es solo normalizacion del valor.

    En CUALQUIER caso de error: el panel de resultado queda oculto y sus
    campos limpios; solo el error_box muestra el mensaje. Asi no queda un
    estado mixto (foto vieja arriba + error abajo).

    Precondicion garantizada por la state machine UI: ``audio_path`` nunca
    es None ni vacio cuando este callback se invoca (el boton arranca
    disabled y solo se habilita en estado READY). Si llegara None de todas
    formas (bug en el wiring), ``validate_audio`` lo mapea a "unreadable"
    sin crashear.
    """
    # Fase 4d (A'): normalizacion in-memory del fingerprint del BrowserState
    # (ver docstring). get_or_create_fingerprint ya esta importado arriba.
    fingerprint = get_or_create_fingerprint(fingerprint)

    # Fase 2 - Nivel 1: validacion inline antes de invocar la Lambda.
    # Si falla, ahorramos cold start y devolvemos un mensaje especifico.
    validation = validate_audio(audio_path)
    if not validation.ok:
        hidden = _hidden_result()
        return (
            hidden[0],
            gr.update(visible=True, value=validation.error_message),
            hidden[1], hidden[2], hidden[3],
            _format_quota_indicator(None),
            *_feedback_preserve_all(),
        )

    result = predict(
        audio_path,
        top_k=TOP_K,
        fingerprint=fingerprint,
        training_consent=training_consent,
    )
    # Fase 4d.2d — indicador de cuota: se computa una vez del rate_info de
    # esta respuesta y se reusa en todos los returns post-predict.
    quota = _format_quota_indicator(result.rate_info)

    if not result.ok:
        if result.error_kind == "bad_request":
            detail = result.error_message or "el archivo no pudo procesarse"
            msg = (
                f"No se pudo procesar el audio: {detail}. "
                "Proba con otro archivo (mp3 o wav, que contenga sonido)."
            )
        else:
            msg = _ERROR_MESSAGES.get(
                result.error_kind, "Ocurrio un error inesperado."
            )
        # Fase 5A.1: toast extra en errores de cold-start aún despues del retry
        # del client.py. Si el usuario llega aca con timeout/network, ya hubo
        # 2 intentos contra el backend (timeout total ~35s + 60s = 95s) y aun
        # asi fallo — el container probablemente esta tomando >60s o hay un
        # problema de red. El toast da contexto adicional al texto del error.
        if result.error_kind in ("timeout", "network"):
            gr.Warning(
                "El servidor sigue iniciandose o tu conexion esta inestable. "
                "Esperá unos segundos y volvé a probar — la siguiente request "
                "deberia ser casi instantanea."
            )
        hidden = _hidden_result()
        return (
            hidden[0],
            gr.update(visible=True, value=msg),
            hidden[1], hidden[2], hidden[3],
            quota,
            *_feedback_preserve_all(),
        )

    # Fase 2 - Nivel 2: backend rechazo el audio por gating. Mensaje
    # especifico por reason. Panel de resultado oculto (sin foto/ficha
    # colgando del estado anterior) y error_box con texto user-friendly.
    if not result.detected:
        reason = result.reject_reason or "not_a_bird"
        msg = _REJECT_MESSAGES.get(reason, _REJECT_MESSAGES["not_a_bird"])
        hidden = _hidden_result()
        return (
            hidden[0],
            gr.update(visible=True, value=msg),
            hidden[1], hidden[2], hidden[3],
            quota,
            *_feedback_preserve_all(),
        )

    if not result.predictions:
        hidden = _hidden_result()
        return (
            hidden[0],
            gr.update(visible=True, value="El servidor no devolvio predicciones."),
            hidden[1], hidden[2], hidden[3],
            quota,
            *_feedback_preserve_all(),
        )

    top1 = result.predictions[0]
    species = top1.get("species") or ""
    photo = _resolve_bird_photo(species)
    info_md = _build_details(top1)
    labels = {
        _format_species_key(p): float(p.get("confidence") or 0.0)
        for p in result.predictions
    }

    return (
        gr.update(visible=True),                  # result_group
        gr.update(visible=False, value=""),       # error_box
        gr.update(value=photo, visible=photo is not None),  # bird_photo
        info_md,                                  # bird_info
        labels,                                   # label_out
        quota,                                    # quota_indicator
        *_feedback_reset(
            result.prediction_id, True,
            _render_feedback_grid(exclude=species),
        ),  # feedback (4d.3): grid sin la especie predicha
    )


# ---------------------------------------------------------------------------
# Fase 4d.3 — UI de feedback (👍/👎 + correccion por galeria)
# ---------------------------------------------------------------------------
# error_kind de client.FeedbackResult -> mensaje user-facing. Sin entrada para
# un kind => fallback generico. Mismo patron que _ERROR_MESSAGES.
_FEEDBACK_ERRORS = {
    "already_submitted": "Ya diste feedback para esta predicción.",
    "not_found": (
        "No se encontró esa predicción. Probá clasificando un nuevo audio."
    ),
}
_FEEDBACK_OK = "¡Gracias! Tu feedback nos ayuda a mejorar las predicciones."
_FEEDBACK_FALLBACK = "No se pudo registrar tu feedback. Probá de nuevo en un rato."


# Orden canonico de las 20 aves con foto. Lo usan _render_feedback_grid (el
# grid visual) y el wiring de los botones-puente — MISMO orden, asi la card #N
# del grid clickea el boton fb-pick-N, que tiene la especie #N fijada.
_FEEDBACK_SPECIES = [s for s in sorted(_SPECIES_INFO) if _resolve_bird_photo(s)]


def _render_feedback_grid(exclude: str | None = None) -> str:
    """Grid HTML clickeable de las aves — reemplaza gr.Gallery (B1).

    Si ``exclude`` es un nombre cientifico, esa card se omite (4d.3): cuando el
    modelo predijo una especie y el usuario marca 👎, esa especie NO se ofrece
    como correccion — es ilogico (👎 = "no es eso") y bloquea feedback
    troll/contradictorio (corrected_species == lo que el modelo predijo).
    Quedan 19 cards.

    gr.Gallery colgaba el evento que lo abre en este app (ver memoria
    gradio-dynamic-visibility); gr.HTML SI funciona (Tab 2 lo prueba). El
    onclick de cada card clickea un gr.Button oculto (fb-pick-<i>): el .click()
    de un gr.Button es 100% confiable. Cada boton tiene su especie fijada por
    closure => la especie viaja directo. Las cards NO excluidas mantienen su
    indice original `i` (el del excluido simplemente no se renderiza).
    """
    cards: list[str] = []
    for i, species in enumerate(_FEEDBACK_SPECIES):
        if species == exclude:
            continue
        info = _SPECIES_INFO[species]
        common = html_mod.escape(info.get("es") or species)
        url = "/gradio_api/file=" + str(_resolve_bird_photo(species)).replace(
            "\\", "/")
        onclick = "document.getElementById('fb-pick-" + str(i) + "').click()"
        cards.append(
            '<div class="fb-bird-card" data-species="'
            + html_mod.escape(species) + '" title="' + common + '" onclick="'
            + onclick + '">'
            '<div class="fb-bird-img"><img src="' + url + '" loading="lazy" '
            'decoding="async" alt="' + common + '"></div>'
            '<div class="fb-bird-name">' + common + '</div></div>'
        )
    return '<div class="fb-bird-grid">' + "".join(cards) + "</div>"


def _feedback_status_update(result):
    """gr.update para feedback_status segun el FeedbackResult de send_feedback."""
    if result.ok:
        msg = _FEEDBACK_OK
    else:
        msg = _FEEDBACK_ERRORS.get(result.error_kind, _FEEDBACK_FALLBACK)
    return gr.update(value=msg)  # feedback_status always-mounted: solo value


def _on_thumbs_up(prediction_id: str | None, fingerprint: str | None) -> tuple:
    """👍 Acertó -> feedback 'confirmed'. Oculta los botones, muestra el status.

    Outputs: (feedback_buttons_group, feedback_status)
    """
    result = send_feedback(prediction_id or "", fingerprint or "", "confirmed")
    return gr.update(visible=False), _feedback_status_update(result)


def _on_thumbs_down() -> tuple:
    """👎 Falló -> oculta los botones, abre el grupo de correccion (grid HTML).

    Outputs: (feedback_buttons_group, feedback_correction_group)
    """
    return gr.update(visible=False), gr.update(visible=True)


def _on_pick_species(
    species: str, prediction_id: str | None, fingerprint: str | None
) -> tuple:
    """Click en una card del grid -> feedback 'corrected' (B1).

    ``species`` (nombre cientifico) viene fijado por closure en el wiring del
    boton-puente fb-pick-<i> que la card clickeo — viaja directo, sin indice.
    Outputs: (feedback_correction_group, feedback_status)
    """
    result = send_feedback(
        prediction_id or "", fingerprint or "", "corrected",
        corrected_species=species,
    )
    return gr.update(visible=False), _feedback_status_update(result)


def _on_non_bird(prediction_id: str | None, fingerprint: str | None) -> tuple:
    """'Esto no era un pájaro' -> feedback 'rejected_as_non_bird'.

    Outputs: (feedback_correction_group, feedback_status)
    """
    result = send_feedback(
        prediction_id or "", fingerprint or "", "rejected_as_non_bird"
    )
    return gr.update(visible=False), _feedback_status_update(result)


def _on_cancel() -> tuple:
    """'Volver' -> cierra la correccion, vuelve a mostrar 👍/👎.

    Outputs: (feedback_buttons_group, feedback_correction_group)
    """
    return gr.update(visible=True), gr.update(visible=False)


# ---------------------------------------------------------------------------
# Tab 2 — galeria de las 20 aves
# ---------------------------------------------------------------------------
def _render_species_gallery_html() -> str:
    """Tab 2 entero como UN solo string HTML.

    Reemplaza los 20 pares (gr.Image + gr.Markdown) anteriores por cards puras
    HTML. Beneficios:
    - HTML inicial ~30 KB en vez de ~130 KB (cada gr.Image inflaba el payload).
    - `loading="lazy"` difiere la descarga de cada foto hasta que esta cerca
      del viewport. Si el usuario no abre tab 2, las fotos NUNCA se descargan.
    """
    cards: list[str] = []
    for species, info in sorted(_SPECIES_INFO.items()):
        photo_path = _resolve_bird_photo(species)
        common = info.get("es") or species
        en = info.get("en") or ""
        desc = (info.get("description") or "").strip()
        credit = (info.get("photo_credit") or "").strip()

        img_html = ""
        if photo_path:
            # Forward slashes para que la URL no rompa en Windows.
            url = "/gradio_api/file=" + str(photo_path).replace("\\", "/")
            alt = html_mod.escape(common)
            img_html = (
                '<div class="bird-card-img">'
                f'<img src="{url}" loading="lazy" decoding="async" alt="{alt}">'
                "</div>"
            )

        body_parts = [
            f'<div class="bird-card-name">{html_mod.escape(common)}</div>',
            f'<div class="bird-card-sci">{html_mod.escape(species)}</div>',
        ]
        if en:
            body_parts.append(
                f'<div class="bird-card-en">{html_mod.escape(en)}</div>'
            )
        if desc:
            body_parts.append(
                f'<div class="bird-card-desc">{html_mod.escape(desc)}</div>'
            )
        else:
            body_parts.append(
                '<div class="bird-card-desc"><em>contenido proximamente</em></div>'
            )
        if credit:
            body_parts.append(
                f'<div class="photo-credit">Foto: {html_mod.escape(credit)}</div>'
            )

        body = '<div class="bird-card-body">' + "".join(body_parts) + "</div>"
        cards.append(f'<div class="bird-card-html">{img_html}{body}</div>')

    return '<div class="bird-grid">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Tema "bosque calido"
# ---------------------------------------------------------------------------
def _build_theme() -> gr.themes.Base:
    """gr.themes.Soft tunneado a la paleta verde oliva."""
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="stone",
    ).set(
        body_background_fill=BG_DEEP,
        body_background_fill_dark=BG_DEEP,
        body_text_color=TEXT,
        body_text_color_dark=TEXT,
        body_text_color_subdued=TEXT_DIM,
        body_text_color_subdued_dark=TEXT_DIM,
        background_fill_primary=BG_CARD,
        background_fill_primary_dark=BG_CARD,
        background_fill_secondary=BG_DEEP,
        background_fill_secondary_dark=BG_DEEP,
        block_background_fill=BG_CARD,
        block_background_fill_dark=BG_CARD,
        block_border_color=PRIMARY,
        block_border_color_dark=PRIMARY,
        block_label_text_color=SECONDARY,
        block_label_text_color_dark=SECONDARY,
        block_title_text_color=SECONDARY,
        block_title_text_color_dark=SECONDARY,
        button_primary_background_fill=PRIMARY,
        button_primary_background_fill_dark=PRIMARY,
        button_primary_background_fill_hover="#6f9059",
        button_primary_background_fill_hover_dark="#6f9059",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        color_accent=SECONDARY,
        color_accent_soft="#4c5d40",
        color_accent_soft_dark="#4c5d40",
        border_color_primary=PRIMARY,
        border_color_primary_dark=PRIMARY,
        panel_background_fill=BG_CARD,
        panel_background_fill_dark=BG_CARD,
    )
    return theme


# ---------------------------------------------------------------------------
# CSS custom
# ---------------------------------------------------------------------------
# Usamos placeholders {{TOKEN}} + .replace() en vez de % formatting porque el
# CSS tiene chars % literales (max-width: 100%) que rompen el formatter.
_CSS = (
    """
.app-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 24px;
    padding: 18px 0 18px 0;
    flex-wrap: wrap;
}
.app-header img {
    height: 180px;
    width: auto;
    max-width: 100%;
}
.app-header-text {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
}
.app-title {
    font-size: 2.6em;
    font-weight: 700;
    color: {{SECONDARY}};
    margin: 0;
    letter-spacing: 0.5px;
    line-height: 1.1;
}
.app-subtitle {
    color: {{TEXT_DIM}};
    margin: 6px 0 0 0;
    font-size: 1em;
}
.cold-note {
    text-align: center;
    color: {{TEXT_DIM}};
    font-size: 0.82em;
    margin-bottom: 14px;
    font-style: italic;
}
.audio-block {
    padding: 6px 0 0 0;
}
.classify-btn {
    margin-top: 4px;
}
.status-hint {
    text-align: center;
    color: {{TEXT_DIM}};
    font-size: 0.88em;
    font-style: italic;
    margin: 6px 0 2px 0;
    min-height: 1.3em;  /* reserva altura para que el layout no salte cuando esta vacio */
}
.formats-hint {
    text-align: center;
    color: {{TEXT_DIM}};
    font-size: 0.8em;
    margin: 4px 0 0 0;
    opacity: 0.8;
}
.error-box {
    background: #5a3a3a !important;
    border: 1px solid #a06060 !important;
    border-radius: 8px;
    padding: 10px 14px !important;
    color: #f5dada !important;
}
.bird-card {
    padding: 10px;
    border-radius: 10px;
    background: {{BG_CARD}};
}
.photo-credit {
    font-size: 0.72em;
    color: {{TEXT_DIM}};
    margin-top: 6px;
    opacity: 0.75;
    line-height: 1.35;
}
.fb-hidden {
    display: none !important;
}
.fb-bird-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 8px;
    max-height: 420px;
    overflow-y: auto;
}
.fb-bird-card {
    background: {{BG_CARD}};
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border-color 0.15s;
}
.fb-bird-card:hover {
    border-color: {{SECONDARY}};
}
.fb-bird-img {
    width: 100%;
    height: 110px;
    overflow: hidden;
    background: rgba(0,0,0,0.15);
}
.fb-bird-img img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.fb-bird-name {
    padding: 5px 7px;
    font-size: 0.8em;
    color: {{SECONDARY}};
    text-align: center;
}
.bird-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
    margin-top: 12px;
}
.bird-card-html {
    background: {{BG_CARD}};
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.bird-card-img {
    width: 100%;
    height: 160px;
    overflow: hidden;
    background: rgba(0,0,0,0.15);
}
.bird-card-img img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.bird-card-body {
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
}
.bird-card-name {
    font-weight: 600;
    color: {{SECONDARY}};
    margin-bottom: 2px;
}
.bird-card-sci {
    font-style: italic;
    font-size: 0.85em;
    color: {{TEXT_DIM}};
    margin-bottom: 2px;
}
.bird-card-en {
    font-size: 0.78em;
    color: {{TEXT_DIM}};
    margin-bottom: 6px;
    opacity: 0.85;
}
.bird-card-desc {
    font-size: 0.85em;
    line-height: 1.4;
    margin-bottom: 6px;
}
"""
    .replace("{{SECONDARY}}", SECONDARY)
    .replace("{{TEXT_DIM}}", TEXT_DIM)
    .replace("{{BG_CARD}}", BG_CARD)
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _header_html() -> str:
    """Header con logo (si existe) al lado de titulo + subtitulo en columna."""
    logo_tag = ""
    if LOGO_PATH.exists():
        logo_tag = f'<img src="/gradio_api/file={LOGO_PATH}" alt="logo" />'
    return (
        '<div class="app-header">'
        f"{logo_tag}"
        '<div class="app-header-text">'
        f'<h1 class="app-title">{APP_TITLE}</h1>'
        f'<p class="app-subtitle">{APP_SUBTITLE}</p>'
        "</div>"
        "</div>"
    )


with gr.Blocks(theme=_build_theme(), title=APP_TITLE, css=_CSS) as demo:
    gr.HTML(_header_html())

    # Estado app-wide (fuera de los tabs). El fingerprint persiste en
    # localStorage del browser; se inicializa en el demo.load del final.
    fingerprint_state = gr.BrowserState(
        default_value=None,
        storage_key="bird_classifier_fp",
        secret=BROWSER_STATE_SECRET,
    )

    with gr.Tabs():
        # ----- Tab 1: Clasificar -----
        with gr.Tab("Clasificar"):
            with gr.Column(elem_classes="audio-block"):
                audio_in = gr.Audio(
                    type="filepath",
                    sources=["upload", "microphone"],
                    label="Audio del ave",
                    show_label=True,
                )
                # Caption estatica de formatos aceptados (no es el hint dinamico
                # de estado — ese es status_hint, debajo).
                gr.Markdown(FORMATS_HINT, elem_classes="formats-hint")
                # State machine UI: hint debajo del audio + boton que arranca
                # disabled. Solo se habilita cuando hay audio cargado.
                status_hint = gr.Markdown(
                    value=_HINT_IDLE, elem_classes="status-hint"
                )
                # Consent opt-in (4d.2b): default DESMARCADO, SIN wiring.
                # No tiene .change() ni es input de evento -> no dispara
                # classify(). Su valor se leerá como input recién en 4d.2c.
                consent_checkbox = gr.Checkbox(
                    value=False,
                    label=CONSENT_LABEL,
                    elem_id="consent-checkbox",
                )
                submit = gr.Button(
                    "Clasificar",
                    variant="primary",
                    size="lg",
                    interactive=False,
                    elem_classes="classify-btn",
                )

            error_box = gr.Markdown(
                value="", visible=False, elem_classes="error-box"
            )

            with gr.Group(visible=False) as result_group:
                with gr.Row():
                    bird_photo = gr.Image(
                        label=None,
                        show_label=False,
                        interactive=False,
                        visible=False,
                        height=240,
                    )
                    bird_info = gr.Markdown()
                label_out = gr.Label(
                    num_top_classes=TOP_K, label="Top 3 predicciones"
                )

            # ---- Fase 4d.3: UI de feedback ----
            # prediction_id de la ultima clasificacion; lo setea classify().
            # gr.State (no BrowserState): estado de sesion, no persiste.
            prediction_id_state = gr.State(None)

            # Botones 👍/👎 — visibles solo tras una deteccion exitosa.
            with gr.Row(visible=False) as feedback_buttons_group:
                feedback_thumbs_up = gr.Button(
                    "👍 Acertó", variant="primary",
                    elem_id="feedback-thumbs-up",
                )
                feedback_thumbs_down = gr.Button(
                    "👎 Falló", variant="secondary",
                    elem_id="feedback-thumbs-down",
                )

            # Correccion — grid HTML clickeable de las 20 especies; se abre
            # con 👎. B1: gr.HTML, NO gr.Gallery — gr.Gallery cuelga el evento
            # en este app (ver memoria gradio-dynamic-visibility). El click de
            # una card escribe en feedback_pick (Textbox-puente) via JS inline
            # y eso dispara _on_pick_species.
            with gr.Group(visible=False) as feedback_correction_group:
                gr.Markdown("¿Cuál era el pájaro? Tocá la foto correcta.")
                feedback_grid = gr.HTML(_render_feedback_grid())
                gr.Markdown(
                    "Fotos: colaboradores de iNaturalist y Wikimedia Commons "
                    "— créditos completos en «Las 20 aves».",
                    elem_classes="photo-credit",
                )
                with gr.Row():
                    feedback_non_bird_btn = gr.Button(
                        "Esto no era un pájaro", variant="secondary",
                        elem_id="feedback-non-bird-btn",
                    )
                    feedback_cancel_btn = gr.Button(
                        "Volver", variant="secondary",
                        elem_id="feedback-cancel-btn",
                    )

            # Mensaje post-feedback ("¡Gracias!" o error). SIEMPRE montado
            # (value="" cuando no hay nada que decir): Gradio 6.14 no monta un
            # gr.Markdown visible=False mostrado por un handler de feedback
            # (verificado server-side — el handler devuelve un gr.update
            # valido pero el componente no aparece). Always-mounted lo evita;
            # los handlers solo cambian su `value`, nunca su `visible`.
            feedback_status = gr.Markdown(value="", elem_id="feedback-status")

            # Botones-puente ocultos (B1): el onclick de cada card del grid
            # clickea su boton fb-pick-<i>. gr.Button.click es 100% confiable.
            # Siempre montados (visible=True, ocultos por CSS .fb-hidden) — un
            # visible=False no estaria en el DOM (Gradio 6.14).
            fb_pick_btns = [
                gr.Button("", elem_id=f"fb-pick-{i}", elem_classes="fb-hidden")
                for i in range(len(_FEEDBACK_SPECIES))
            ]

            # Fase 4d.2d — indicador de cuota (escalating visibility). Oculto
            # por default; classify() lo actualiza via _format_quota_indicator.
            quota_indicator = gr.Markdown(
                value="", visible=False, elem_id="quota-indicator"
            )

            # ---- State machine wiring (Fase 2 - Nivel 1 UX) -----------------
            # IDLE inicial: el boton ya arranca con interactive=False y el hint
            # con _HINT_IDLE. No hace falta evento de "load" para setearlo.

            # IDLE/READY -> RECORDING: arranca grabacion de mic.
            audio_in.start_recording(
                fn=_state_recording,
                inputs=None,
                outputs=[submit, status_hint],
            )

            # Cubre: upload completo, stop_recording (con audio listo), y clear.
            # Por eso NO wireamos stop_recording por separado — change fires
            # despues con el value materializado, evitando race.
            audio_in.change(
                fn=_on_audio_change,
                inputs=audio_in,
                outputs=[submit, status_hint],
            )

            # READY -> PROCESSING -> (READY|IDLE). El primer .then() bloquea el
            # boton ANTES de que classify arranque, asi un doble-click se
            # serializa visualmente (no solo en queue de Gradio). El segundo
            # .then() recalcula el estado mirando el audio actual.
            submit.click(
                fn=_state_processing,
                inputs=None,
                outputs=[submit, status_hint],
            ).then(
                fn=classify,
                inputs=[audio_in, fingerprint_state, consent_checkbox],
                outputs=[
                    result_group, error_box, bird_photo, bird_info,
                    label_out, quota_indicator,
                    prediction_id_state, feedback_buttons_group,
                    feedback_correction_group, feedback_status, feedback_grid,
                ],
            ).then(
                fn=_on_audio_change,
                inputs=audio_in,
                outputs=[submit, status_hint],
            )

            # ---- Feedback wiring (Fase 4d.3) --------------------------------
            feedback_thumbs_up.click(
                fn=_on_thumbs_up,
                inputs=[prediction_id_state, fingerprint_state],
                outputs=[feedback_buttons_group, feedback_status],
            )
            feedback_thumbs_down.click(
                fn=_on_thumbs_down,
                inputs=None,
                outputs=[feedback_buttons_group, feedback_correction_group],
            )
            # Cada boton-puente -> _on_pick_species con su especie fijada por
            # closure (sp=_pick_sp captura el valor en el momento del loop).
            for _pick_btn, _pick_sp in zip(fb_pick_btns, _FEEDBACK_SPECIES):
                _pick_btn.click(
                    fn=lambda pid, fp, sp=_pick_sp: _on_pick_species(sp, pid, fp),
                    inputs=[prediction_id_state, fingerprint_state],
                    outputs=[feedback_correction_group, feedback_status],
                )
            feedback_non_bird_btn.click(
                fn=_on_non_bird,
                inputs=[prediction_id_state, fingerprint_state],
                outputs=[feedback_correction_group, feedback_status],
            )
            feedback_cancel_btn.click(
                fn=_on_cancel,
                inputs=None,
                outputs=[feedback_buttons_group, feedback_correction_group],
            )

        # ----- Tab 2: Las 20 aves -----
        with gr.Tab("Las 20 aves"):
            gr.Markdown(
                "Las 20 especies que el modelo aprendio a reconocer."
            )
            # Toda la galeria en UN solo gr.HTML para que el HTML inicial sea
            # liviano y las fotos se carguen lazy (solo al abrir el tab).
            gr.HTML(_render_species_gallery_html())

    # Al cargar la app: (1) dispara warm fire-and-forget al Lambda (Fase 5A.2)
    # y (2) reusa/regenera el fingerprint. demo.load corre per-sesion del lado
    # del Space, despues de que BrowserState hidrata desde localStorage.
    #
    # El warm va en un thread daemon: kick + retorno inmediato. NO esperamos
    # respuesta — si el cold start del Lambda tarda 30s, no queremos retrasar
    # el render de la pagina. El usuario tipicamente tarda 5-15s en seleccionar
    # un audio y clickear, lo cual da tiempo al container a warmarse en
    # background. Si el usuario clickea instantaneo, paga el cold start de
    # todas formas (el thread del demo.load es additive, no bloqueante).
    def _on_demo_load(fp: str | None) -> str:
        threading.Thread(target=warm_lambda, daemon=True).start()
        return get_or_create_fingerprint(fp)

    demo.load(
        fn=_on_demo_load,
        inputs=[fingerprint_state],
        outputs=[fingerprint_state],
    )


if __name__ == "__main__":
    # allowed_paths habilita que el HTML del header (logo) y las fotos del
    # tab 2 se sirvan desde app/assets/.
    demo.launch(allowed_paths=[str(ASSETS_DIR)])
