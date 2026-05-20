"""Unit tests del handler de Lambda.

Cubre tres bloques:
- Funciones puras de detection: ``_sigmoid``, ``_evaluate_detection``,
  ``_classify_synthetic`` (Fase 2 - Nivel 2).
- ``/upload-url`` y validacion de ``s3_key`` en ``/predict`` (Fase 3).
- Orquestacion de ``_handle_predict`` con rate limit + writes a DynamoDB
  (Fase 4): las 9 fases, los 4 desenlaces, idempotency y fail-loud.

Importar ``handler`` dispara la carga de modelos (~5 s cold start). Los tests
de Fase 4 mockean ``_DDB_TABLE`` y el pipeline ML — no tocan AWS ni invocan
BirdNET/ONNX. Integration test con audio real corre aparte.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "lambda"))

import handler as h  # noqa: E402

# ===========================================================================
# Fase 4 — helpers y fixtures compartidos (rate limit + writes a DynamoDB)
# ===========================================================================
_VALID_FP = "fp_" + "a" * 32
_AUDIO_B64 = base64.b64encode(b"fake-audio-bytes").decode("ascii")

_TOP3 = [
    {"species": "Chauna torquata", "common_name": "Chajá",
     "common_name_en": "Southern Screamer", "confidence": 0.95},
    {"species": "Rhea americana", "common_name": "Ñandú",
     "common_name_en": "Greater Rhea", "confidence": 0.03},
    {"species": "Jabiru mycteria", "common_name": "Jabirú",
     "common_name_en": "Jabiru", "confidence": 0.02},
]


def _predict_event(**overrides):
    """Construye un event API GW para /predict.

    Por default incluye fingerprint + audio_b64 validos. Pasar un campo en
    None lo OMITE del body (para testear validacion); cualquier otro kwarg
    (top_k, s3_key, training_consent) se agrega tal cual.
    """
    body = {"fingerprint": _VALID_FP, "audio_b64": _AUDIO_B64}
    body.update(overrides)
    body = {k: v for k, v in body.items() if v is not None}
    return {"rawPath": "/predict", "body": json.dumps(body)}


@pytest.fixture
def ccfe_class():
    """ConditionalCheckFailedException como clase real.

    Un MagicMock no se puede usar en una clausula `except` — el handler hace
    `except _DDB_TABLE.meta.client.exceptions.ConditionalCheckFailedException`,
    asi que el mock necesita una clase de excepcion de verdad.
    """
    return type("ConditionalCheckFailedException", (Exception,), {})


@pytest.fixture
def mock_ddb(monkeypatch, ccfe_class):
    """Mockea _DDB_TABLE. Default: update_item OK (rate limit pasa, count=1),
    put_item OK. Cada test reconfigura .return_value / .side_effect."""
    mock = MagicMock()
    mock.meta.client.exceptions.ConditionalCheckFailedException = ccfe_class
    mock.update_item.return_value = {"Attributes": {"request_count": 1}}
    mock.put_item.return_value = {}
    monkeypatch.setattr(h, "_DDB_TABLE", mock)
    return mock


class _PipelineKnobs:
    """Setters para forzar el desenlace del pipeline ML mockeado."""

    def __init__(self, monkeypatch):
        self._mp = monkeypatch

    def synthetic(self, reason):
        """_classify_synthetic -> reason ('white_noise'/'pure_tone'/None)."""
        self._mp.setattr(h, "_classify_synthetic", lambda y: reason)

    def embed(self, max_conf, n_windows=1):
        """_embed -> (embedding zeros, [max_conf], n_windows).

        _evaluate_detection corre real: max_conf >= 0.10 define detected.
        """
        self._mp.setattr(
            h, "_embed",
            lambda y: (np.zeros(h.EMBEDDING_DIM, dtype=np.float32),
                       np.array([float(max_conf)]), n_windows),
        )

    def classify(self, predictions):
        """_classify -> lista de predicciones fija."""
        self._mp.setattr(h, "_classify", lambda emb, k: predictions)


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Mockea el pipeline ML. _load_audio_bytes -> zeros siempre; el desenlace
    se controla con los setters de _PipelineKnobs. No invoca BirdNET/ONNX."""
    monkeypatch.setattr(h, "_load_audio_bytes",
                        lambda b: np.zeros(h.SAMPLE_RATE, dtype=np.float32))
    return _PipelineKnobs(monkeypatch)


def _setup_desenlace(mp, name):
    """Configura mock_pipeline para producir uno de los 4 desenlaces."""
    if name in ("white_noise", "pure_tone"):
        mp.synthetic(name)
    elif name == "not_a_bird":
        mp.synthetic(None)
        mp.embed(max_conf=0.05, n_windows=3)
    elif name == "detected":
        mp.synthetic(None)
        mp.embed(max_conf=0.9, n_windows=2)
        mp.classify(_TOP3)
    else:
        raise ValueError(f"desenlace desconocido: {name}")


# ---------------------------------------------------------------------------
# _sigmoid: estabilidad y propiedades
# ---------------------------------------------------------------------------
class TestSigmoid:
    def test_zero(self) -> None:
        assert h._sigmoid(np.array([0.0])).item() == pytest.approx(0.5, abs=1e-9)

    def test_positive_saturates_near_one(self) -> None:
        out = h._sigmoid(np.array([30.0])).item()
        assert 0.99 < out <= 1.0

    def test_negative_saturates_near_zero(self) -> None:
        out = h._sigmoid(np.array([-30.0])).item()
        assert 0.0 <= out < 0.01

    def test_extreme_inputs_no_overflow(self) -> None:
        """np.exp(1000) en float32 overflea; en float64 da inf -> 0 o 1 sin warning."""
        with np.errstate(over="raise", invalid="raise"):
            out = h._sigmoid(np.array([-1000.0, 1000.0], dtype=np.float32))
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[1] == pytest.approx(1.0, abs=1e-9)

    def test_preserves_shape(self) -> None:
        arr = np.array([[1.0, -1.0], [0.0, 2.0]])
        out = h._sigmoid(arr)
        assert out.shape == arr.shape

    def test_float32_input_returns_float64(self) -> None:
        """Casteo interno a float64 para evitar overflow. Output sigue float64."""
        out = h._sigmoid(np.array([0.5], dtype=np.float32))
        assert out.dtype == np.float64

    def test_monotonic(self) -> None:
        """Sanity: sigmoid es monotonicamente creciente."""
        xs = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        ys = h._sigmoid(xs)
        assert np.all(np.diff(ys) > 0)


# ---------------------------------------------------------------------------
# _evaluate_detection: MAX aggregation + threshold boundary
# ---------------------------------------------------------------------------
class TestEvaluateDetection:
    def test_all_above_threshold(self) -> None:
        max_conf = np.array([0.6, 0.7, 0.8])
        confidence, detected = h._evaluate_detection(max_conf)
        assert detected is True
        assert confidence == pytest.approx(0.8)

    def test_all_below_threshold(self) -> None:
        max_conf = np.array([0.01, 0.02, 0.03])
        confidence, detected = h._evaluate_detection(max_conf)
        assert detected is False
        assert confidence == pytest.approx(0.03)

    def test_mixed_max_above_passes(self) -> None:
        """MAX aggregation: una ventana arriba alcanza para detectar."""
        max_conf = np.array([0.01, 0.02, 0.9, 0.005])
        confidence, detected = h._evaluate_detection(max_conf)
        assert detected is True
        assert confidence == pytest.approx(0.9)

    def test_exactly_at_threshold_accepts(self) -> None:
        """Threshold >= 0.10: 0.10 exacto pasa (no es < 0.10)."""
        max_conf = np.array([0.10])
        confidence, detected = h._evaluate_detection(max_conf)
        assert detected is True
        assert confidence == pytest.approx(0.10)

    def test_just_below_threshold_rejects(self) -> None:
        max_conf = np.array([0.099])
        confidence, detected = h._evaluate_detection(max_conf)
        assert detected is False
        assert confidence == pytest.approx(0.099)

    def test_single_window(self) -> None:
        """Audio < 3 s -> 1 ventana. Comportamiento debe ser consistente."""
        confidence, detected = h._evaluate_detection(np.array([0.75]))
        assert detected is True
        assert confidence == pytest.approx(0.75)

    def test_returns_python_float(self) -> None:
        """confidence debe ser builtin float (serializable a JSON sin .item())."""
        confidence, _ = h._evaluate_detection(np.array([0.42]))
        assert isinstance(confidence, float)
        assert not isinstance(confidence, np.floating)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
def test_threshold_is_0_10() -> None:
    """Sanity: que no nos cambien el threshold por accidente sin discutirlo.

    0.10 (no 0.5 de Wood & Kahl 2024) porque nuestro caso de uso es app
    interactiva donde el usuario YA dijo "esto es un ave" — FN es peor que
    FP. Tuneo empirico en ADR D11.
    """
    assert h.BIRDNET_DETECTION_THRESHOLD == 0.10


def test_non_bird_indices_population() -> None:
    """18 clases non-bird (12 meta + 6 ranas). Si BirdNET actualiza el
    catalogo, regenerar la lista — ver scripts/benchmark_baseline_vs_finetuned.py.
    """
    assert len(h._NON_BIRD_INDICES) == 18
    # Indices verificados contra V2.4 GLOBAL 6K (6522 labels).
    assert 2143 in h._NON_BIRD_INDICES  # Engine
    assert 3927 in h._NON_BIRD_INDICES  # Noise
    assert 2818 in h._NON_BIRD_INDICES  # Human non-vocal
    assert all(0 <= i < 6522 for i in h._NON_BIRD_INDICES)


# ---------------------------------------------------------------------------
# _classify_synthetic: pre-filter espectral (white_noise / pure_tone / None)
# Thresholds derivados en scripts/measure_flatness.py (ADR D11).
# ---------------------------------------------------------------------------
class TestClassifySynthetic:
    SR = 48_000
    DUR_S = 5.0
    N = int(SR * DUR_S)

    def test_white_noise_detected(self) -> None:
        """White noise estable -> flatness alta uniformemente + rms suficiente."""
        rng = np.random.default_rng(seed=42)
        y = rng.normal(0, 0.05, self.N).astype(np.float32)
        assert h._classify_synthetic(y) == "white_noise"

    def test_pure_tone_detected(self) -> None:
        """Tono 440 Hz -> flatness ~0 + bandwidth pequeño."""
        t = np.arange(self.N) / self.SR
        y = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        assert h._classify_synthetic(y) == "pure_tone"

    def test_silence_not_classified_as_synthetic(self) -> None:
        """Silencio puro: flatness alta por div-cero, PERO rms~0 -> no es white_noise.
        Tampoco tono puro (flat_mean ~ 1.0, no < 0.01). Devuelve None y deja
        que el gate de BirdNET lo rechace como not_a_bird.
        """
        y = np.zeros(self.N, dtype=np.float32)
        assert h._classify_synthetic(y) is None

    def test_bird_like_chirp_not_classified_as_synthetic(self) -> None:
        """Senal tipo-canto: chirp 800->3000 Hz con harmonics.
        Flatness intermedia + bandwidth amplio -> debe pasar (None).
        """
        t = np.arange(self.N) / self.SR
        f_inst = 800 + (3000 - 800) * t / self.DUR_S
        phase = 2 * np.pi * np.cumsum(f_inst) / self.SR
        y = 0.3 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
        y = y.astype(np.float32)
        assert h._classify_synthetic(y) is None


# ---------------------------------------------------------------------------
# Spectral thresholds — sanity de constantes (no se cambian sin discutirlo)
# ---------------------------------------------------------------------------
def test_spectral_thresholds_pinned() -> None:
    """Thresholds derivados empiricamente de scripts/measure_flatness.py.
    Cambiarlos requiere re-medir y actualizar ADR D11.
    """
    assert h.FLATNESS_WHITE_NOISE_P95 == 0.30
    assert h.WHITE_NOISE_MIN_RMS == 0.001
    assert h.FLATNESS_PURE_TONE_MEAN == 0.01
    assert h.BANDWIDTH_PURE_TONE_HZ == 1000.0


# ---------------------------------------------------------------------------
# Fase 3: /upload-url handler — presigned PUT URL generation
# ---------------------------------------------------------------------------
class TestUploadUrl:
    """Tests del nuevo handler /upload-url. Mockeamos _S3_CLIENT para no
    necesitar AWS real; verificamos response shape, validacion estricta de
    ext (lower + strip + whitelist), y que generate_presigned_url se
    invoca con ContentType explicito (defense in depth)."""

    @pytest.fixture
    def mock_s3(self, monkeypatch):
        mock = MagicMock()
        mock.generate_presigned_url.return_value = "https://s3.test/url-placeholder"
        monkeypatch.setattr(h, "_S3_CLIENT", mock)
        return mock

    def test_handle_upload_url_ok_mp3(self, mock_s3):
        """Happy path: ext=mp3 -> 200 con URL + s3_key UUID + ContentType correcto."""
        event = {"rawPath": "/upload-url", "body": '{"ext": "mp3"}'}
        resp = h.handler(event)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["upload_url"] == "https://s3.test/url-placeholder"
        assert body["s3_key"].startswith("uploads/")
        assert body["s3_key"].endswith(".mp3")
        assert body["content_type"] == "audio/mpeg"
        assert body["expires_in"] == 600

        # generate_presigned_url debe recibir ContentType explicito + bucket + key.
        mock_s3.generate_presigned_url.assert_called_once()
        call = mock_s3.generate_presigned_url.call_args
        assert call.kwargs["ClientMethod"] == "put_object"
        assert call.kwargs["HttpMethod"] == "PUT"
        assert call.kwargs["Params"]["ContentType"] == "audio/mpeg"
        assert call.kwargs["Params"]["Bucket"] == h.S3_UPLOADS_BUCKET

    def test_handle_upload_url_invalid_ext(self, mock_s3):
        """ext no whitelisted (exe) -> 400, NO toca S3."""
        event = {"rawPath": "/upload-url", "body": '{"ext": "exe"}'}
        resp = h.handler(event)

        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "exe" in body["error"]
        mock_s3.generate_presigned_url.assert_not_called()

    def test_handle_upload_url_missing_ext(self, mock_s3):
        """body sin ext -> 400."""
        event = {"rawPath": "/upload-url", "body": "{}"}
        resp = h.handler(event)

        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "ext" in body["error"].lower()
        mock_s3.generate_presigned_url.assert_not_called()

    def test_handle_upload_url_all_formats(self, mock_s3):
        """mp3/wav/ogg/flac todos OK con su MIME correcto. Tambien valida
        normalizacion: 'WAV' / ' mp3 ' deben aceptarse (lower + strip)."""
        cases = [
            ("mp3", "audio/mpeg"),
            ("wav", "audio/wav"),
            ("ogg", "audio/ogg"),
            ("flac", "audio/flac"),
            ("WAV", "audio/wav"),       # normalizacion lower
            (" mp3 ", "audio/mpeg"),    # normalizacion strip
        ]
        for ext_in, expected_mime in cases:
            event = {"rawPath": "/upload-url", "body": json.dumps({"ext": ext_in})}
            resp = h.handler(event)
            assert resp["statusCode"] == 200, f"falló para ext={ext_in!r}"
            body = json.loads(resp["body"])
            assert body["content_type"] == expected_mime, (
                f"ext={ext_in!r} dio MIME {body['content_type']}, esperaba {expected_mime}"
            )


# ---------------------------------------------------------------------------
# Fase 3: /predict con s3_key — validacion estricta de key
# ---------------------------------------------------------------------------
class TestS3KeyValidation:
    """Tests del path /predict con s3_key. Verifican que keys invalidos NO
    llegan a tocar S3 (defense in depth contra deserializacion no-segura).

    Tres clases de invalidez cubiertas:
        - path traversal (../etc/passwd)
        - prefix wrong (other/file.mp3)
        - formato wrong (uploads/random_name.mp3 — no es UUID4)
    """

    @pytest.fixture
    def mock_s3(self, monkeypatch):
        mock = MagicMock()
        # NoSuchKey es exception del client S3. Mockeamos como clase real.
        mock.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        monkeypatch.setattr(h, "_S3_CLIENT", mock)
        return mock

    def test_s3_key_path_traversal_rejected(self, mock_s3, mock_ddb):
        """s3_key con '..' -> 400 sin tocar S3 (Fase 4 valida el key)."""
        event = {"rawPath": "/predict",
                 "body": json.dumps({"fingerprint": _VALID_FP,
                                      "s3_key": "../etc/passwd"})}
        resp = h.handler(event)

        assert resp["statusCode"] == 400
        mock_s3.get_object.assert_not_called()

    def test_s3_key_wrong_prefix_rejected(self, mock_s3, mock_ddb):
        """s3_key con prefix distinto a uploads/ -> 400 sin tocar S3."""
        event = {"rawPath": "/predict",
                 "body": json.dumps({"fingerprint": _VALID_FP,
                                      "s3_key": "other/file.mp3"})}
        resp = h.handler(event)

        assert resp["statusCode"] == 400
        mock_s3.get_object.assert_not_called()

    def test_s3_key_invalid_format_rejected(self, mock_s3, mock_ddb):
        """s3_key con prefix OK pero NOT UUID4 -> 400 sin tocar S3.

        Defense in depth: solo aceptamos keys que NOSOTROS generamos.
        Si alguien intenta uploads/algo_random.mp3 que por casualidad
        existe, se rechaza ANTES del S3 call."""
        event = {
            "rawPath": "/predict",
            "body": json.dumps({"fingerprint": _VALID_FP,
                                "s3_key": "uploads/not_a_uuid.mp3"}),
        }
        resp = h.handler(event)

        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "formato" in body["error"].lower()
        mock_s3.get_object.assert_not_called()

    def test_s3_key_valid_format_routes_to_s3(self, mock_s3, mock_ddb, monkeypatch):
        """s3_key con formato valido -> SI llama a S3 con bucket+key correctos.

        Tomamos camino corto: synthetic detector devuelve early para no
        invocar BirdNET/ONNX en este unit test. Igual valida que el
        pipeline llega hasta esa capa con el audio leido de S3.
        """
        # S3 devuelve bytes (no importa el contenido, _load_audio_bytes esta mockeado).
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"fake_bytes")}
        # Mock decode para no requerir audio real decodable.
        monkeypatch.setattr(h, "_load_audio_bytes",
                            lambda b: np.zeros(48_000, dtype=np.float32))
        # Mock synthetic detector -> early return ok (saltea BirdNET/ONNX).
        monkeypatch.setattr(h, "_classify_synthetic", lambda y: "white_noise")

        valid_key = "uploads/12345678-1234-1234-1234-123456789abc.mp3"
        event = {"rawPath": "/predict",
                 "body": json.dumps({"fingerprint": _VALID_FP, "s3_key": valid_key})}
        resp = h.handler(event)

        # S3 invocado con bucket + key exactos.
        mock_s3.get_object.assert_called_once_with(
            Bucket=h.S3_UPLOADS_BUCKET, Key=valid_key
        )
        # Pipeline corto: synthetic detected -> 200 con detected=false.
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["detected"] is False
        assert body["reason"] == "white_noise"


# ---------------------------------------------------------------------------
# Fase 4: _handle_predict — 9 fases, rate limit, writes a DynamoDB
# ---------------------------------------------------------------------------
class TestPredictFase4:
    """Orquestacion de _handle_predict tras el refactor de Fase 4 (Step 2.e).

    _DDB_TABLE y el pipeline ML estan mockeados (fixtures mock_ddb /
    mock_pipeline) — cero AWS, cero BirdNET/ONNX. _check_and_increment_rate_limit
    y _write_prediction_item corren REALES contra el _DDB_TABLE mockeado.
    """

    # --- Fase 1: validacion de input ---------------------------------------
    def test_fingerprint_faltante_devuelve_400(self, mock_ddb):
        """Body sin fingerprint -> 400; rate limit y pipeline intactos."""
        resp = h.handler(_predict_event(fingerprint=None))
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "fingerprint" in body["error"]
        assert "rate_info" not in body
        assert "prediction_id" not in body
        mock_ddb.update_item.assert_not_called()
        mock_ddb.put_item.assert_not_called()

    @pytest.mark.parametrize("bad_fp", [
        "fp_xyz",                # muy corto
        "fp_" + "a" * 31,        # 31 hex
        "fp_" + "a" * 33,        # 33 hex
        "fp_" + "A" * 32,        # hex en mayuscula
        "FP_" + "a" * 32,        # prefijo mayuscula
        "abc123",                # sin prefijo fp_
        "",                      # vacio
    ])
    def test_fingerprint_malformado_devuelve_400(self, mock_ddb, bad_fp):
        """fingerprint que no matchea ^fp_[0-9a-f]{32}$ -> 400."""
        resp = h.handler(_predict_event(fingerprint=bad_fp))
        assert resp["statusCode"] == 400
        assert "fingerprint" in json.loads(resp["body"])["error"]
        mock_ddb.update_item.assert_not_called()

    @pytest.mark.parametrize("kwargs,substr", [
        ({"s3_key": "uploads/abc.mp3"}, "ambos"),   # audio_b64 + s3_key
        ({"audio_b64": None}, "audio_b64"),         # ninguno
    ])
    def test_audio_source_xor_devuelve_400(self, mock_ddb, kwargs, substr):
        """Audio source XOR: ambos o ninguno -> 400 con mensajes distintos."""
        resp = h.handler(_predict_event(**kwargs))
        assert resp["statusCode"] == 400
        assert substr in json.loads(resp["body"])["error"]
        mock_ddb.update_item.assert_not_called()

    @pytest.mark.parametrize("top_k_in,expect", [
        ("abc", 400),      # no-int -> 400
        (None, 3),         # ausente -> DEFAULT_TOP_K
        (5, 5),            # valido -> tal cual
        (9999, "clamp"),   # oversized -> clamp a _NUM_CLASSES
    ])
    def test_top_k_validation(self, mock_ddb, mock_pipeline, monkeypatch,
                              top_k_in, expect):
        """top_k: invalido 400, ausente usa default, oversized se clampea."""
        classify_spy = MagicMock(return_value=_TOP3)
        monkeypatch.setattr(h, "_classify", classify_spy)
        mock_pipeline.synthetic(None)
        mock_pipeline.embed(max_conf=0.9)  # detected -> se invoca _classify
        resp = h.handler(_predict_event(top_k=top_k_in))
        if expect == 400:
            assert resp["statusCode"] == 400
            assert "top_k" in json.loads(resp["body"])["error"]
            classify_spy.assert_not_called()
        else:
            assert resp["statusCode"] == 200
            called_k = classify_spy.call_args[0][1]  # _classify(embedding, top_k)
            expected_k = h._NUM_CLASSES if expect == "clamp" else expect
            assert called_k == expected_k

    # --- Fase 2: rate limit ------------------------------------------------
    def test_rate_limit_ok_procede_al_pipeline(self, mock_ddb, mock_pipeline):
        """update_item OK -> request procede; rate_info refleja el contador."""
        mock_ddb.update_item.return_value = {"Attributes": {"request_count": 7}}
        mock_pipeline.synthetic("white_noise")
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        rate_info = json.loads(resp["body"])["rate_info"]
        assert rate_info["requests_today"] == 7
        assert rate_info["limit"] == h.DAILY_REQUEST_LIMIT
        assert rate_info["remaining"] == h.DAILY_REQUEST_LIMIT - 7
        mock_ddb.update_item.assert_called_once()

    def test_rate_limit_excedido_devuelve_429(self, mock_ddb, ccfe_class):
        """update_item raises ConditionalCheckFailed -> 429 con rate_info,
        SIN prediction_id, sin escribir item PREDICTION."""
        exc = ccfe_class("limite excedido")
        exc.response = {"Item": {"request_count": h.DAILY_REQUEST_LIMIT}}
        mock_ddb.update_item.side_effect = exc
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 429
        body = json.loads(resp["body"])
        assert body["rate_info"]["remaining"] == 0
        assert "reset_at" in body["rate_info"]
        assert "prediction_id" not in body
        mock_ddb.put_item.assert_not_called()

    # --- Fase 8: los 4 desenlaces escriben item PREDICTION -----------------
    @pytest.mark.parametrize("reason", ["white_noise", "pure_tone"])
    def test_desenlace_synthetic_reject_escribe_item(self, mock_ddb,
                                                     mock_pipeline, reason):
        """Reject del pre-filter sintetico -> 200 detected=false + item con
        result_status=reason, sin top1_species/n_windows/max_birdnet_confidence."""
        mock_pipeline.synthetic(reason)
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["detected"] is False
        assert body["reason"] == reason
        mock_ddb.put_item.assert_called_once()
        item = mock_ddb.put_item.call_args.kwargs["Item"]
        assert item["item_type"] == "PREDICTION"
        assert item["result_status"] == reason
        assert "top1_species" not in item
        assert "n_windows" not in item
        assert "max_birdnet_confidence" not in item

    def test_desenlace_not_a_bird_escribe_item(self, mock_ddb, mock_pipeline):
        """Gate BirdNET no detecta ave -> 200 reason=not_a_bird + item con
        max_birdnet_confidence + n_windows, sin top1_species."""
        mock_pipeline.synthetic(None)
        mock_pipeline.embed(max_conf=0.05, n_windows=3)  # 0.05 < 0.10
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["detected"] is False
        assert body["reason"] == "not_a_bird"
        item = mock_ddb.put_item.call_args.kwargs["Item"]
        assert item["result_status"] == "not_a_bird"
        assert item["n_windows"] == 3
        assert float(item["max_birdnet_confidence"]) == pytest.approx(0.05, abs=1e-4)
        assert "top1_species" not in item

    def test_desenlace_detected_escribe_item_con_top3(self, mock_ddb,
                                                      mock_pipeline):
        """Happy path -> 200 con predictions + item result_status=detected
        con top1_species/top1_confidence/top3_predictions."""
        mock_pipeline.synthetic(None)
        mock_pipeline.embed(max_conf=0.9, n_windows=2)
        mock_pipeline.classify(_TOP3)
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["predictions"] == _TOP3
        item = mock_ddb.put_item.call_args.kwargs["Item"]
        assert item["result_status"] == "detected"
        assert item["top1_species"] == "Chauna torquata"
        assert float(item["top1_confidence"]) == pytest.approx(0.95, abs=1e-4)
        assert len(item["top3_predictions"]) == 3

    # --- Fase 8: resiliencia del write (fail-loud) -------------------------
    def test_write_idempotente_no_rompe_response(self, mock_ddb, mock_pipeline,
                                                 ccfe_class):
        """put_item raises ConditionalCheckFailed (item ya existe) ->
        response sigue 200, el handler no propaga la excepcion."""
        mock_pipeline.synthetic("white_noise")
        mock_ddb.put_item.side_effect = ccfe_class("item ya existe")
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        assert json.loads(resp["body"])["reason"] == "white_noise"

    def test_write_fail_generico_no_rompe_response(self, mock_ddb,
                                                   mock_pipeline, caplog):
        """put_item raises excepcion generica -> response sigue 200 (fail-loud
        no rompe UX) Y se loguea a ERROR (blinda la observabilidad)."""
        mock_pipeline.synthetic("white_noise")
        mock_ddb.put_item.side_effect = RuntimeError("simulated DDB failure")
        with caplog.at_level(logging.ERROR):
            resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        assert json.loads(resp["body"])["reason"] == "white_noise"
        assert any(
            rec.levelname == "ERROR"
            and "PutItem PREDICTION fallo" in rec.getMessage()
            for rec in caplog.records
        )

    # --- Fase 9: shape de respuesta ----------------------------------------
    @pytest.mark.parametrize("desenlace", ["white_noise", "not_a_bird", "detected"])
    def test_prediction_id_en_responses_post_fase3(self, mock_ddb,
                                                   mock_pipeline, desenlace):
        """Toda respuesta de desenlace (200) incluye prediction_id UUID4,
        identico al prediction_id del item PREDICTION escrito."""
        _setup_desenlace(mock_pipeline, desenlace)
        resp = h.handler(_predict_event())
        assert resp["statusCode"] == 200
        pid = json.loads(resp["body"])["prediction_id"]
        assert uuid.UUID(pid).version == 4
        item = mock_ddb.put_item.call_args.kwargs["Item"]
        assert item["prediction_id"] == pid
        assert item["pk"] == f"PRED#{pid}"

    @pytest.mark.parametrize(
        "desenlace", ["white_noise", "not_a_bird", "detected", "rate_limited"])
    def test_rate_info_en_responses_post_fase2(self, mock_ddb, mock_pipeline,
                                               ccfe_class, desenlace):
        """rate_info presente en los 3 desenlaces (200) y en el 429.
        El 429 lleva rate_info pero NO prediction_id."""
        if desenlace == "rate_limited":
            exc = ccfe_class("excedido")
            exc.response = {"Item": {"request_count": h.DAILY_REQUEST_LIMIT}}
            mock_ddb.update_item.side_effect = exc
        else:
            _setup_desenlace(mock_pipeline, desenlace)
        resp = h.handler(_predict_event())
        body = json.loads(resp["body"])
        assert "rate_info" in body
        if desenlace == "rate_limited":
            assert resp["statusCode"] == 429
            assert "prediction_id" not in body
        else:
            assert resp["statusCode"] == 200
            assert "prediction_id" in body

    # --- Consent -----------------------------------------------------------
    @pytest.mark.parametrize("consent_in,expected", [
        (True, True),
        (False, False),
        (None, False),   # ausente -> default False
    ])
    def test_training_consent_propagado_al_item(self, mock_ddb, mock_pipeline,
                                                consent_in, expected):
        """training_consent del body se propaga al item; ausente -> False."""
        mock_pipeline.synthetic("white_noise")
        resp = h.handler(_predict_event(training_consent=consent_in))
        assert resp["statusCode"] == 200
        item = mock_ddb.put_item.call_args.kwargs["Item"]
        assert item["training_consent"] is expected
