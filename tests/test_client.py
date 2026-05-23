"""Tests del cliente HTTP del Lambda (Fase 3 — S3 presigned URLs).

Cubre:
- ``_derive_upload_url_url`` (helper sin red)
- ``_get_upload_url``: happy + errores HTTP
- ``_upload_to_s3``: happy + errores HTTP/network
- ``predict``: flow completo de 3 pasos + fallos en cada paso

Mocks de ``requests`` para no hacer HTTP real. ``_get_session`` se patcha
para inyectar un MagicMock con ``.post()`` configurado.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "app"))

import client as c  # noqa: E402

# Datos compartidos para los tests de Fase 4.
_FP = "fp_" + "a" * 32                              # fingerprint valido
_PRED_ID = "12345678-1234-4234-8234-123456789abc"   # UUID4 canonico


def _mock_response(status: int, json_body: dict | None = None) -> MagicMock:
    """Helper: crea un mock de requests.Response con status + json()."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = json_body if json_body is not None else {}
    return mock


# ---------------------------------------------------------------------------
# _derive_upload_url_url
# ---------------------------------------------------------------------------
class TestDeriveUploadUrl:
    def test_predict_url_replaced(self) -> None:
        url = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
        assert c._derive_upload_url_url(url) == (
            "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/upload-url"
        )

    def test_fallback_when_no_predict_suffix(self) -> None:
        """URL sin /predict al final: append /upload-url al base sin crashear."""
        assert c._derive_upload_url_url("https://example.com/api") == (
            "https://example.com/api/upload-url"
        )


# ---------------------------------------------------------------------------
# _get_upload_url
# ---------------------------------------------------------------------------
class TestGetUploadUrl:
    URL = "https://api.test/prod/predict"

    @patch.object(c, "_get_session")
    def test_ok(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(200, {
            "upload_url": "https://s3.amazonaws.com/presigned-url",
            "s3_key": "uploads/abc-uuid.mp3",
            "expires_in": 600,
            "content_type": "audio/mpeg",
        })
        mock_session_factory.return_value = mock_session

        result = c._get_upload_url("mp3", api_url=self.URL)

        assert result.ok is True
        assert result.upload_url == "https://s3.amazonaws.com/presigned-url"
        assert result.s3_key == "uploads/abc-uuid.mp3"
        assert result.content_type == "audio/mpeg"

        # URL llamada es la DERIVADA (/upload-url), no la /predict.
        mock_session.post.assert_called_once()
        call = mock_session.post.call_args
        assert call.args[0] == "https://api.test/prod/upload-url"
        assert call.kwargs["json"] == {"ext": "mp3"}

    @patch.object(c, "_get_session")
    def test_bad_request_400(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(
            400, {"error": "extension no soportada: 'exe'"},
        )
        mock_session_factory.return_value = mock_session

        result = c._get_upload_url("exe", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "bad_request"
        assert "exe" in result.error_message

    @patch.object(c, "_get_session")
    def test_throttled_503(self, mock_session_factory: MagicMock) -> None:
        """Throttling del HTTP API devuelve 503 (verificado Fase 1)."""
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(503, {})
        mock_session_factory.return_value = mock_session

        result = c._get_upload_url("mp3", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "throttled"


# ---------------------------------------------------------------------------
# _upload_to_s3
# ---------------------------------------------------------------------------
class TestUploadToS3:
    @patch("client.requests.put")
    def test_ok_200(self, mock_put: MagicMock) -> None:
        mock_put.return_value = _mock_response(200)

        ok, err = c._upload_to_s3(
            "https://s3.test/url", b"audio_bytes", "audio/mpeg",
        )

        assert ok is True
        assert err is None
        # Content-Type explicito en headers (S3 valida match con presigned).
        call = mock_put.call_args
        assert call.kwargs["headers"]["Content-Type"] == "audio/mpeg"
        assert call.kwargs["data"] == b"audio_bytes"

    @patch("client.requests.put")
    def test_403_signature_mismatch(self, mock_put: MagicMock) -> None:
        """S3 devuelve 403 si Content-Type del PUT no matchea el del presigned."""
        mock_put.return_value = _mock_response(403)

        ok, err = c._upload_to_s3(
            "https://s3.test/url", b"bytes", "audio/mpeg",
        )

        assert ok is False
        assert err is not None
        assert "403" in err

    @patch("client.requests.put")
    def test_timeout(self, mock_put: MagicMock) -> None:
        mock_put.side_effect = requests.exceptions.Timeout()

        ok, err = c._upload_to_s3(
            "https://s3.test/url", b"bytes", "audio/mpeg",
        )

        assert ok is False
        assert err is not None
        assert "timeout" in err.lower()


# ---------------------------------------------------------------------------
# predict (full flow — 3 pasos)
# ---------------------------------------------------------------------------
class TestPredictFullFlow:
    URL = "https://api.test/prod/predict"

    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_happy_path(
        self, mock_session_factory: MagicMock, mock_put: MagicMock, tmp_path: Path,
    ) -> None:
        """Flow completo OK: upload-url -> S3 PUT -> predict -> PredictResult."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake_mp3_bytes")

        # Step 2 (POST /upload-url): 200 con presigned.
        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/presigned",
            "s3_key": "uploads/abc-uuid.mp3",
            "content_type": "audio/mpeg",
        })
        # Step 4 (POST /predict): 200 con predictions.
        predict_resp = _mock_response(200, {
            "predictions": [
                {"species": "Test bird", "common_name": "Pajaro test", "confidence": 0.95},
            ],
            "model_version": "classifier_v1",
            "n_windows": 2,
            "inference_time_ms": 350.5,
            "max_birdnet_confidence": 0.8,
        })

        mock_session = MagicMock()
        # POSTs en orden: upload-url primero, predict segundo.
        mock_session.post.side_effect = [upload_resp, predict_resp]
        mock_session_factory.return_value = mock_session

        # Step 3 (PUT a S3): 200.
        mock_put.return_value = _mock_response(200)

        result = c.predict(
            str(audio_file), top_k=3, fingerprint=_FP, api_url=self.URL,
        )

        assert result.ok is True
        assert result.detected is True
        assert len(result.predictions) == 1
        assert result.predictions[0]["species"] == "Test bird"

        # 2 POSTs al session (upload-url + predict), 1 PUT a S3.
        assert mock_session.post.call_count == 2
        assert mock_put.call_count == 1
        # Body del segundo POST: s3_key + top_k + fingerprint + training_consent.
        second_call = mock_session.post.call_args_list[1]
        assert second_call.kwargs["json"] == {
            "s3_key": "uploads/abc-uuid.mp3", "top_k": 3,
            "fingerprint": _FP, "training_consent": False,
        }

    @patch.object(c, "_get_session")
    def test_fails_at_upload_url(
        self, mock_session_factory: MagicMock, tmp_path: Path,
    ) -> None:
        """Si /upload-url falla, predict() propaga el error_kind sin tocar S3."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake_bytes")

        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(503)
        mock_session_factory.return_value = mock_session

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "throttled"
        # Solo 1 POST (upload-url falló, no se llegó al predict).
        assert mock_session.post.call_count == 1

    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_fails_at_s3_put(
        self, mock_session_factory: MagicMock, mock_put: MagicMock, tmp_path: Path,
    ) -> None:
        """Si PUT a S3 falla, error_kind=upload_failed y NO se llega al /predict."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/url",
            "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })
        mock_session = MagicMock()
        mock_session.post.return_value = upload_resp
        mock_session_factory.return_value = mock_session

        # PUT a S3 falla con 403.
        mock_put.return_value = _mock_response(403)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "upload_failed"
        # Solo 1 POST al session (upload-url); el predict no se llamó.
        assert mock_session.post.call_count == 1

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_fails_at_predict_lambda(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """Steps 1+2+3 OK, paso final /predict devuelve 500 persistente.

        Fase 5A.1: 500 es retryable, así que el client reintenta 1 vez. Si
        ambos intentos fallan con 500, el error final es 'server'.
        """
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/url",
            "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })
        # 2 intentos al /predict, ambos 500 (lambda persistentemente roto).
        predict_resp_1 = _mock_response(500, {"error": "lambda boom"})
        predict_resp_2 = _mock_response(500, {"error": "lambda boom"})

        mock_session = MagicMock()
        mock_session.post.side_effect = [upload_resp, predict_resp_1, predict_resp_2]
        mock_session_factory.return_value = mock_session

        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "server"
        # 3 POSTs: upload-url + predict-fail-1 + predict-retry-fail-2.
        assert mock_session.post.call_count == 3
        assert mock_put.call_count == 1
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Fase 4 — RETRY_STATUS, rate limit, prediction_id/rate_info
# ---------------------------------------------------------------------------
class TestRetryConfig:
    def test_session_no_reintenta_429(self) -> None:
        """429 NO esta en status_forcelist (rate limit terminal); 503 si."""
        assert c.RETRY_STATUS == (503,)
        assert 429 not in c.RETRY_STATUS
        # El Retry real del adapter refleja la config.
        retry = c._build_session().get_adapter("https://x").max_retries
        assert 429 not in retry.status_forcelist
        assert 503 in retry.status_forcelist


# Fase 5A.1 — cold-start retry del cliente. El segundo intento de POST /predict
# se dispara cuando el primero falla con error transitorio (cold start de
# Lambda cortado por el cap 30s del API GW HTTP API).
class TestColdStartRetry:
    URL = "https://api.test/prod/predict"

    def _upload_resp(self) -> MagicMock:
        return _mock_response(200, {
            "upload_url": "https://s3.test/url", "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })

    def _predict_ok_resp(self) -> MagicMock:
        return _mock_response(200, {
            "predictions": [{"species": "Test bird", "confidence": 0.9}],
            "prediction_id": _PRED_ID,
            "rate_info": {"requests_today": 1, "limit": 30, "remaining": 29},
        })

    @patch("client.time.sleep")  # acelera el test, evita el backoff de 2s real
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_retry_en_connection_error_recupera(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """Primer intento ConnectionError (API GW cierra abruptamente cold
        start) → retry con backoff → segundo intento OK."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        mock_session = MagicMock()
        # 3 POSTs en orden: upload-url, predict (fail), predict (retry ok).
        mock_session.post.side_effect = [
            self._upload_resp(),
            requests.exceptions.ConnectionError("connection reset"),
            self._predict_ok_resp(),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is True
        assert result.prediction_id == _PRED_ID
        # 3 POSTs (upload-url + predict-fail + predict-retry); 1 sleep.
        assert mock_session.post.call_count == 3
        mock_sleep.assert_called_once_with(c.COLD_START_RETRY_BACKOFF_S)
        # El reintento usa timeout extendido (60s read).
        retry_call = mock_session.post.call_args_list[2]
        assert retry_call.kwargs["timeout"] == c.COLD_START_RETRY_TIMEOUT

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_retry_en_timeout_recupera(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """Primer intento Timeout → retry → segundo intento OK."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        mock_session = MagicMock()
        mock_session.post.side_effect = [
            self._upload_resp(),
            requests.exceptions.Timeout("read timeout"),
            self._predict_ok_resp(),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is True
        assert mock_session.post.call_count == 3
        mock_sleep.assert_called_once()

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_retry_en_500_recupera(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """Primer intento HTTP 500 → retry → segundo intento OK."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        mock_session = MagicMock()
        mock_session.post.side_effect = [
            self._upload_resp(),
            _mock_response(500, {"error": "internal"}),
            self._predict_ok_resp(),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is True
        assert mock_session.post.call_count == 3

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_NO_retry_en_429(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """429 (rate_limited) es terminal: NO se reintenta."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        rate_info = {"requests_today": 30, "limit": 30, "remaining": 0}
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            self._upload_resp(),
            _mock_response(429, {"error": "limit", "rate_info": rate_info}),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "rate_limited"
        # Solo 2 POSTs (upload-url + predict), NO hubo retry, NO hubo sleep.
        assert mock_session.post.call_count == 2
        mock_sleep.assert_not_called()

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_NO_retry_en_400(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """400 (bad_request, fingerprint inválido por ej.) es terminal."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        mock_session = MagicMock()
        mock_session.post.side_effect = [
            self._upload_resp(),
            _mock_response(400, {"error": "fingerprint inválido"}),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "bad_request"
        assert mock_session.post.call_count == 2
        mock_sleep.assert_not_called()

    @patch("client.time.sleep")
    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_retry_tambien_falla_propaga_error(
        self, mock_session_factory: MagicMock, mock_put: MagicMock,
        mock_sleep: MagicMock, tmp_path: Path,
    ) -> None:
        """Si ambos intentos fallan con ConnectionError: error_kind=network."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        mock_session = MagicMock()
        mock_session.post.side_effect = [
            self._upload_resp(),
            requests.exceptions.ConnectionError("first fail"),
            requests.exceptions.ConnectionError("second fail"),
        ]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "network"
        # 3 POSTs (upload-url + 2 predict attempts), 1 sleep.
        assert mock_session.post.call_count == 3
        mock_sleep.assert_called_once()


class TestPredictRateLimitAndIds:
    URL = "https://api.test/prod/predict"

    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_predict_429_rate_limited(
        self, mock_session_factory: MagicMock, mock_put: MagicMock, tmp_path: Path,
    ) -> None:
        """/predict 429 -> error_kind=rate_limited + rate_info, sin reintentos."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")
        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/url", "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })
        rate_info = {"requests_today": 30, "limit": 30, "remaining": 0,
                     "reset_at": "2026-05-21T00:00:00Z"}
        predict_resp = _mock_response(429, {
            "error": "Llegaste al limite diario (30 audios).",
            "rate_info": rate_info,
        })
        mock_session = MagicMock()
        mock_session.post.side_effect = [upload_resp, predict_resp]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "rate_limited"
        assert result.rate_info == rate_info
        assert result.prediction_id is None
        # 2 POSTs (upload-url + predict); el predict NO se reintento.
        assert mock_session.post.call_count == 2

    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_predict_extrae_prediction_id_y_rate_info(
        self, mock_session_factory: MagicMock, mock_put: MagicMock, tmp_path: Path,
    ) -> None:
        """200 detected -> prediction_id + rate_info en el PredictResult."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")
        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/url", "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })
        predict_resp = _mock_response(200, {
            "predictions": [{"species": "Rhea americana", "confidence": 0.9}],
            "prediction_id": _PRED_ID,
            "rate_info": {"requests_today": 3, "limit": 30, "remaining": 27},
        })
        mock_session = MagicMock()
        mock_session.post.side_effect = [upload_resp, predict_resp]
        mock_session_factory.return_value = mock_session
        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), fingerprint=_FP, api_url=self.URL)

        assert result.ok is True
        assert result.prediction_id == _PRED_ID
        assert result.rate_info["remaining"] == 27


# ---------------------------------------------------------------------------
# Fase 4c — _derive_feedback_url + send_feedback
# ---------------------------------------------------------------------------
class TestDeriveFeedbackUrl:
    def test_predict_url_replaced(self) -> None:
        url = "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/predict"
        assert c._derive_feedback_url(url) == (
            "https://1jbbnu85e5.execute-api.us-east-1.amazonaws.com/prod/feedback"
        )

    def test_fallback_when_no_predict_suffix(self) -> None:
        assert c._derive_feedback_url("https://example.com/api") == (
            "https://example.com/api/feedback"
        )


class TestSendFeedback:
    URL = "https://api.test/prod/predict"

    @patch.object(c, "_get_session")
    def test_confirmed_ok(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(200, {
            "prediction_id": _PRED_ID, "feedback_status": "confirmed",
        })
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(_PRED_ID, _FP, "confirmed", api_url=self.URL)

        assert result.ok is True
        assert result.feedback_status == "confirmed"
        call = mock_session.post.call_args
        # URL es la DERIVADA (/feedback), no la /predict.
        assert call.args[0] == "https://api.test/prod/feedback"
        assert call.kwargs["json"] == {
            "prediction_id": _PRED_ID, "fingerprint": _FP, "action": "confirmed",
        }

    @patch.object(c, "_get_session")
    def test_corrected_incluye_corrected_species(
        self, mock_session_factory: MagicMock,
    ) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(
            200, {"feedback_status": "corrected"})
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(
            _PRED_ID, _FP, "corrected",
            corrected_species="Rhea americana", api_url=self.URL,
        )

        assert result.ok is True
        body = mock_session.post.call_args.kwargs["json"]
        assert body["action"] == "corrected"
        assert body["corrected_species"] == "Rhea americana"

    @patch.object(c, "_get_session")
    def test_rejected_as_non_bird_sin_species(
        self, mock_session_factory: MagicMock,
    ) -> None:
        """action=rejected_as_non_bird -> body sin corrected_species."""
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(
            200, {"feedback_status": "rejected_as_non_bird"})
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(
            _PRED_ID, _FP, "rejected_as_non_bird", api_url=self.URL)

        assert result.ok is True
        assert "corrected_species" not in mock_session.post.call_args.kwargs["json"]

    @patch.object(c, "_get_session")
    def test_404_not_found(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(
            404, {"error": "prediction_id no encontrado"})
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(_PRED_ID, _FP, "confirmed", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "not_found"

    @patch.object(c, "_get_session")
    def test_409_already_submitted_con_current_status(
        self, mock_session_factory: MagicMock,
    ) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(409, {
            "error": "esta predicción ya tiene feedback registrado",
            "current_status": "confirmed",
        })
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(_PRED_ID, _FP, "confirmed", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "already_submitted"
        assert result.current_status == "confirmed"

    @patch.object(c, "_get_session")
    def test_400_bad_request(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(
            400, {"error": "action inválido"})
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(_PRED_ID, _FP, "garbage", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "bad_request"

    @patch.object(c, "_get_session")
    def test_503_throttled(self, mock_session_factory: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_response(503, {})
        mock_session_factory.return_value = mock_session

        result = c.send_feedback(_PRED_ID, _FP, "confirmed", api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "throttled"
