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

        result = c.predict(str(audio_file), top_k=3, api_url=self.URL)

        assert result.ok is True
        assert result.detected is True
        assert len(result.predictions) == 1
        assert result.predictions[0]["species"] == "Test bird"

        # 2 POSTs al session (upload-url + predict), 1 PUT a S3.
        assert mock_session.post.call_count == 2
        assert mock_put.call_count == 1
        # Body del segundo POST tiene s3_key + top_k, NO audio_b64.
        second_call = mock_session.post.call_args_list[1]
        assert second_call.kwargs["json"] == {
            "s3_key": "uploads/abc-uuid.mp3", "top_k": 3,
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

        result = c.predict(str(audio_file), api_url=self.URL)

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

        result = c.predict(str(audio_file), api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "upload_failed"
        # Solo 1 POST al session (upload-url); el predict no se llamó.
        assert mock_session.post.call_count == 1

    @patch("client.requests.put")
    @patch.object(c, "_get_session")
    def test_fails_at_predict_lambda(
        self, mock_session_factory: MagicMock, mock_put: MagicMock, tmp_path: Path,
    ) -> None:
        """Steps 1+2+3 OK, paso final /predict devuelve 500."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"bytes")

        upload_resp = _mock_response(200, {
            "upload_url": "https://s3.test/url",
            "s3_key": "uploads/x.mp3",
            "content_type": "audio/mpeg",
        })
        predict_resp = _mock_response(500, {"error": "lambda boom"})

        mock_session = MagicMock()
        mock_session.post.side_effect = [upload_resp, predict_resp]
        mock_session_factory.return_value = mock_session

        mock_put.return_value = _mock_response(200)

        result = c.predict(str(audio_file), api_url=self.URL)

        assert result.ok is False
        assert result.error_kind == "server"
        # Los 3 calls se hicieron (S3 OK, falla en el ultimo).
        assert mock_session.post.call_count == 2
        assert mock_put.call_count == 1
