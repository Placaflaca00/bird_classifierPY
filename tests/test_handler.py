"""Unit tests del gating BirdNET nativo (Fase 2 - Nivel 2).

Testea las funciones puras de detection: ``_sigmoid`` y ``_evaluate_detection``.
Importar ``handler`` dispara la carga de modelos (~5 s cold start), pero esos
modelos no se invocan en estos tests — solo verificamos logica numerica.
Integration test del handler completo (con audio real) corre aparte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "lambda"))

import handler as h  # noqa: E402


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
