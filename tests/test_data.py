"""Tests del subpaquete ``src.data``."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.augmentations import build_audio_pipeline

SAMPLE_RATE = 48_000
N_SAMPLES = 144_000  # 3 s @ 48 kHz, ventana de BirdNET


@pytest.fixture
def audio_clip_3s() -> np.ndarray:
    """Tono 1 kHz a 0.3 de amplitud + ruido leve, float32 mono."""
    rng = np.random.default_rng(42)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    return (tone + 0.01 * rng.standard_normal(N_SAMPLES)).astype(np.float32)


def test_build_audio_pipeline_preserves_shape_and_dtype(audio_clip_3s):
    np.random.seed(0)
    pipe = build_audio_pipeline()
    out = pipe(samples=audio_clip_3s, sample_rate=SAMPLE_RATE)
    assert out.shape == audio_clip_3s.shape
    assert out.dtype == np.float32


def test_build_audio_pipeline_modifies_signal_with_p_one(audio_clip_3s):
    """Con p=1 forzado las tres transforms se aplican y el output difiere."""
    np.random.seed(0)
    pipe = build_audio_pipeline(p_noise=1.0, p_shift=1.0, p_gain=1.0)
    out = pipe(samples=audio_clip_3s.copy(), sample_rate=SAMPLE_RATE)
    assert not np.allclose(out, audio_clip_3s)
