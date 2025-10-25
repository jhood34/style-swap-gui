"""Wrapper utilities for applying the Filmulator-inspired engine to images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .filmulator_engine import FilmulatorEngine, FilmulatorParameters
from .fingerprint import StyleFingerprint


def apply_style(
    image: Image.Image,
    fingerprint: StyleFingerprint,
    params: FilmulatorParameters | None = None,
) -> Image.Image:
    engine = FilmulatorEngine(fingerprint)
    return engine.apply(image, params or FilmulatorParameters())


def apply_style_to_path(
    input_path: Path,
    output_path: Path,
    fingerprint: StyleFingerprint,
    params: FilmulatorParameters | None = None,
) -> Path:
    with Image.open(input_path) as img:
        styled = apply_style(img, fingerprint, params=params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        styled.save(output_path)
    return output_path

