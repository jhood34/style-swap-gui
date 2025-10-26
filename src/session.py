"""Utility session for driving the style-transfer pipeline programmatically."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .agent import AgentConfig
from .fingerprint import FingerprintExtractor, StyleFingerprint
from .filmulator_engine import FilmulatorParameters
from .interaction import interpret_feedback
from .transformer import apply_style_to_path
from .utils.filesystem import cleanup_files, list_images


@dataclass
class StylisedImage:
    input_path: Path
    output_path: Path


class StyleTransferSession:
    """High-level API for styling images and applying iterative feedback."""

    def __init__(
        self,
        reference_dir: Path,
        input_dir: Path,
        output_dir: Path,
        device: str | None = None,
    ) -> None:
        self.config = AgentConfig(
            reference_dir=reference_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            device=device,
            interactive=False,
        )
        self.output_dir = output_dir
        self.extractor = FingerprintExtractor(device=device)
        self._fingerprint: StyleFingerprint | None = None
        self._image_params: dict[Path, FilmulatorParameters] = {}
        self._fallback_fingerprints: dict[Path, StyleFingerprint] = {}

    # ------------------------------------------------------------------
    def list_inputs(self) -> List[Path]:
        paths = list_images(self.config.input_dir)
        if not paths:
            raise ValueError(f"No input images found in {self.config.input_dir}")
        return paths

    def _gather_reference_images(self) -> list[Path]:
        return list_images(self.config.reference_dir)

    def fingerprint(self) -> StyleFingerprint:
        if self._fingerprint is None:
            # Cache the expensive CLIP fingerprint so multiple images reuse it.
            references = self._gather_reference_images()
            if not references:
                raise ValueError(f"No reference images found in {self.config.reference_dir}")
            self._fingerprint = self.extractor.compute(references)
            self._fallback_fingerprints.clear()
        return self._fingerprint

    def _params_for(self, input_path: Path) -> FilmulatorParameters:
        return self._image_params.setdefault(input_path, FilmulatorParameters())

    def stylise_image(self, input_path: Path) -> Path:
        output_path = self.output_dir / input_path.name
        try:
            fingerprint = self.fingerprint()
        except ValueError:
            fingerprint = self._fallback_fingerprints.get(input_path)
            if fingerprint is None:
                fingerprint = self.extractor.compute([input_path])
                self._fallback_fingerprints[input_path] = fingerprint
        apply_style_to_path(input_path, output_path, fingerprint, params=self._params_for(input_path))
        return output_path

    def stylise_all(self) -> List[StylisedImage]:
        """Run stylise_image for every file currently on disk."""
        results: List[StylisedImage] = []
        for path in self.list_inputs():
            results.append(StylisedImage(path, self.stylise_image(path)))
        return results

    def reset_parameters(self) -> None:
        """Forget all per-image adjustments so future renders use neutral sliders."""
        for key in list(self._image_params.keys()):
            self._image_params[key] = FilmulatorParameters()

    def reset_parameters_for(self, input_path: Path) -> None:
        """Reset the parameters for a single image only."""
        self._image_params[input_path] = FilmulatorParameters()

    def refresh_fingerprint(self) -> None:
        references = self._gather_reference_images()
        if not references:
            raise ValueError(f"No reference images found in {self.config.reference_dir}")
        self._fingerprint = self.extractor.compute(references)
        self._fallback_fingerprints.clear()

    def has_references(self) -> bool:
        return bool(self._gather_reference_images())

    def has_fingerprint(self) -> bool:
        return self._fingerprint is not None or bool(self._fallback_fingerprints)

    # Cleanup -----------------------------------------------------------
    def cleanup_assets(self) -> None:
        cleanup_files(
            (
                self.config.reference_dir,
                self.config.input_dir,
                self.output_dir,
            )
        )
        self._fallback_fingerprints.clear()

    def apply_feedback(self, feedback: str, input_path: Path) -> tuple[bool, list[str]]:
        """Delegate natural-language feedback to the interpreter and persist the change."""
        params = self._params_for(input_path)
        updated, changed, messages = interpret_feedback(feedback, params)
        if changed:
            self._image_params[input_path] = updated
        return changed, messages

    def set_parameter(self, input_path: Path, name: str, value: float) -> None:
        params = self._params_for(input_path)
        if isinstance(value, (int, float)):
            value = max(-100.0, min(100.0, float(value)))
        setattr(params, name, value)

    def current_parameters(self, input_path: Path) -> FilmulatorParameters:
        return self._params_for(input_path)
