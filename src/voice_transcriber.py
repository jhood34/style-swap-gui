"""Voice-to-text utilities powered by faster-whisper.

This module wraps the WhisperModel from faster-whisper with a small helper that
prioritises fast start-up, single-sentence latency, and Apple Silicon support.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

try:  # pragma: no cover - exercised indirectly via monkeypatch in tests
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled dynamically at runtime
    WhisperModel = None  # type: ignore


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Container for a single transcription call."""

    text: str
    language: Optional[str]
    duration: Optional[float]
    average_logprob: Optional[float]


def _detect_device() -> Tuple[str, str]:
    """Return a (device, compute_type) tuple suited to the current machine."""

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch may not be installed
        return "cpu", "int8"

    try:
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


def suggest_device_settings(
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Tuple[str, str]:
    """Return the device configuration, filling missing values automatically."""

    if device and compute_type:
        return device, compute_type

    detected_device, detected_compute = _detect_device()

    if device and not compute_type:
        if device == "cpu":
            return device, "int8"
        return device, "float16"

    if compute_type and not device:
        if "float16" in compute_type:
            if detected_device == "cuda":
                return "cuda", compute_type
            return "cpu", "int8"
        return detected_device, compute_type

    return detected_device, detected_compute


class FasterWhisperTranscriber:
    """Lightweight wrapper around faster-whisper with sensible defaults."""

    def __init__(
        self,
        model_size: str = "base.en",
        *,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[Path] = None,
    ) -> None:
        self.model_size = model_size
        # Auto-pick sensible defaults so beginners rarely have to think about acceleration flags.
        self.device, self.compute_type = suggest_device_settings(device, compute_type)
        self.download_root = download_root
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        """Lazy-load WhisperModel once, retrying with a CPU/INT8 fallback when necessary."""
        if self._model is not None:
            return self._model

        if WhisperModel is None:
            raise TranscriptionError(
                "faster-whisper is not installed. Install it with `pip install faster-whisper`."
            )

        with self._lock:
            if self._model is not None:
                return self._model

            kwargs = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.download_root is not None:
                kwargs["download_root"] = str(self.download_root)
            try:
                self._model = WhisperModel(self.model_size, **kwargs)
            except ValueError as exc:  # pragma: no cover - fallback exercised in tests
                if kwargs.get("compute_type") != "int8":
                    # Some combinations (e.g. Metal + float16) may not be supported, so fall back to a safe CPU mode.
                    kwargs["compute_type"] = "int8"
                    kwargs["device"] = "cpu"
                    self.compute_type = "int8"
                    self.device = "cpu"
                    try:
                        self._model = WhisperModel(self.model_size, **kwargs)
                    except Exception as retry_exc:  # pragma: no cover - escalated to user
                        raise TranscriptionError(
                            f"Failed to load faster-whisper model after fallback: {retry_exc}"
                        ) from retry_exc
                else:
                    raise TranscriptionError(f"Failed to load faster-whisper model: {exc}") from exc
            return self._model

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: str | Path | Iterable[float] | Any,
        *,
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = False,
        temperature: float | None = None,
        compression_ratio_threshold: float | None = None,
        no_speech_threshold: float | None = None,
        condition_on_previous_text: bool | None = None,
        without_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file or waveform and return structured metadata."""

        model = self._ensure_model()

        source = str(audio) if isinstance(audio, Path) else audio
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if compression_ratio_threshold is not None:
            kwargs["compression_ratio_threshold"] = compression_ratio_threshold
        if no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = no_speech_threshold
        if condition_on_previous_text is not None:
            kwargs["condition_on_previous_text"] = condition_on_previous_text
        if without_timestamps is not None:
            kwargs["without_timestamps"] = without_timestamps

        try:
            segments, info = model.transcribe(
                source,
                beam_size=beam_size,
                language=language,
                vad_filter=vad_filter,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        text_parts = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if text:
                # Stitch sentence fragments together because we ask Whisper for streaming-style segments.
                text_parts.append(text.strip())

        text = " ".join(part for part in text_parts if part).strip()

        language_value = getattr(info, "language", None)
        duration_value = getattr(info, "duration", None)
        avg_logprob_value = getattr(info, "average_logprob", None)

        return TranscriptionResult(
            text=text,
            language=language_value,
            duration=duration_value,
            average_logprob=avg_logprob_value,
        )

    # ------------------------------------------------------------------
    def transcribe_text(
        self,
        audio: str | Path | Iterable[float] | Any,
        *,
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = False,
        temperature: float | None = None,
        compression_ratio_threshold: float | None = None,
        no_speech_threshold: float | None = None,
        condition_on_previous_text: bool | None = None,
        without_timestamps: bool | None = None,
    ) -> str:
        """Convenience helper that returns just the recognised text."""

        result = self.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            without_timestamps=without_timestamps,
        )
        return result.text
