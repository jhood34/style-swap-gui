from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent import StyleTransferAgent
from src.voice_transcriber import TranscriptionError


def _stub_agent(transcribed_text: str = "hello world") -> StyleTransferAgent:
    agent = StyleTransferAgent.__new__(StyleTransferAgent)
    agent._voice_transcriber = None

    def _ensure():
        return SimpleNamespace(transcribe_text=lambda path: transcribed_text)

    agent._ensure_voice_transcriber = _ensure  # type: ignore[attr-defined]
    return agent


def test_maybe_transcribe_voice_returns_text(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"abc")

    agent = _stub_agent("spoken feedback")

    result, used_voice = agent._maybe_transcribe_voice_feedback(f":voice {audio_path}")

    assert used_voice
    assert result == "spoken feedback"


def test_maybe_transcribe_voice_accepts_alias(tmp_path):
    audio_path = tmp_path / "note.m4a"
    audio_path.write_bytes(b"abc")

    agent = _stub_agent("alias text")

    result, used_voice = agent._maybe_transcribe_voice_feedback(f"voice {audio_path}")

    assert used_voice
    assert result == "alias text"


def test_maybe_transcribe_voice_requires_path():
    agent = _stub_agent()

    with pytest.raises(TranscriptionError):
        agent._maybe_transcribe_voice_feedback(":voice")


def test_maybe_transcribe_voice_missing_file(tmp_path):
    missing = tmp_path / "missing.wav"
    agent = _stub_agent()

    with pytest.raises(TranscriptionError) as exc:
        agent._maybe_transcribe_voice_feedback(f":voice {missing}")

    assert "Audio file not found" in str(exc.value)


def test_non_voice_feedback_passes_through():
    agent = _stub_agent()
    text, used_voice = agent._maybe_transcribe_voice_feedback("make it warmer")
    assert not used_voice
    assert text == "make it warmer"
