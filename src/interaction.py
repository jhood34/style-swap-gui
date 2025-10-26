"""Human-in-the-loop feedback interpretation driven entirely by the LLM."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .filmulator_engine import FilmulatorParameters
from .llm_adapter import parse_feedback_with_llm

CONTROL_MIN = -100.0
CONTROL_MAX = 100.0


def interpret_feedback(feedback: str, params: FilmulatorParameters) -> Tuple[FilmulatorParameters, bool, List[str]]:
    """Delegate feedback parsing to the LLM and apply returned adjustments."""

    if not feedback or not feedback.strip():
        return params, False, []

    llm_result = parse_feedback_with_llm(feedback)
    if not llm_result:
        return params, False, []

    messages = llm_result.get("messages")
    if isinstance(messages, list):
        llm_result = {k: v for k, v in llm_result.items() if k != "messages"}
    else:
        messages = []

    updated = replace(params)
    changed = _apply_llm_result(updated, llm_result)
    if changed:
        _clamp_parameters(updated)
        return updated, True, messages
    return params, False, messages


def _apply_llm_result(params: FilmulatorParameters, llm_result: Dict[str, Any]) -> bool:
    changed = False

    def _apply_delta(attr: str, key: str) -> None:
        nonlocal changed
        if key not in llm_result:
            return
        value = float(llm_result[key])
        # Each delta nudges the current slider value, so clamp after the addition.
        setattr(params, attr, _clamp_control(getattr(params, attr) + value))
        changed = True

    _apply_delta("strength", "strength_delta")
    _apply_delta("saturation_scale", "saturation_delta")
    _apply_delta("brightness_shift", "brightness_delta")
    _apply_delta("shadow_lift", "shadow_delta")
    _apply_delta("highlight_compress", "highlight_delta")
    _apply_delta("contrast", "contrast_delta")
    _apply_delta("clarity", "clarity_delta")
    _apply_delta("color_temperature", "temperature_delta")

    if "grain_strength" in llm_result:
        params.grain_strength = _clamp_control(float(llm_result["grain_strength"]))
        changed = True

    if "grayscale" in llm_result:
        params.grayscale = bool(llm_result["grayscale"])
        changed = True

    if "rotation" in llm_result:
        params.rotation_degrees = (params.rotation_degrees + int(llm_result["rotation"])) % 360
        changed = True

    if "flip_horizontal" in llm_result:
        params.flip_horizontal = bool(llm_result["flip_horizontal"])
        changed = True

    if "flip_vertical" in llm_result:
        params.flip_vertical = bool(llm_result["flip_vertical"])
        changed = True

    if "reset" in llm_result and llm_result["reset"]:
        # Reset is easiest by overwriting the dataclass with a fresh default instance.
        params.__dict__.update(FilmulatorParameters().__dict__)
        changed = True

    return changed


def _clamp_parameters(params: FilmulatorParameters) -> None:
    params.strength = _clamp_control(params.strength)
    params.saturation_scale = _clamp_control(params.saturation_scale)
    params.brightness_shift = _clamp_control(params.brightness_shift)
    params.shadow_lift = _clamp_control(params.shadow_lift)
    params.highlight_compress = _clamp_control(params.highlight_compress)
    params.contrast = _clamp_control(params.contrast)
    params.clarity = _clamp_control(params.clarity)
    params.color_temperature = _clamp_control(params.color_temperature)
    params.grain_strength = _clamp_control(params.grain_strength)


def _clamp_control(value: float) -> float:
    # Keep every control inside the GUI's [-100, 100] safety band.
    return max(CONTROL_MIN, min(CONTROL_MAX, float(value)))
