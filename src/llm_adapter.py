"""Interface to a local LLM (via Ollama) for feedback interpretation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency for faster HTTP calls
    import requests
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_MODEL = "llama3.1:8b"


def _ollama_available() -> bool:
    return shutil.which("ollama") is not None


def parse_feedback_with_llm(feedback: str, model: str = DEFAULT_MODEL) -> Optional[Dict[str, Any]]:
    """Use a local LLM (Ollama) to convert free-form feedback into structured actions."""

    if not _ollama_available():
        return None

    prompt = textwrap.dedent(
        f"""
        You are an expert photo editing assistant. Always respond with a SINGLE JSON object that
        fits the following schema. Do not include prose, markdown, or comments.

        Schema:
        {{
          "actions": [
            {{
              "type": "update_parameters",
              "parameters": {{
                "strength_delta": float?,
                "saturation_delta": float?,
                "brightness_delta": float?,
                "shadow_delta": float?,
                "highlight_delta": float?,
                "contrast_delta": float?,
                "clarity_delta": float?,
                "temperature_delta": float?,
                "grain_strength": float?,
                "grayscale": bool?,
                "rotation": int?,
                "flip_horizontal": bool?,
                "flip_vertical": bool?,
                "reset": bool?
              }}
            }},
            {{
              "type": "respond",
              "message": string
            }}
          ]
        }}

        Rules:
        - Always include at least one action.
        - Use "respond" when you want to talk to the user.
        - Use "update_parameters" when you want to change image settings.
        - Every numeric parameter uses a unified [-100, 100] scale where 0 = neutral/current value, +100 = maximum increase, -100 = maximum decrease.
        - If you cannot act, respond with an empty parameter update and a message explaining why.

        Parameters (all on the [-100, 100] scale):
        - strength_delta: adjusts how strongly the reference style is blended.
        - saturation_delta: negative desaturates, positive boosts color.
        - brightness_delta: negative darkens, positive brightens.
        - shadow_delta: negative deepens shadows, positive lifts them.
        - highlight_delta: negative brightens highlights, positive compresses them.
        - contrast_delta: negative flattens, positive increases contrast.
        - clarity_delta: negative softens, positive adds midtone detail.
        - temperature_delta: negative cools, positive warms.
        - grain_strength: absolute target level on the same scale (0 = none, 100 = maximum grain).
        - grayscale: boolean to enable or disable monochrome.
        - rotation: integer degrees to rotate relative to the current orientation.
        - flip_horizontal / flip_vertical: booleans to set mirror state.
        - reset: true to restore all parameters to their defaults.

        Input: "{feedback}"
        JSON:
        """
    ).strip()

    output = _call_ollama(prompt, model)
    if output is None:
        return None

    return _extract_actions(output)


_HTTP_SESSION: Optional["requests.Session"] = None


def _call_ollama(prompt: str, model: str) -> Optional[str]:
    if requests is not None and os.environ.get("OLLAMA_DISABLE_HTTP", "0") != "1":
        global _HTTP_SESSION
        if _HTTP_SESSION is None:  # pragma: no cover - simple lazy init
            _HTTP_SESSION = requests.Session()
        try:
            # Prefer the local HTTP API when requests is available—it avoids spawning a process per call.
            response = _HTTP_SESSION.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "keep_alive": -1,
                    "stream": False,
                },
                timeout=float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "60")),
            )
            if response.ok:
                data = response.json()
                if isinstance(data, dict) and "response" in data:
                    return str(data["response"])
        except Exception:  # pragma: no cover - network errors fallback to CLI
            pass

    env = os.environ.copy()
    env.setdefault("OLLAMA_KEEP_ALIVE", "-1")
    try:
        # Fallback: shell out to `ollama run`, piping the prompt via stdin.
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout


def _extract_actions(output: str) -> Optional[Dict[str, Any]]:
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None

    snippet = output[start : end + 1]
    # LLMs sometimes wrap valid JSON in markdown or comments—strip them eagerly.
    snippet = snippet.replace("```json", "").replace("```", "")
    snippet = "\n".join(line for line in snippet.splitlines() if not line.strip().startswith("//"))
    snippet = re.sub(r"//.*", "", snippet)

    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    actions = data.get("actions")
    if not isinstance(actions, list):
        return None

    normalized: Dict[str, Any] = {}
    messages: list[str] = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = action.get("type")
        if action_type == "respond" and isinstance(action.get("message"), str):
            messages.append(action["message"].strip())
        elif action_type == "update_parameters":
            params = action.get("parameters", {})
            if isinstance(params, dict):
                normalized.update(_normalize_parameter_update(params))

    if messages:
        normalized["messages"] = messages

    return normalized or None


def _normalize_parameter_update(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "strength_delta",
        "saturation_delta",
        "brightness_delta",
        "shadow_delta",
        "highlight_delta",
        "contrast_delta",
        "clarity_delta",
        "temperature_delta",
        "grain_strength",
        "grayscale",
        "rotation",
        "flip_horizontal",
        "flip_vertical",
        "reset",
    }

    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed:
            continue
        if key in {"grayscale", "flip_horizontal", "flip_vertical", "reset"}:
            normalized[key] = bool(value)
        elif key == "rotation":
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                continue

    return normalized
