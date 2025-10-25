"""Utilities for computing style fingerprints from reference images."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
os.environ.setdefault("TORCHVISION_DISABLE_VIDEO_OPT", "1")

import open_clip
import torch
from PIL import Image

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency in open-clip
    snapshot_download = None


@dataclass
class StyleFingerprint:
    """Aggregated statistics describing a visual style."""

    clip_mean: np.ndarray
    clip_std: np.ndarray
    color_mean: np.ndarray  # RGB channel means in [0, 1]
    color_std: np.ndarray   # RGB channel std in [0, 1]


class FingerprintExtractor:
    """Helper that wraps a CLIP model for embedding extraction."""

    _MODEL_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    _REQUIRED_PATTERNS = (
        "open_clip_config.json",
        "open_clip_model.safetensors",
        "open_clip_pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "preprocessor_config.json",
    )

    def __init__(
        self,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        prefer_offline_cache: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._local_repo_path: Path | None = None

        if prefer_offline_cache:
            self._local_repo_path = self._ensure_weights_available()

        model_name = "ViT-B-32"
        pretrained: str | None = "laion2b_s34b_b79k"

        if self._local_repo_path is not None:
            model_name = f"local-dir:{self._local_repo_path}"
            pretrained = None

        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        self.model = self.model.to(self.device).eval()
        self.preprocess = preprocess

    def _ensure_weights_available(self) -> Path | None:
        """Download the CLIP weights once, then default to offline cache usage."""
        if snapshot_download is None:
            # The open_clip fallback download logic will handle the rest.
            return None

        cache_dir = str(self.cache_dir) if self.cache_dir else None

        def _snapshot(local_only: bool) -> Path | None:
            try:
                path = Path(
                    snapshot_download(
                        repo_id=self._MODEL_REPO,
                        cache_dir=cache_dir,
                        local_files_only=local_only,
                        allow_patterns=self._REQUIRED_PATTERNS,
                    )
                )
                return path
            except Exception:
                return None

        # First check whether a local snapshot already exists.
        commit_path = _snapshot(local_only=True)
        if commit_path is not None and self._has_required_files(commit_path):
            self._force_offline_mode()
            return commit_path
        commit_path = None

        # Cache miss: allow an online download for the first run.
        commit_path = _snapshot(local_only=False)  # pragma: no cover - network path
        if commit_path is not None and self._has_required_files(commit_path):  # pragma: no cover - network path
            self._force_offline_mode()
            return commit_path

        raise RuntimeError(  # pragma: no cover - network path
            "Unable to download CLIP weights for the fingerprint extractor. "
            "Please ensure you have an internet connection for the first run."
        )

    @staticmethod
    def _force_offline_mode() -> None:
        """Ensure subsequent huggingface_hub calls stay local-only."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    @classmethod
    def _has_required_files(cls, base_dir: Path) -> bool:
        return all((base_dir / pattern).exists() for pattern in cls._REQUIRED_PATTERNS)

    def _embed(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0]

    @staticmethod
    def _color_stats(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return mean, std

    def compute(self, images: Iterable[Path]) -> StyleFingerprint:
        embeddings: List[np.ndarray] = []
        color_means: List[np.ndarray] = []
        color_stds: List[np.ndarray] = []

        for path in images:
            with Image.open(path) as img:
                embeddings.append(self._embed(img))
                mean, std = self._color_stats(img)
                color_means.append(mean)
                color_stds.append(std)

        if not embeddings:
            raise ValueError("No reference images provided for fingerprint computation.")

        clip_stack = np.stack(embeddings, axis=0)
        color_mean_stack = np.stack(color_means, axis=0)
        color_std_stack = np.stack(color_stds, axis=0)

        return StyleFingerprint(
            clip_mean=clip_stack.mean(axis=0),
            clip_std=clip_stack.std(axis=0),
            color_mean=color_mean_stack.mean(axis=0),
            color_std=color_std_stack.mean(axis=0),
        )
