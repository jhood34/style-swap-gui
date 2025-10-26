"""Shared filesystem helpers (import/copy/cleanup) used by the GUI and CLI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")


def list_images(directory: Path) -> list[Path]:
    """Return all supported image files within ``directory`` in sorted order."""

    return sorted(path for pattern in IMAGE_PATTERNS for path in directory.glob(pattern))


def copy_images(paths: Sequence[Path], destination: Path) -> list[Path]:
    """Copy ``paths`` into ``destination`` and return the list of copied files."""

    copied: list[Path] = []
    for src in paths:
        dst = destination / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        copied.append(dst)
    return copied


def cleanup_files(directories: Iterable[Path]) -> None:
    """Delete files or subdirectories beneath each directory (best-effort)."""

    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                continue
