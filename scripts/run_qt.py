#!/usr/bin/env python3
"""Launch the PyQt GUI for the style transfer agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui import launch_gui
from src.session import StyleTransferSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Style transfer GUI")
    parser.add_argument("--references", type=Path, default=Path("data/references"))
    parser.add_argument("--inputs", type=Path, default=Path("data/inputs"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/styled"))
    parser.add_argument("--device", default=None, help="Torch device to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = StyleTransferSession(args.references, args.inputs, args.outputs, device=args.device)
    try:
        launch_gui(session)
    finally:
        session.cleanup_assets()


if __name__ == "__main__":
    main()
