import numpy as np
from PIL import Image

from src.fingerprint import StyleFingerprint
from src.transformer import apply_style


def test_apply_style_adjusts_mean():
    # Create a blue-ish image and a red-ish reference fingerprint.
    blue = Image.fromarray(np.full((32, 32, 3), [30, 60, 200], dtype=np.uint8))
    fingerprint = StyleFingerprint(
        clip_mean=np.zeros(512),
        clip_std=np.ones(512),
        color_mean=np.array([0.8, 0.2, 0.2]),
        color_std=np.array([0.1, 0.1, 0.1]),
    )

    result = apply_style(blue, fingerprint)
    arr = np.asarray(result, dtype=np.float32) / 255.0

    # Red channel should be boosted relative to blue channel.
    assert arr.mean() > 0.4
    assert arr[..., 0].mean() > arr[..., 2].mean()
