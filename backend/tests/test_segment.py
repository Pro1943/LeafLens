import cv2
import numpy as np
import pytest

from leaflens.segment import segment_leaf


def test_segment_leaf_green():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    img[50:150, 50:150] = (30, 160, 40)

    mask, info = segment_leaf(img)
    assert mask.shape == (200, 200)
    assert info["method"] == "hsv_green"
    assert info["mask_area_ratio"] > 0.2
    assert mask[100, 100] == 255
    assert mask[10, 10] == 0


def test_segment_leaf_non_green_fallback():
    img = np.full((200, 200, 3), 245, dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (30, 20, 140), -1)

    mask, info = segment_leaf(img)
    assert mask.shape == (200, 200)
    assert info["is_fallback"] is True
    assert mask[100, 100] == 255


def test_segment_leaf_invalid():
    with pytest.raises(ValueError):
        segment_leaf(np.zeros((0, 0, 3), dtype=np.uint8))
