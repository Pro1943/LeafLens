import cv2
import numpy as np
import pytest

from leaflens.enhance import auto_crop_to_leaf, fix_exposure, remove_glare, sharpen


def test_fix_exposure():
    dark_img = np.full((150, 150, 3), 20, dtype=np.uint8)
    enhanced, info = fix_exposure(dark_img)

    assert enhanced.shape == dark_img.shape
    assert info["gamma_applied"] is True
    assert np.mean(enhanced) > np.mean(dark_img)


def test_sharpen():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    blurred = cv2.GaussianBlur(img, (7, 7), 2.0)

    sharpened, info = sharpen(blurred, amount=1.5)
    assert sharpened.shape == blurred.shape
    assert info["amount"] == 1.5


def test_remove_glare():
    img = np.full((100, 100, 3), (35, 140, 45), dtype=np.uint8)
    img[40:60, 40:60] = (255, 255, 255)

    repaired, info, recon_mask = remove_glare(img)
    assert repaired.shape == img.shape
    assert info["glare_pixels_repaired"] > 0
    assert not np.array_equal(repaired, img)
    assert recon_mask.dtype == bool
    assert np.count_nonzero(recon_mask) > 0


def test_auto_crop_to_leaf():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    img[50:150, 60:140] = (35, 145, 45)

    cropped, info = auto_crop_to_leaf(img, padding=10)
    assert info["cropped"] is True
    h, w = cropped.shape[:2]
    assert h < 200
    assert w < 200
    assert h >= 100
    assert w >= 80


def test_enhance_invalid_raises():
    with pytest.raises(ValueError):
        fix_exposure(np.array([]))
    with pytest.raises(ValueError):
        sharpen(None)
