import cv2
import numpy as np
import pytest

from leaflens.quality import (
    measure_blur,
    measure_exposure,
    measure_glare,
    measure_leaf_coverage,
    quality_report,
)


def create_synthetic_sharp():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (255, 255, 255)
    for i in range(60, 240, 15):
        cv2.line(img, (i, 60), (i, 240), (0, 0, 0), 2)
    return img


def create_synthetic_blurred():
    sharp = create_synthetic_sharp()
    return cv2.GaussianBlur(sharp, (31, 31), 10)


def create_synthetic_dark():
    return np.full((300, 300, 3), 10, dtype=np.uint8)


def create_synthetic_overexposed():
    return np.full((300, 300, 3), 250, dtype=np.uint8)


def create_synthetic_glare():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (30, 120, 30)
    img[100:180, 100:180] = (255, 255, 255)
    return img


def create_synthetic_leaf(leaf_fraction=0.4):
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    h, w = 300, 300
    side = int(np.sqrt(h * w * leaf_fraction))
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    img[y1 : y1 + side, x1 : x1 + side] = (30, 160, 40)
    return img


def test_measure_blur():
    sharp = create_synthetic_sharp()
    blurred = create_synthetic_blurred()

    res_sharp = measure_blur(sharp)
    res_blur = measure_blur(blurred)

    assert res_sharp["blur_score"] > res_blur["blur_score"]
    assert res_blur["is_blurry"] is True
    assert res_sharp["is_blurry"] is False


def test_measure_exposure():
    dark = create_synthetic_dark()
    bright = create_synthetic_overexposed()
    normal = np.full((300, 300, 3), 128, dtype=np.uint8)

    res_dark = measure_exposure(dark)
    res_bright = measure_exposure(bright)
    res_normal = measure_exposure(normal)

    assert res_dark["status"] == "dark"
    assert res_bright["status"] == "overexposed"
    assert res_normal["status"] == "ok"


def test_measure_glare():
    glare_img = create_synthetic_glare()
    no_glare_img = np.full((300, 300, 3), (30, 150, 40), dtype=np.uint8)

    res_glare, mask = measure_glare(glare_img)
    res_no_glare, _ = measure_glare(no_glare_img)

    assert res_glare["has_glare"] is True
    assert res_glare["glare_ratio"] > 0.015
    assert res_no_glare["has_glare"] is False
    assert mask.shape == glare_img.shape[:2]


def test_measure_leaf_coverage():
    big_leaf = create_synthetic_leaf(0.4)
    tiny_leaf = create_synthetic_leaf(0.01)

    res_big = measure_leaf_coverage(big_leaf)
    res_tiny = measure_leaf_coverage(tiny_leaf)

    assert res_big["coverage"] > 0.08
    assert res_big["too_small"] is False
    assert res_tiny["too_small"] is True


def test_quality_report():
    dark_blurry = create_synthetic_blurred()
    dark_blurry = (dark_blurry * 0.1).astype(np.uint8)

    report = quality_report(dark_blurry)
    assert "blur" in report["needs_fix"]
    assert "dark" in report["needs_fix"]
    assert report["is_acceptable"] is False


def test_quality_report_excessive_reconstruction():
    leaf_img = create_synthetic_leaf(0.4)
    recon_mask = np.full((300, 300), False, dtype=bool)
    recon_mask[100:200, 100:200] = True

    report = quality_report(leaf_img, reconstructed_mask=recon_mask)
    assert "excessive_reconstruction" in report["needs_fix"]
    assert report["reconstructed"]["is_excessive"] is True
    assert report["is_acceptable"] is False


def test_invalid_image_raises_error():
    with pytest.raises(ValueError):
        measure_blur(np.array([]))
    with pytest.raises(ValueError):
        measure_exposure(None)
