import cv2
import numpy as np
import pytest

from leaflens.discolor import find_worst_patch, measure_discoloration


def create_leaf_with_discoloration():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)

    img[40:160, 40:160] = (35, 150, 45)
    mask[40:160, 40:160] = 255

    img[50:80, 50:80] = (25, 210, 220)
    img[120:145, 120:145] = (20, 60, 110)
    img[100:115, 60:75] = (15, 15, 20)

    cv2.circle(img, (130, 70), 8, (240, 240, 240), -1)
    mask[62:78, 122:138] = 0

    return img, mask


def test_measure_discoloration():
    img, mask = create_leaf_with_discoloration()
    stats, overlay = measure_discoloration(img, mask)

    assert overlay.shape == img.shape
    assert stats["healthy_green_pct"] > 0
    assert stats["yellowing_pct"] > 0
    assert stats["browning_pct"] > 0
    assert stats["dark_spots_pct"] > 0
    assert stats["holes_count"] >= 1


def test_find_worst_patch():
    img, mask = create_leaf_with_discoloration()
    res = find_worst_patch(img, mask)

    bbox = res["bbox"]
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert res["discoloration_score"] > 0
    assert res["discolored_pixel_count"] > 0


def test_discolor_empty_mask():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    stats, overlay = measure_discoloration(img, mask)

    assert stats["leaf_pixel_count"] == 0
    assert overlay.shape == img.shape


def test_discolor_invalid_input():
    with pytest.raises(ValueError):
        measure_discoloration(np.array([]))


def test_yellow_patch_under_synthetic_glare_not_turned_green():
    from leaflens.enhance import remove_glare

    img = np.full((160, 160, 3), 240, dtype=np.uint8)
    mask = np.zeros((160, 160), dtype=np.uint8)

    img[20:140, 20:140] = (35, 150, 45)
    mask[20:140, 20:140] = 255

    img[40:80, 40:80] = (25, 210, 220)

    glare_img = img.copy()
    glare_img[40:80, 40:80] = (255, 255, 255)

    repaired, _, recon_mask = remove_glare(glare_img, leaf_mask=mask)

    stats_with_exclusion, _ = measure_discoloration(
        repaired,
        mask=mask,
        reconstructed_mask=recon_mask,
    )

    stats_without_exclusion, _ = measure_discoloration(
        repaired,
        mask=mask,
        reconstructed_mask=None,
    )

    assert stats_with_exclusion["reconstructed_pct"] > 0
    assert stats_with_exclusion["valid_leaf_pixel_count"] < stats_with_exclusion["leaf_pixel_count"]
    
    green_count_with_exclusion = int(
        round((stats_with_exclusion["healthy_green_pct"] / 100.0) * stats_with_exclusion["valid_leaf_pixel_count"])
    )
    green_count_without_exclusion = int(
        round((stats_without_exclusion["healthy_green_pct"] / 100.0) * stats_without_exclusion["leaf_pixel_count"])
    )
    assert green_count_with_exclusion < green_count_without_exclusion
