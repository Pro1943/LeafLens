from typing import Any
import cv2
import numpy as np

from leaflens.config import (
    GREEN_HUE_MIN,
    GREEN_HUE_MAX,
    GREEN_SAT_MIN,
    GREEN_VAL_MIN,
    MAX_IMAGE_DIMENSION,
)
from leaflens.io_utils import downscale_if_needed, validate_image


def segment_leaf(
    img: np.ndarray,
    green_hue_range: tuple[int, int] = (GREEN_HUE_MIN, GREEN_HUE_MAX),
    sat_min: int = GREEN_SAT_MIN,
    val_min: int = GREEN_VAL_MIN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment leaf from image using HSV green detection with GrabCut fallback."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    height, width = proc_img.shape[:2]
    total_pixels = height * width

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([green_hue_range[0], sat_min, val_min], dtype=np.uint8)
    upper_green = np.array([green_hue_range[1], 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_hsv_mask = cv2.morphologyEx(cleaned_hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    hsv_ratio = float(np.count_nonzero(cleaned_hsv_mask) / total_pixels)

    if hsv_ratio >= 0.05:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_hsv_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            final_mask = np.zeros_like(cleaned_hsv_mask)
            final_mask[labels == largest_label] = 255
            filled_mask = _fill_mask_holes(final_mask)
            area_ratio = float(np.count_nonzero(filled_mask) / total_pixels)
            return filled_mask, {
                "method": "hsv_green",
                "mask_area_ratio": round(area_ratio, 4),
                "is_fallback": False,
            }

    grabcut_mask = _segment_grabcut_fallback(proc_img)
    grabcut_ratio = float(np.count_nonzero(grabcut_mask) / total_pixels)

    if grabcut_ratio < 0.01:
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fallback_mask = np.zeros_like(otsu)
            fallback_mask[labels == largest_label] = 255
            filled = _fill_mask_holes(fallback_mask)
            ratio = float(np.count_nonzero(filled) / total_pixels)
            return filled, {
                "method": "morphological_fallback",
                "mask_area_ratio": round(ratio, 4),
                "is_fallback": True,
            }

    return grabcut_mask, {
        "method": "grabcut",
        "mask_area_ratio": round(grabcut_ratio, 4),
        "is_fallback": True,
    }


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _segment_grabcut_fallback(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))

    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            largest_mask = np.zeros_like(binary_mask)
            largest_mask[labels == largest_label] = 255
            return _fill_mask_holes(largest_mask)
        return binary_mask
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)
