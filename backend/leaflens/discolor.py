from typing import Any, Optional
import cv2
import numpy as np

from leaflens.config import (
    BROWN_HUE_MAX,
    BROWN_HUE_MIN,
    BROWN_SAT_MAX,
    BROWN_SAT_MIN,
    BROWN_VAL_MAX,
    BROWN_VAL_MIN,
    DARK_SPOT_VAL_MAX,
    GREEN_HUE_MAX,
    GREEN_HUE_MIN,
    GREEN_SAT_MIN,
    GREEN_VAL_MIN,
    MAX_HOLE_AREA_RATIO,
    MAX_IMAGE_DIMENSION,
    MIN_HOLE_AREA,
    PATCH_WINDOW_SIZES,
    YELLOW_HUE_MAX,
    YELLOW_HUE_MIN,
    YELLOW_SAT_MIN,
    YELLOW_VAL_MIN,
)
from leaflens.io_utils import downscale_if_needed, validate_image
from leaflens.segment import segment_leaf


def measure_discoloration(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reconstructed_mask: Optional[np.ndarray] = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure healthy green, yellowing, browning, dark spots, holes, and return overlay image excluding reconstructed pixels."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)

    if mask is None:
        mask, _ = segment_leaf(proc_img)
    else:
        validate_image(mask)
        if mask.shape[:2] != proc_img.shape[:2]:
            mask = cv2.resize(mask, (proc_img.shape[1], proc_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    leaf_binary = (mask > 127).astype(np.uint8)
    leaf_pixel_count = int(np.count_nonzero(leaf_binary))

    recon_bool = np.zeros(proc_img.shape[:2], dtype=bool)
    if reconstructed_mask is not None:
        if reconstructed_mask.shape[:2] != proc_img.shape[:2]:
            reconstructed_mask = cv2.resize(
                reconstructed_mask.astype(np.uint8),
                (proc_img.shape[1], proc_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        recon_bool = (reconstructed_mask > 0) & (leaf_binary > 0)

    reconstructed_count = int(np.count_nonzero(recon_bool))
    reconstructed_pct = round((reconstructed_count / leaf_pixel_count * 100.0), 2) if leaf_pixel_count > 0 else 0.0

    valid_leaf_mask = (leaf_binary > 0) & (~recon_bool)
    valid_leaf_count = int(np.count_nonzero(valid_leaf_mask))

    holes_count, holes_mask = _count_internal_holes(leaf_binary)

    overlay = proc_img.copy()

    if leaf_pixel_count == 0:
        empty_stats = {
            "healthy_green_pct": 0.0,
            "yellowing_pct": 0.0,
            "browning_pct": 0.0,
            "dark_spots_pct": 0.0,
            "reconstructed_pct": 0.0,
            "holes_count": holes_count,
            "leaf_pixel_count": 0,
            "valid_leaf_pixel_count": 0,
        }
        return empty_stats, overlay

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    denom_pixel_count = valid_leaf_count if valid_leaf_count > 0 else leaf_pixel_count

    healthy_mask = (
        valid_leaf_mask
        & (h_channel >= GREEN_HUE_MIN)
        & (h_channel <= GREEN_HUE_MAX)
        & (s_channel >= GREEN_SAT_MIN)
        & (v_channel >= GREEN_VAL_MIN)
    ).astype(bool)

    dark_spots_mask = (valid_leaf_mask & (v_channel <= DARK_SPOT_VAL_MAX)).astype(bool)

    yellow_mask = (
        valid_leaf_mask
        & (~dark_spots_mask)
        & (h_channel >= YELLOW_HUE_MIN)
        & (h_channel <= YELLOW_HUE_MAX)
        & (s_channel >= YELLOW_SAT_MIN)
        & (v_channel >= YELLOW_VAL_MIN)
    ).astype(bool)

    brown_mask = (
        valid_leaf_mask
        & (~dark_spots_mask)
        & (~yellow_mask)
        & (h_channel >= BROWN_HUE_MIN)
        & (h_channel <= BROWN_HUE_MAX)
        & (s_channel >= BROWN_SAT_MIN)
        & (s_channel <= BROWN_SAT_MAX)
        & (v_channel >= BROWN_VAL_MIN)
        & (v_channel <= BROWN_VAL_MAX)
    ).astype(bool)

    holes_bool = (holes_mask > 0)

    healthy_count = int(np.count_nonzero(healthy_mask))
    yellow_count = int(np.count_nonzero(yellow_mask))
    brown_count = int(np.count_nonzero(brown_mask))
    dark_count = int(np.count_nonzero(dark_spots_mask))

    healthy_pct = round((healthy_count / denom_pixel_count) * 100.0, 2)
    yellow_pct = round((yellow_count / denom_pixel_count) * 100.0, 2)
    brown_pct = round((brown_count / denom_pixel_count) * 100.0, 2)
    dark_pct = round((dark_count / denom_pixel_count) * 100.0, 2)

    color_layer = np.zeros_like(proc_img)
    color_layer[healthy_mask] = (0, 220, 0)
    color_layer[yellow_mask] = (0, 220, 220)
    color_layer[brown_mask] = (19, 69, 139)
    color_layer[dark_spots_mask] = (0, 0, 220)
    color_layer[holes_bool] = (255, 0, 255)
    color_layer[recon_bool] = (200, 180, 50)

    active_regions = healthy_mask | yellow_mask | brown_mask | dark_spots_mask | holes_bool | recon_bool
    alpha = 0.45
    blended = cv2.addWeighted(proc_img, 1.0 - alpha, color_layer, alpha, 0)
    overlay[active_regions] = blended[active_regions]

    stats = {
        "healthy_green_pct": healthy_pct,
        "yellowing_pct": yellow_pct,
        "browning_pct": brown_pct,
        "dark_spots_pct": dark_pct,
        "reconstructed_pct": reconstructed_pct,
        "holes_count": holes_count,
        "leaf_pixel_count": leaf_pixel_count,
        "valid_leaf_pixel_count": valid_leaf_count,
    }
    return stats, overlay


def find_worst_patch(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reconstructed_mask: Optional[np.ndarray] = None,
    window_sizes: tuple[int, ...] = PATCH_WINDOW_SIZES,
) -> dict[str, Any]:
    """Find bounding box of most discolored leaf region for agent zooming."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)

    if mask is None:
        mask, _ = segment_leaf(proc_img)
    else:
        validate_image(mask)
        if mask.shape[:2] != proc_img.shape[:2]:
            mask = cv2.resize(mask, (proc_img.shape[1], proc_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    leaf_binary = (mask > 127).astype(np.uint8)
    recon_bool = np.zeros(proc_img.shape[:2], dtype=bool)
    if reconstructed_mask is not None:
        if reconstructed_mask.shape[:2] != proc_img.shape[:2]:
            reconstructed_mask = cv2.resize(
                reconstructed_mask.astype(np.uint8),
                (proc_img.shape[1], proc_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        recon_bool = (reconstructed_mask > 0) & (leaf_binary > 0)

    leaf_eval_mask = (leaf_binary > 0) & (~recon_bool)
    height, width = proc_img.shape[:2]

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    dark_spots = leaf_eval_mask & (v_channel <= DARK_SPOT_VAL_MAX)
    yellowing = (
        leaf_eval_mask
        & (~dark_spots)
        & (h_channel >= YELLOW_HUE_MIN)
        & (h_channel <= YELLOW_HUE_MAX)
        & (s_channel >= YELLOW_SAT_MIN)
        & (v_channel >= YELLOW_VAL_MIN)
    )
    browning = (
        leaf_eval_mask
        & (~dark_spots)
        & (~yellowing)
        & (h_channel >= BROWN_HUE_MIN)
        & (h_channel <= BROWN_HUE_MAX)
        & (s_channel >= BROWN_SAT_MIN)
        & (s_channel <= BROWN_SAT_MAX)
        & (v_channel >= BROWN_VAL_MIN)
        & (v_channel <= BROWN_VAL_MAX)
    )

    discolored_mask = (yellowing | browning | dark_spots).astype(np.uint8)
    total_discolored = int(np.count_nonzero(discolored_mask))

    if total_discolored == 0:
        contours, _ = cv2.findContours(leaf_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return {
                "bbox": [int(x), int(y), int(w), int(h)],
                "discoloration_score": 0.0,
                "discolored_pixel_count": 0,
            }
        return {
            "bbox": [0, 0, width, height],
            "discoloration_score": 0.0,
            "discolored_pixel_count": 0,
        }

    integral_discolor = cv2.integral(discolored_mask)
    integral_leaf = cv2.integral(leaf_binary)

    best_bbox = [0, 0, width, height]
    best_score = -1.0
    best_count = 0

    for win_size in window_sizes:
        w_size = min(win_size, min(height, width))
        if w_size <= 0:
            continue
        step = max(8, w_size // 4)

        for y in range(0, height - w_size + 1, step):
            for x in range(0, width - w_size + 1, step):
                disc_sum = int(
                    integral_discolor[y + w_size, x + w_size]
                    - integral_discolor[y, x + w_size]
                    - integral_discolor[y + w_size, x]
                    + integral_discolor[y, x]
                )
                leaf_sum = int(
                    integral_leaf[y + w_size, x + w_size]
                    - integral_leaf[y, x + w_size]
                    - integral_leaf[y + w_size, x]
                    + integral_leaf[y, x]
                )

                if leaf_sum > 0:
                    score = float(disc_sum / float(w_size * w_size))
                    if score > best_score or (score == best_score and disc_sum > best_count):
                        best_score = score
                        best_count = disc_sum
                        best_bbox = [int(x), int(y), int(w_size), int(w_size)]

    return {
        "bbox": best_bbox,
        "discoloration_score": round(max(0.0, best_score), 4),
        "discolored_pixel_count": best_count,
    }


def _count_internal_holes(leaf_binary: np.ndarray) -> tuple[int, np.ndarray]:
    contours, hierarchy = cv2.findContours(leaf_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes_mask = np.zeros_like(leaf_binary)
    holes_count = 0

    if hierarchy is None or len(contours) == 0:
        return 0, holes_mask

    leaf_area = float(np.count_nonzero(leaf_binary))
    for i, h in enumerate(hierarchy[0]):
        parent_idx = h[3]
        if parent_idx != -1:
            area = cv2.contourArea(contours[i])
            if MIN_HOLE_AREA <= area <= (leaf_area * MAX_HOLE_AREA_RATIO):
                holes_count += 1
                cv2.drawContours(holes_mask, contours, i, 255, thickness=cv2.FILLED)

    return holes_count, holes_mask
