from typing import Any, Optional
import cv2
import numpy as np

from leaflens.config import (
    DEFAULT_CROP_PADDING,
    DEFAULT_SHARPEN_AMOUNT,
    EXPOSURE_MEAN_DARK_THRESHOLD,
    GLARE_INPAINT_RADIUS,
    MAX_IMAGE_DIMENSION,
)
from leaflens.io_utils import downscale_if_needed, validate_image
from leaflens.quality import measure_glare
from leaflens.segment import segment_leaf


def fix_exposure(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fix exposure using CLAHE on L channel in Lab, with gamma correction when too dark."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    if len(proc_img.shape) == 2:
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(proc_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    mean_l = float(np.mean(l_channel))
    applied_gamma = 1.0

    if mean_l < EXPOSURE_MEAN_DARK_THRESHOLD:
        applied_gamma = round(max(0.4, min(0.8, mean_l / 100.0)), 2)
        table = np.array([((i / 255.0) ** applied_gamma) * 255 for i in range(256)]).astype(np.uint8)
        l_channel = cv2.LUT(l_channel, table)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    info = {
        "clip_limit": clip_limit,
        "tile_grid_size": list(tile_grid_size),
        "initial_mean_l": round(mean_l, 2),
        "gamma_correction": applied_gamma,
        "gamma_applied": applied_gamma != 1.0,
    }
    return result, info


def sharpen(
    img: np.ndarray,
    amount: float = DEFAULT_SHARPEN_AMOUNT,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sharpen image using an unsharp mask single-pass filter."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)

    blurred = cv2.GaussianBlur(proc_img, kernel_size, sigma)
    sharpened = cv2.addWeighted(proc_img, 1.0 + amount, blurred, -amount, 0)

    info = {
        "amount": amount,
        "kernel_size": list(kernel_size),
        "sigma": sigma,
    }
    return sharpened, info


def remove_glare(
    img: np.ndarray,
    glare_mask: Optional[np.ndarray] = None,
    leaf_mask: Optional[np.ndarray] = None,
    inpaint_radius: int = GLARE_INPAINT_RADIUS,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Inpaint glare areas and restore color, returning repaired image, info, and boolean reconstructed mask."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)

    if glare_mask is None:
        if leaf_mask is None:
            leaf_mask, _ = segment_leaf(proc_img)
        _, detected_mask = measure_glare(proc_img, leaf_mask=leaf_mask)
    else:
        detected_mask = glare_mask

    if len(detected_mask.shape) == 3:
        detected_mask = cv2.cvtColor(detected_mask, cv2.COLOR_BGR2GRAY)

    glare_pixels = int(np.count_nonzero(detected_mask))
    if glare_pixels == 0:
        empty_reconstructed = np.zeros(proc_img.shape[:2], dtype=bool)
        return proc_img.copy(), {
            "glare_pixels_repaired": 0,
            "inpaint_method": "none",
            "reconstructed_pixel_count": 0,
        }, empty_reconstructed

    inpainted = cv2.inpaint(proc_img, detected_mask, inpaint_radius, cv2.INPAINT_TELEA)

    if leaf_mask is not None:
        if leaf_mask.shape[:2] != proc_img.shape[:2]:
            leaf_mask = cv2.resize(leaf_mask, (proc_img.shape[1], proc_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(leaf_mask.shape) == 3:
            leaf_mask = cv2.cvtColor(leaf_mask, cv2.COLOR_BGR2GRAY)
        leaf_bool = leaf_mask > 127
    else:
        leaf_bool = np.ones(proc_img.shape[:2], dtype=bool)

    hsv_inpaint = cv2.cvtColor(inpainted, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_channel = hsv_inpaint[:, :, 1]
    val_channel = hsv_inpaint[:, :, 2]

    flare_veil = np.clip((val_channel - 140.0) / 100.0, 0.0, 1.0) * np.clip((180.0 - sat_channel) / 140.0, 0.0, 1.0)
    flare_veil[~leaf_bool] = 0.0
    flare_veil = cv2.GaussianBlur(flare_veil, (31, 31), 0)

    healthy_leaf_bool = leaf_bool & (flare_veil < 0.15) & (val_channel < 180) & (sat_channel > 120)

    lab_img = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = lab_img[:, :, 0], lab_img[:, :, 1], lab_img[:, :, 2]

    if np.count_nonzero(healthy_leaf_bool) > 0:
        target_l = float(np.median(l_ch[healthy_leaf_bool]))
        target_a = float(np.median(a_ch[healthy_leaf_bool]))
        target_b = float(np.median(b_ch[healthy_leaf_bool]))
    else:
        target_l = 125.0
        target_a = 105.0
        target_b = 155.0

    veil_weight = np.clip(flare_veil * 0.90, 0.0, 1.0)
    restored_l = l_ch * (1.0 - veil_weight * 0.65) + target_l * (veil_weight * 0.65)
    restored_a = a_ch * (1.0 - veil_weight) + target_a * veil_weight
    restored_b = b_ch * (1.0 - veil_weight) + target_b * veil_weight

    restored_lab = np.stack([restored_l, restored_a, restored_b], axis=-1)
    restored_lab = np.clip(restored_lab, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2BGR)

    pixel_diff = np.any(np.abs(restored_bgr.astype(np.int16) - proc_img.astype(np.int16)) > 2, axis=2)
    reconstructed_mask = (pixel_diff | (detected_mask > 0)) & leaf_bool

    info = {
        "glare_pixels_repaired": glare_pixels,
        "inpaint_method": "telea_with_lab_color_restoration",
        "inpaint_radius": inpaint_radius,
        "reconstructed_pixel_count": int(np.count_nonzero(reconstructed_mask)),
    }
    return restored_bgr, info, reconstructed_mask


def auto_crop_to_leaf(
    img: np.ndarray,
    padding: int = DEFAULT_CROP_PADDING,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop image to bounding box of the largest leaf contour with optional padding."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    mask, _ = segment_leaf(proc_img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = proc_img.shape[:2]

    if not contours:
        return proc_img.copy(), {
            "cropped": False,
            "bbox": [0, 0, width, height],
            "padding": padding,
        }

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    cropped = proc_img[y1:y2, x1:x2].copy()
    return cropped, {
        "cropped": True,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "padding": padding,
        "original_shape": [height, width],
        "cropped_shape": [y2 - y1, x2 - x1],
    }
