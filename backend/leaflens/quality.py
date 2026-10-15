from typing import Any, Optional
import cv2
import numpy as np

from leaflens.config import (
    BLUR_THRESHOLD,
    COVERAGE_MIN_RATIO,
    EXPOSURE_BRIGHT_RATIO_THRESHOLD,
    EXPOSURE_DARK_RATIO_THRESHOLD,
    EXPOSURE_MEAN_BRIGHT_THRESHOLD,
    EXPOSURE_MEAN_DARK_THRESHOLD,
    GLARE_RATIO_THRESHOLD,
    GLARE_SATURATION_MAX,
    GLARE_VALUE_MIN,
    MAX_IMAGE_DIMENSION,
    RECONSTRUCTED_RATIO_THRESHOLD,
)
from leaflens.io_utils import downscale_if_needed, validate_image
from leaflens.segment import segment_leaf


def measure_blur(img: np.ndarray) -> dict[str, Any]:
    """Measure image blur using the variance of the Laplacian on grayscale."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    if len(proc_img.shape) == 3:
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = proc_img

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    is_blurry = bool(variance < BLUR_THRESHOLD)
    return {
        "blur_score": round(variance, 2),
        "is_blurry": is_blurry,
    }


def measure_exposure(img: np.ndarray) -> dict[str, Any]:
    """Measure mean brightness and histogram clipping ratios."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    if len(proc_img.shape) == 3:
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = proc_img

    mean_val = float(np.mean(gray))
    total_pixels = float(gray.size)
    dark_pixels = float(np.count_nonzero(gray <= 20))
    bright_pixels = float(np.count_nonzero(gray >= 235))

    dark_ratio = float(dark_pixels / total_pixels)
    bright_ratio = float(bright_pixels / total_pixels)

    if mean_val < EXPOSURE_MEAN_DARK_THRESHOLD or dark_ratio > EXPOSURE_DARK_RATIO_THRESHOLD:
        status = "dark"
    elif mean_val > EXPOSURE_MEAN_BRIGHT_THRESHOLD or bright_ratio > EXPOSURE_BRIGHT_RATIO_THRESHOLD:
        status = "overexposed"
    else:
        status = "ok"

    return {
        "mean": round(mean_val, 2),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "status": status,
    }


def measure_glare(img: np.ndarray, leaf_mask: Optional[np.ndarray] = None) -> tuple[dict[str, Any], np.ndarray]:
    """Measure ratio of near-saturated, low-saturation pixels and return glare mask."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    if len(proc_img.shape) == 2:
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    raw_glare = (sat <= GLARE_SATURATION_MAX) & (val >= GLARE_VALUE_MIN)

    if leaf_mask is not None:
        if leaf_mask.shape[:2] != proc_img.shape[:2]:
            leaf_mask = cv2.resize(leaf_mask, (proc_img.shape[1], proc_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(leaf_mask.shape) == 3:
            leaf_mask = cv2.cvtColor(leaf_mask, cv2.COLOR_BGR2GRAY)
        leaf_binary = leaf_mask > 127
        glare_binary = raw_glare & leaf_binary
        denom = float(np.count_nonzero(leaf_binary))
        glare_ratio = float(np.count_nonzero(glare_binary) / denom) if denom > 0 else 0.0
    else:
        glare_binary = raw_glare
        glare_ratio = float(np.count_nonzero(glare_binary) / glare_binary.size)

    glare_mask = glare_binary.astype(np.uint8) * 255
    has_glare = bool(glare_ratio >= GLARE_RATIO_THRESHOLD)

    return {
        "glare_ratio": round(glare_ratio, 4),
        "has_glare": has_glare,
    }, glare_mask


def measure_leaf_coverage(img: np.ndarray) -> dict[str, Any]:
    """Measure fraction of the frame marked as leaf by segment_leaf."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)
    mask, _ = segment_leaf(proc_img)
    coverage = float(np.count_nonzero(mask) / mask.size)
    too_small = bool(coverage < COVERAGE_MIN_RATIO)
    return {
        "coverage": round(coverage, 4),
        "too_small": too_small,
    }


def quality_report(
    img: np.ndarray,
    reconstructed_mask: Optional[np.ndarray] = None,
    reconstructed_pct: Optional[float] = None,
) -> dict[str, Any]:
    """Run all quality measurements and return a combined report with recommendations."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img, MAX_IMAGE_DIMENSION)

    mask, _ = segment_leaf(proc_img)
    coverage_val = float(np.count_nonzero(mask) / mask.size)
    too_small = bool(coverage_val < COVERAGE_MIN_RATIO)
    coverage_res = {
        "coverage": round(coverage_val, 4),
        "too_small": too_small,
    }

    blur_res = measure_blur(proc_img)
    exposure_res = measure_exposure(proc_img)
    glare_res, _ = measure_glare(proc_img, leaf_mask=mask if not too_small else None)

    if reconstructed_pct is None and reconstructed_mask is not None:
        leaf_count = float(np.count_nonzero(mask))
        if leaf_count > 0:
            recon_count = float(np.count_nonzero((reconstructed_mask > 0) & (mask > 0)))
            reconstructed_pct = round((recon_count / leaf_count) * 100.0, 2)
        else:
            reconstructed_pct = 0.0

    needs_fix: list[str] = []
    if blur_res["is_blurry"]:
        needs_fix.append("blur")
    if exposure_res["status"] == "dark":
        needs_fix.append("dark")
    elif exposure_res["status"] == "overexposed":
        needs_fix.append("overexposed")
    if glare_res["has_glare"]:
        needs_fix.append("glare")
    if coverage_res["too_small"]:
        needs_fix.append("coverage")
    if reconstructed_pct is not None and (reconstructed_pct / 100.0) >= RECONSTRUCTED_RATIO_THRESHOLD:
        needs_fix.append("excessive_reconstruction")

    report_dict: dict[str, Any] = {
        "blur": blur_res,
        "exposure": exposure_res,
        "glare": glare_res,
        "coverage": coverage_res,
        "needs_fix": needs_fix,
        "is_acceptable": len(needs_fix) == 0,
    }
    if reconstructed_pct is not None:
        report_dict["reconstructed"] = {
            "reconstructed_pct": reconstructed_pct,
            "threshold_pct": round(RECONSTRUCTED_RATIO_THRESHOLD * 100.0, 2),
            "is_excessive": (reconstructed_pct / 100.0) >= RECONSTRUCTED_RATIO_THRESHOLD,
        }
    return report_dict
