from typing import Any, Callable, Optional
import uuid
import numpy as np

from leaflens.discolor import find_worst_patch, measure_discoloration
from leaflens.enhance import auto_crop_to_leaf, fix_exposure, remove_glare, sharpen
from leaflens.io_utils import downscale_if_needed, validate_image
from leaflens.quality import quality_report
from leaflens.segment import segment_leaf

_IMAGE_STORE: dict[str, np.ndarray] = {}


def register_image(img: np.ndarray, image_id: Optional[str] = None) -> str:
    """Store an image in memory and return its unique image_id."""
    validate_image(img)
    proc_img, _ = downscale_if_needed(img)
    gid = image_id or str(uuid.uuid4())
    _IMAGE_STORE[gid] = proc_img
    return gid


def get_image(image_id: str) -> np.ndarray:
    """Retrieve an image from memory by image_id."""
    if image_id not in _IMAGE_STORE:
        raise KeyError(f"Image ID '{image_id}' not found in store")
    return _IMAGE_STORE[image_id]


def clear_store() -> None:
    """Clear all images in memory store."""
    _IMAGE_STORE.clear()


def tool_check_quality(image_id: str) -> dict[str, Any]:
    """Run full quality check on image."""
    img = get_image(image_id)
    report = quality_report(img)
    return {"image_id": image_id, "quality_report": report}


def tool_fix_exposure(image_id: str, clip_limit: float = 2.0) -> dict[str, Any]:
    """Fix exposure on stored image and return new image_id."""
    img = get_image(image_id)
    fixed, info = fix_exposure(img, clip_limit=clip_limit)
    new_id = register_image(fixed)
    return {"new_image_id": new_id, "info": info}


def tool_sharpen(image_id: str, amount: float = 1.0) -> dict[str, Any]:
    """Sharpen stored image and return new image_id."""
    img = get_image(image_id)
    sharpened, info = sharpen(img, amount=amount)
    new_id = register_image(sharpened)
    return {"new_image_id": new_id, "info": info}


def tool_remove_glare(image_id: str) -> dict[str, Any]:
    """Inpaint glare on stored image and return new image_id and reconstructed_mask_id."""
    img = get_image(image_id)
    repaired, info, recon_mask = remove_glare(img)
    new_id = register_image(repaired)
    recon_id = register_image((recon_mask.astype(np.uint8) * 255))
    return {"new_image_id": new_id, "info": info, "reconstructed_mask_id": recon_id}


def tool_crop_to_leaf(image_id: str, padding: int = 20) -> dict[str, Any]:
    """Crop stored image to detected leaf bounding box."""
    img = get_image(image_id)
    cropped, info = auto_crop_to_leaf(img, padding=padding)
    new_id = register_image(cropped)
    return {"new_image_id": new_id, "info": info}


def tool_segment_leaf(image_id: str) -> dict[str, Any]:
    """Segment leaf and store resulting mask as an image_id."""
    img = get_image(image_id)
    mask, info = segment_leaf(img)
    mask_id = register_image(mask)
    return {"mask_image_id": mask_id, "info": info}


def tool_measure_discoloration(
    image_id: str,
    mask_image_id: Optional[str] = None,
    reconstructed_mask_id: Optional[str] = None,
) -> dict[str, Any]:
    """Measure leaf discoloration excluding reconstructed pixels and return metrics with overlay image_id."""
    img = get_image(image_id)
    mask = get_image(mask_image_id) if mask_image_id else None
    recon_mask = get_image(reconstructed_mask_id) if reconstructed_mask_id else None
    stats, overlay = measure_discoloration(img, mask=mask, reconstructed_mask=recon_mask)
    overlay_id = register_image(overlay)
    return {"stats": stats, "overlay_image_id": overlay_id}


def tool_find_worst_patch(
    image_id: str,
    mask_image_id: Optional[str] = None,
    reconstructed_mask_id: Optional[str] = None,
) -> dict[str, Any]:
    """Find bounding box of worst discolored leaf patch."""
    img = get_image(image_id)
    mask = get_image(mask_image_id) if mask_image_id else None
    recon_mask = get_image(reconstructed_mask_id) if reconstructed_mask_id else None
    result = find_worst_patch(img, mask=mask, reconstructed_mask=recon_mask)
    return {"worst_patch": result}


TOOLS: dict[str, dict[str, Any]] = {
    "check_quality": {
        "fn": tool_check_quality,
        "description": "Analyze image quality including blur, exposure, glare, and leaf coverage.",
        "params": {
            "type": "object",
            "properties": {"image_id": {"type": "string", "description": "ID of image in store"}},
            "required": ["image_id"],
        },
    },
    "fix_exposure": {
        "fn": tool_fix_exposure,
        "description": "Adjust contrast and exposure using CLAHE and gamma correction.",
        "params": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of image in store"},
                "clip_limit": {"type": "number", "default": 2.0, "description": "CLAHE clip limit"},
            },
            "required": ["image_id"],
        },
    },
    "sharpen": {
        "fn": tool_sharpen,
        "description": "Apply unsharp mask filter to sharpen image details.",
        "params": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of image in store"},
                "amount": {"type": "number", "default": 1.0, "description": "Sharpening intensity amount"},
            },
            "required": ["image_id"],
        },
    },
    "remove_glare": {
        "fn": tool_remove_glare,
        "description": "Detect and inpaint saturated glare reflections on the leaf.",
        "params": {
            "type": "object",
            "properties": {"image_id": {"type": "string", "description": "ID of image in store"}},
            "required": ["image_id"],
        },
    },
    "crop_to_leaf": {
        "fn": tool_crop_to_leaf,
        "description": "Crop image to leaf bounding box with margin padding.",
        "params": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of image in store"},
                "padding": {"type": "integer", "default": 20, "description": "Pixel padding around leaf"},
            },
            "required": ["image_id"],
        },
    },
    "segment_leaf": {
        "fn": tool_segment_leaf,
        "description": "Generate binary segmentation mask of the leaf.",
        "params": {
            "type": "object",
            "properties": {"image_id": {"type": "string", "description": "ID of image in store"}},
            "required": ["image_id"],
        },
    },
    "measure_discoloration": {
        "fn": tool_measure_discoloration,
        "description": "Calculate yellowing, browning, dark spots, holes, and return overlay image excluding reconstructed pixels.",
        "params": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of leaf image in store"},
                "mask_image_id": {
                    "type": "string",
                    "description": "Optional ID of leaf binary mask in store",
                },
                "reconstructed_mask_id": {
                    "type": "string",
                    "description": "Optional ID of reconstructed area mask in store",
                },
            },
            "required": ["image_id"],
        },
    },
    "find_worst_patch": {
        "fn": tool_find_worst_patch,
        "description": "Locate bounding box [x, y, w, h] of most discolored leaf patch.",
        "params": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of leaf image in store"},
                "mask_image_id": {
                    "type": "string",
                    "description": "Optional ID of leaf binary mask in store",
                },
                "reconstructed_mask_id": {
                    "type": "string",
                    "description": "Optional ID of reconstructed area mask in store",
                },
            },
            "required": ["image_id"],
        },
    },
}
