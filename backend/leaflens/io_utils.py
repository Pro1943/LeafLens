from pathlib import Path
from typing import Union
import cv2
import numpy as np

from leaflens.config import MAX_IMAGE_DIMENSION


def validate_image(img: np.ndarray) -> np.ndarray:
    """Validate that input is a non-empty numpy image array."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Invalid or empty image provided")
    if len(img.shape) not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got shape {img.shape}")
    return img


def downscale_if_needed(img: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> tuple[np.ndarray, float]:
    """Downscale image so its longest dimension does not exceed max_dim."""
    validate_image(img)
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return img.copy(), 1.0

    scale = max_dim / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def load_image(path: Union[str, Path], max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Load an image from disk in BGR format, downscaling if longest side exceeds max_dim."""
    resolved_path = str(Path(path).expanduser().resolve())
    img = cv2.imread(resolved_path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to read image from path: {resolved_path}")
    resized, _ = downscale_if_needed(img, max_dim=max_dim)
    return resized


def save_image(path: Union[str, Path], img: np.ndarray) -> str:
    """Save an image array to disk."""
    validate_image(img)
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_path), img)
    if not success:
        raise IOError(f"Failed to save image to path: {out_path}")
    return str(out_path)


def encode_image_to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    """Encode an image array to JPEG bytes."""
    validate_image(img)
    success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buffer.tobytes()


def decode_image_from_bytes(data: bytes, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Decode an image from raw bytes, downscaling if needed."""
    if not data:
        raise ValueError("Empty image bytes supplied")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Failed to decode image from bytes")
    resized, _ = downscale_if_needed(img, max_dim=max_dim)
    return resized
