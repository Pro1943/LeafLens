import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leaflens.io_utils import load_image, save_image


def create_degraded_images(source_path: Path, output_dir: Path) -> None:
    """Read an image and create blurred, dark, overexposed, and glare variations."""
    img = load_image(source_path)
    base_name = source_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    blurred = cv2.GaussianBlur(img, (25, 25), 9.0)
    save_image(output_dir / f"{base_name}_blurred.png", blurred)

    dark = (img * 0.25).astype(np.uint8)
    save_image(output_dir / f"{base_name}_dark.png", dark)

    overexposed = np.clip(img.astype(np.float32) * 1.8 + 60, 0, 255).astype(np.uint8)
    save_image(output_dir / f"{base_name}_overexposed.png", overexposed)

    glare = img.copy()
    h, w = glare.shape[:2]
    center_x, center_y = w // 2, h // 2
    cv2.circle(glare, (center_x, center_y), min(w, h) // 6, (255, 255, 255), -1)
    glare = cv2.GaussianBlur(glare, (9, 9), 0)
    save_image(output_dir / f"{base_name}_glare.png", glare)

    from leaflens.segment import segment_leaf
    leaf_mask, _ = segment_leaf(img)
    leaf_bool = leaf_mask > 127

    combined = img.copy().astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    lighting_gradient = 0.55 + 0.65 * (1.0 - (xx / float(w) * 0.6 + yy / float(h) * 0.4))
    lighting_gradient = np.clip(lighting_gradient, 0.4, 1.15)
    for c in range(3):
        channel = combined[:, :, c]
        channel[leaf_bool] *= lighting_gradient[leaf_bool]
        combined[:, :, c] = channel

    leaf_y_indices, leaf_x_indices = np.where(leaf_bool)
    if len(leaf_x_indices) > 0:
        spot_x = int(np.percentile(leaf_x_indices, 35))
        spot_y = int(np.percentile(leaf_y_indices, 35))
    else:
        spot_x, spot_y = int(w * 0.4), int(h * 0.4)

    sigma_glare = max(15.0, min(w, h) * 0.10)
    dist_sq = (xx - spot_x) ** 2 + (yy - spot_y) ** 2
    glare_intensity = np.exp(-dist_sq / (2.0 * (sigma_glare ** 2)))
    for c in range(3):
        channel = combined[:, :, c]
        channel[leaf_bool] = (
            channel[leaf_bool] * (1.0 - glare_intensity[leaf_bool] * 0.75)
            + 255.0 * (glare_intensity[leaf_bool] * 0.75)
        )
        combined[:, :, c] = channel

    combined = np.clip(combined, 0, 255).astype(np.uint8)

    blur_kernel_size = max(15, (min(w, h) // 30) * 2 + 1)
    fully_blurred = cv2.GaussianBlur(combined, (blur_kernel_size, blur_kernel_size), 5.0)
    blur_weight = np.clip((xx / float(w) * 0.5 + yy / float(h) * 0.7), 0.0, 1.0)
    blur_weight = cv2.GaussianBlur(blur_weight.astype(np.float32), (25, 25), 0)
    blur_weight_3c = np.repeat(blur_weight[:, :, np.newaxis], 3, axis=2)

    combined_leaf = (
        combined.astype(np.float32) * (1.0 - blur_weight_3c * 0.8)
        + fully_blurred.astype(np.float32) * (blur_weight_3c * 0.8)
    )
    for c in range(3):
        channel = combined[:, :, c]
        channel[leaf_bool] = np.clip(combined_leaf[:, :, c][leaf_bool], 0, 255)
        combined[:, :, c] = channel

    save_image(output_dir / f"{base_name}_Degraded_Image.png", combined)
    save_image(output_dir / "Degraded Image.png", combined)


def process_folder(input_dir: str, output_dir: str) -> int:
    """Process all valid images in a folder and output degraded versions."""
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [p for p in in_path.iterdir() if p.suffix.lower() in valid_extensions]

    for p in image_files:
        create_degraded_images(p, out_path)

    return len(image_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create degraded image variations.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for degraded images")
    args = parser.parse_args()

    count = process_folder(args.input, args.output)
    print(f"Processed {count} images and generated degraded variants in {args.output}")


if __name__ == "__main__":
    main()
