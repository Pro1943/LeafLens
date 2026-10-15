import argparse
import json
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leaflens.discolor import find_worst_patch, measure_discoloration
from leaflens.enhance import auto_crop_to_leaf, fix_exposure, remove_glare, sharpen
from leaflens.io_utils import load_image, save_image
from leaflens.quality import quality_report
from leaflens.segment import segment_leaf


def run_all_tools(image_path: str, output_dir: str) -> None:
    """Run all tools on one image and write outputs to output_dir."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(image_path)
    base_name = Path(image_path).stem

    report = quality_report(img)
    print("Quality Report:")
    print(json.dumps(report, indent=2))

    with open(out_dir / f"{base_name}_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_image(out_dir / f"{base_name}_0_original.png", img)

    current_img = img.copy()
    mask, seg_info = segment_leaf(current_img)
    reconstructed_mask = None

    if "glare" in report["needs_fix"]:
        current_img, _, reconstructed_mask = remove_glare(current_img, leaf_mask=mask)
        save_image(out_dir / f"{base_name}_1_glare_removed.png", current_img)
        mask, seg_info = segment_leaf(current_img)

    if "dark" in report["needs_fix"] or "overexposed" in report["needs_fix"]:
        current_img, _ = fix_exposure(current_img)
        save_image(out_dir / f"{base_name}_2_exposure_fixed.png", current_img)

    if "blur" in report["needs_fix"]:
        current_img, _ = sharpen(current_img, amount=1.2)
        save_image(out_dir / f"{base_name}_3_sharpened.png", current_img)

    save_image(out_dir / f"{base_name}_Combined_Fixed_image.png", current_img)
    save_image(out_dir / "Combined Fixed image.png", current_img)

    mask, seg_info = segment_leaf(current_img)
    save_image(out_dir / f"{base_name}_4_mask.png", mask)

    cropped, crop_info = auto_crop_to_leaf(current_img)
    save_image(out_dir / f"{base_name}_5_cropped.png", cropped)

    discolor_stats, overlay = measure_discoloration(current_img, mask, reconstructed_mask=reconstructed_mask)
    save_image(out_dir / f"{base_name}_6_discoloration_overlay.png", overlay)

    worst_patch = find_worst_patch(current_img, mask, reconstructed_mask=reconstructed_mask)
    annotated_patch = overlay.copy()
    bx, by, bw, bh = worst_patch["bbox"]
    cv2.rectangle(annotated_patch, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
    save_image(out_dir / f"{base_name}_7_worst_patch.png", annotated_patch)

    if reconstructed_mask is not None:
        save_image(out_dir / f"{base_name}_8_reconstructed_mask.png", (reconstructed_mask.astype(np.uint8) * 255))
        report = quality_report(current_img, reconstructed_mask=reconstructed_mask)

    summary = {
        "quality_report": report,
        "segmentation": seg_info,
        "cropping": crop_info,
        "discoloration": discolor_stats,
        "worst_patch": worst_patch,
    }
    with open(out_dir / f"{base_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Toolkit run complete. Outputs saved in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LeafLens toolkit on a leaf image.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory (default: ./out)")
    args = parser.parse_args()

    run_all_tools(args.image, args.output)


if __name__ == "__main__":
    main()
