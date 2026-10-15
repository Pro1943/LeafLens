import argparse
import csv
from pathlib import Path
import sys
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent
from leaflens.config import HEALTHY_THRESHOLD_PCT, ISSUE_THRESHOLD_PCT
from leaflens.io_utils import load_image
from leaflens.tools import TOOLS, clear_store, register_image
from scripts.degrade import create_degraded_images


def run_pipeline_agent_disabled(img: np.ndarray) -> tuple[str, int]:
    clear_store()
    img_id = register_image(img)
    steps = 1

    try:
        seg_res = TOOLS["segment_leaf"]["fn"](img_id)
        steps += 1
        mask_id = seg_res["mask_image_id"]
        disc_res = TOOLS["measure_discoloration"]["fn"](img_id, mask_image_id=mask_id)
        steps += 1
        stats = disc_res["stats"]
        affected = (
            stats.get("yellowing_pct", 0.0)
            + stats.get("browning_pct", 0.0)
            + stats.get("dark_spots_pct", 0.0)
        )
        if affected < HEALTHY_THRESHOLD_PCT:
            verdict = "healthy"
        elif affected > ISSUE_THRESHOLD_PCT:
            verdict = "possible_issue"
        else:
            verdict = "uncertain"
        return verdict, steps
    except Exception:
        return "retake_needed", steps
    finally:
        clear_store()


def run_pipeline_full_agent(img: np.ndarray) -> tuple[str, int]:
    clear_store()
    try:
        response = run_agent(img)
        return response.verdict, len(response.trace)
    finally:
        clear_store()


def evaluate_folder(input_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir = output_dir / "degraded_eval"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]

    variant_keys = ["blur", "dark", "overexposed", "glare", "combined"]
    records: dict[str, list[dict[str, Any]]] = {k: [] for k in variant_keys}

    for img_path in images:
        base_name = img_path.stem
        create_degraded_images(img_path, degraded_dir)

        clean_img = load_image(img_path)
        clean_verdict, _ = run_pipeline_full_agent(clean_img)

        variant_files = {
            "blur": degraded_dir / f"{base_name}_blurred.png",
            "dark": degraded_dir / f"{base_name}_dark.png",
            "overexposed": degraded_dir / f"{base_name}_overexposed.png",
            "glare": degraded_dir / f"{base_name}_glare.png",
            "combined": degraded_dir / f"{base_name}_Degraded_Image.png",
        }

        for var_name, var_file in variant_files.items():
            if not var_file.exists():
                continue
            deg_img = load_image(var_file)

            off_verdict, off_steps = run_pipeline_agent_disabled(deg_img)
            on_verdict, on_steps = run_pipeline_full_agent(deg_img)

            records[var_name].append({
                "clean_verdict": clean_verdict,
                "off_verdict": off_verdict,
                "off_steps": off_steps,
                "on_verdict": on_verdict,
                "on_steps": on_steps,
            })

    summary_rows = []
    for var_name in variant_keys:
        items = records[var_name]
        total = len(items)
        if total == 0:
            continue

        off_matches = sum(1 for it in items if it["off_verdict"] == it["clean_verdict"])
        on_matches = sum(1 for it in items if it["on_verdict"] == it["clean_verdict"])
        off_retakes = sum(1 for it in items if it["off_verdict"] == "retake_needed")
        on_retakes = sum(1 for it in items if it["on_verdict"] == "retake_needed")

        avg_off_steps = sum(it["off_steps"] for it in items) / total
        avg_on_steps = sum(it["on_steps"] for it in items) / total

        row = {
            "degradation_type": var_name,
            "images_count": total,
            "match_share_agent_off": round(off_matches / total, 4),
            "match_share_agent_on": round(on_matches / total, 4),
            "avg_steps_agent_off": round(avg_off_steps, 2),
            "avg_steps_agent_on": round(avg_on_steps, 2),
            "retake_share_agent_off": round(off_retakes / total, 4),
            "retake_share_agent_on": round(on_retakes / total, 4),
        }
        summary_rows.append(row)

    csv_path = output_dir / "eval_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "degradation_type",
            "images_count",
            "match_share_agent_off",
            "match_share_agent_on",
            "avg_steps_agent_off",
            "avg_steps_agent_on",
            "retake_share_agent_off",
            "retake_share_agent_on",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\n" + "=" * 90)
    print(f"{'Degradation':<12} | {'Off Match':<10} | {'On Match':<10} | {'Off Steps':<10} | {'On Steps':<10} | {'Off Retake':<10} | {'On Retake':<10}")
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['degradation_type']:<12} | "
            f"{r['match_share_agent_off']:<10.2%} | "
            f"{r['match_share_agent_on']:<10.2%} | "
            f"{r['avg_steps_agent_off']:<10.1f} | "
            f"{r['avg_steps_agent_on']:<10.1f} | "
            f"{r['retake_share_agent_off']:<10.2%} | "
            f"{r['retake_share_agent_on']:<10.2%}"
        )
    print("=" * 90)
    print(f"Summary CSV written to: {csv_path}\n")

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LeafLens agent on degraded images.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "samples"),
        help="Input directory of clean images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "out"),
        help="Output directory for evaluations and CSV",
    )
    args = parser.parse_args()

    evaluate_folder(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
