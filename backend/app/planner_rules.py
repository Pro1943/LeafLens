from typing import Any


def plan_next_action(
    quality_report: dict[str, Any],
    fixes_applied: list[str],
    current_image_id: str,
) -> dict[str, Any]:
    needs_fix = quality_report.get("needs_fix", [])
    if ("dark" in needs_fix or "overexposed" in needs_fix) and "fix_exposure" not in fixes_applied:
        return {
            "tool": "fix_exposure",
            "args": {"clip_limit": 2.0},
            "reason": "Image exposure requires correction to normalize luminance distribution.",
        }
    if "glare" in needs_fix and "remove_glare" not in fixes_applied:
        return {
            "tool": "remove_glare",
            "args": {},
            "reason": "Glare detected across leaf surface requiring inpainting and color reconstruction.",
        }
    if "blur" in needs_fix and "sharpen" not in fixes_applied:
        return {
            "tool": "sharpen",
            "args": {"amount": 1.0},
            "reason": "Image details are blurry and require unsharp masking enhancement.",
        }
    if "coverage" in needs_fix and "crop_to_leaf" not in fixes_applied:
        return {
            "tool": "crop_to_leaf",
            "args": {"padding": 20},
            "reason": "Leaf fills too little of the frame and requires tight bounding cropping.",
        }
    return {
        "tool": "finish",
        "args": {},
        "reason": "Image quality is acceptable for leaf segmentation and discoloration measurement.",
    }
