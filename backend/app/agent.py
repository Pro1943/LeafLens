import base64
import time
from typing import Any, Callable, Optional
import cv2
import numpy as np

from app.interpret import interpret
from app.planner_llm import plan_next_action as plan_next_action_llm
from app.planner_rules import plan_next_action as plan_next_action_rules
from app.schemas import (
    AnalyzeResponse,
    DiscolorationMetrics,
    Images,
    QualityMetrics,
    TraceStep,
    UserAction,
    Verdict,
)
from app.trace import TraceRecorder
from leaflens.config import UNCERTAIN_BAND
from leaflens.io_utils import downscale_if_needed
from leaflens.quality import quality_report
from leaflens.tools import TOOLS, get_image, register_image


class AgentState:
    def __init__(self, original_img: np.ndarray) -> None:
        self.original_img: np.ndarray = original_img
        self.current_img: np.ndarray = original_img
        self.overlay_img: np.ndarray = original_img
        self.quality_before: Optional[QualityMetrics] = None
        self.quality_after: Optional[QualityMetrics] = None
        self.fixes_applied: list[str] = []
        self.discoloration: Optional[DiscolorationMetrics] = None
        self.trace: list[TraceStep] = []
        self.verdict: Verdict = "uncertain"
        self.fallback_planner_used: bool = False
        self.user_action: Optional[UserAction] = None


def encode_image_to_data_url(img: np.ndarray, quality: int = 85, max_dim: int = 1024) -> str:
    proc_img, _ = downscale_if_needed(img, max_dim=max_dim)
    success, buf = cv2.imencode(".jpg", proc_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise ValueError("Failed to encode image to data URL")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_quality_metrics(report: dict[str, Any]) -> QualityMetrics:
    blur = report.get("blur", {})
    exposure = report.get("exposure", {})
    glare = report.get("glare", {})
    coverage = report.get("coverage", {})
    return QualityMetrics(
        blur_score=float(blur.get("blur_score", 0.0)),
        mean_brightness=float(exposure.get("mean", 0.0)),
        glare_ratio=float(glare.get("glare_ratio", 0.0)),
        coverage=float(coverage.get("coverage", 0.0)),
        needs_fix=list(report.get("needs_fix", [])),
    )


def _summarize_quality(report: dict[str, Any]) -> str:
    b = report.get("blur", {}).get("blur_score", 0.0)
    e = report.get("exposure", {}).get("mean", 0.0)
    g = round(float(report.get("glare", {}).get("glare_ratio", 0.0)) * 100.0, 1)
    c = round(float(report.get("coverage", {}).get("coverage", 0.0)) * 100.0, 1)
    needs = report.get("needs_fix", [])
    issues = ", ".join(needs) if needs else "none"
    return f"Quality checked: blur={b}, mean_l={e}, glare={g}%, coverage={c}%. Issues: {issues}."


def _build_human_readable_result(
    tool_name: str,
    res: dict[str, Any],
    reconstructed_pct: float,
) -> str:
    info = res.get("info", {})
    if tool_name == "fix_exposure":
        return f"Exposure adjusted (clip limit {info.get('clip_limit', 2.0)}, gamma applied {info.get('gamma_applied', False)})."
    if tool_name == "sharpen":
        return f"Sharpened image using unsharp mask filter (amount {info.get('amount', 1.0)})."
    if tool_name == "remove_glare":
        recon_share = round(reconstructed_pct)
        return f"Specular glare removed and inpainted. {recon_share}% of the leaf was reconstructed and excluded from measurements."
    if tool_name == "crop_to_leaf":
        return f"Cropped to leaf bounding box with {info.get('padding', 20)}px margin."
    if tool_name == "segment_leaf":
        return f"Leaf segmented using {info.get('method', 'algorithm')} covering {round(info.get('mask_area_ratio', 0.0) * 100.0, 1)}% of frame."
    if tool_name == "measure_discoloration":
        stats = res.get("stats", {})
        recon_share = round(float(stats.get("reconstructed_pct", 0.0)))
        return (
            f"Discoloration measured (yellowing: {stats.get('yellowing_pct', 0.0)}%, "
            f"browning: {stats.get('browning_pct', 0.0)}%, dark spots: {stats.get('dark_spots_pct', 0.0)}%). "
            f"{recon_share}% of the leaf was reconstructed and excluded from measurements."
        )
    if tool_name == "find_worst_patch":
        patch = res.get("worst_patch", {})
        return f"Worst patch located at bbox {patch.get('bbox', [])} with score {patch.get('discoloration_score', 0.0)}."
    return "Operation completed."


def run_agent(
    img: np.ndarray,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    max_steps: int = 8,
    state: Optional[AgentState] = None,
    planner_override: Optional[Callable[..., dict[str, Any]]] = None,
    interpreter_override: Optional[Callable[..., Any]] = None,
) -> AnalyzeResponse:
    if state is None:
        state = AgentState(img)

    recorder = TraceRecorder()
    original_id = register_image(img)
    current_image_id = original_id
    reconstructed_mask_id: Optional[str] = None
    mask_id: Optional[str] = None
    overlay_id: Optional[str] = None

    fixes_applied: list[str] = []
    fallback_planner_used: bool = False
    last_tool_call: Optional[tuple[str, frozenset]] = None
    retake_message: Optional[str] = None
    verdict: Optional[Verdict] = None

    t0 = time.perf_counter()
    q_initial_res = TOOLS["check_quality"]["fn"](current_image_id)
    dur_q0 = int((time.perf_counter() - t0) * 1000)
    initial_report = q_initial_res["quality_report"]
    latest_report = initial_report

    recorder.record(
        tool="check_quality",
        reason="Initial quality check before planning fixes",
        result=_summarize_quality(initial_report),
        status="ok",
        duration_ms=dur_q0,
    )

    quality_before = _extract_quality_metrics(initial_report)
    quality_after = quality_before

    state.quality_before = quality_before
    state.quality_after = quality_after
    state.trace = recorder.get_steps()

    if "excessive_reconstruction" in initial_report.get("needs_fix", []):
        verdict = "retake_needed"
        retake_message = "Too much of the leaf was hidden by glare. Please retake the photo from a different angle to avoid direct glare."
    elif float(initial_report.get("coverage", {}).get("coverage", 0.0)) == 0.0:
        verdict = "retake_needed"
        retake_message = "Could not detect a leaf in the photo. Please frame a single leaf against a contrasting background."

    if verdict is None:
        for _ in range(max_steps):
            if latest_report.get("is_acceptable", False):
                break

            action: dict[str, Any]
            if planner_override:
                action = planner_override(latest_report, fixes_applied, current_image_id)
            elif not fallback_planner_used and api_key:
                try:
                    action = plan_next_action_llm(
                        latest_report,
                        fixes_applied,
                        current_image_id,
                        api_key,
                        model_name,
                    )
                except Exception:
                    fallback_planner_used = True
                    state.fallback_planner_used = True
                    recorder.record(
                        tool="planner",
                        reason="LLM planner call failed or timed out",
                        result="Switching to rules-based fallback planner for remaining steps",
                        status="warning",
                        duration_ms=0,
                    )
                    action = plan_next_action_rules(
                        latest_report,
                        fixes_applied,
                        current_image_id,
                    )
            else:
                action = plan_next_action_rules(
                    latest_report,
                    fixes_applied,
                    current_image_id,
                )

            tool_name = action.get("tool", "finish")
            reason = action.get("reason", "")
            raw_args = dict(action.get("args") or {})

            if tool_name == "finish":
                break

            if tool_name not in TOOLS:
                recorder.record(
                    tool=str(tool_name),
                    reason=reason,
                    result=f"Tool '{tool_name}' is not in allowed tools registry",
                    status="error",
                    duration_ms=0,
                )
                continue

            current_call_tuple = (tool_name, frozenset(raw_args.items()))
            if current_call_tuple == last_tool_call:
                recorder.record(
                    tool=tool_name,
                    reason=reason,
                    result=f"Tool '{tool_name}' with identical arguments was run twice in a row",
                    status="warning",
                    duration_ms=0,
                )
                continue

            if tool_name in fixes_applied:
                recorder.record(
                    tool=tool_name,
                    reason=reason,
                    result=f"Fix '{tool_name}' has already been applied to this image",
                    status="warning",
                    duration_ms=0,
                )
                continue

            last_tool_call = current_call_tuple
            exec_args = dict(raw_args)
            exec_args["image_id"] = current_image_id

            t_tool = time.perf_counter()
            try:
                fn = TOOLS[tool_name]["fn"]
                res = fn(**exec_args)
                tool_dur = int((time.perf_counter() - t_tool) * 1000)
            except Exception as err:
                tool_dur = int((time.perf_counter() - t_tool) * 1000)
                recorder.record(
                    tool=tool_name,
                    reason=reason,
                    result=f"Tool execution failed: {str(err)}",
                    status="error",
                    duration_ms=tool_dur,
                )
                continue

            if tool_name in ("fix_exposure", "sharpen", "remove_glare", "crop_to_leaf"):
                fixes_applied.append(tool_name)
                state.fixes_applied = list(fixes_applied)

            if "new_image_id" in res:
                current_image_id = res["new_image_id"]
                state.current_img = get_image(current_image_id)

            recon_pct_step = 0.0
            if "reconstructed_mask_id" in res:
                reconstructed_mask_id = res["reconstructed_mask_id"]
                recon_mask = get_image(reconstructed_mask_id)
                current_img_arr = get_image(current_image_id)
                leaf_total = float(np.count_nonzero(current_img_arr))
                if leaf_total > 0:
                    recon_pct_step = (float(np.count_nonzero(recon_mask > 0)) / leaf_total) * 100.0

            human_res = _build_human_readable_result(tool_name, res, recon_pct_step)
            recorder.record(
                tool=tool_name,
                reason=reason,
                result=human_res,
                status="ok",
                duration_ms=tool_dur,
            )

            t_q = time.perf_counter()
            q_res = TOOLS["check_quality"]["fn"](
                current_image_id,
                reconstructed_mask_id=reconstructed_mask_id,
            )
            q_dur = int((time.perf_counter() - t_q) * 1000)
            latest_report = q_res["quality_report"]
            recorder.record(
                tool="check_quality",
                reason="Fresh quality measurement after image enhancement",
                result=_summarize_quality(latest_report),
                status="ok",
                duration_ms=q_dur,
            )
            quality_after = _extract_quality_metrics(latest_report)
            state.quality_after = quality_after
            state.trace = recorder.get_steps()

            if "excessive_reconstruction" in latest_report.get("needs_fix", []):
                verdict = "retake_needed"
                retake_message = "Too much of the leaf was hidden by glare. Please retake the photo from a different angle to avoid direct glare."
                break

            if "sharpen" in fixes_applied and "blur" in latest_report.get("needs_fix", []):
                verdict = "retake_needed"
                retake_message = "The photo is too blurry to measure. Hold the camera steady and tap to focus."
                break

            if "crop_to_leaf" in fixes_applied and "coverage" in latest_report.get("needs_fix", []):
                verdict = "retake_needed"
                retake_message = "The leaf is too small or far away in the frame. Please move closer and retake the photo."
                break

            if float(latest_report.get("coverage", {}).get("coverage", 0.0)) < 0.01:
                verdict = "retake_needed"
                retake_message = "Could not detect a leaf in the photo. Please frame a single leaf against a contrasting background."
                break

    discolor_metrics: DiscolorationMetrics
    if verdict != "retake_needed":
        t_seg = time.perf_counter()
        seg_res = TOOLS["segment_leaf"]["fn"](current_image_id)
        seg_dur = int((time.perf_counter() - t_seg) * 1000)
        mask_id = seg_res["mask_image_id"]
        mask_arr = get_image(mask_id)

        if np.count_nonzero(mask_arr) == 0:
            verdict = "retake_needed"
            retake_message = "The leaf could not be segmented. Please retake with better contrast against the background."
            recorder.record(
                tool="segment_leaf",
                reason="Segment leaf to extract boundaries for discoloration measurement",
                result="Segmentation produced empty leaf mask",
                status="error",
                duration_ms=seg_dur,
            )
            discolor_metrics = DiscolorationMetrics(
                healthy_pct=0.0,
                yellowing_pct=0.0,
                browning_pct=0.0,
                dark_spots_pct=0.0,
                holes=0,
                reconstructed_pct=0.0,
            )
        else:
            recorder.record(
                tool="segment_leaf",
                reason="Segment leaf to extract boundaries for discoloration measurement",
                result=f"Leaf segmented using {seg_res['info']['method']} covering {round(seg_res['info']['mask_area_ratio'] * 100.0, 1)}% of frame",
                status="ok",
                duration_ms=seg_dur,
            )

            t_disc = time.perf_counter()
            disc_res = TOOLS["measure_discoloration"]["fn"](
                current_image_id,
                mask_image_id=mask_id,
                reconstructed_mask_id=reconstructed_mask_id,
            )
            disc_dur = int((time.perf_counter() - t_disc) * 1000)
            stats = disc_res["stats"]
            overlay_id = disc_res["overlay_image_id"]
            state.overlay_img = get_image(overlay_id)

            discolor_metrics = DiscolorationMetrics(
                healthy_pct=float(stats.get("healthy_green_pct", 0.0)),
                yellowing_pct=float(stats.get("yellowing_pct", 0.0)),
                browning_pct=float(stats.get("browning_pct", 0.0)),
                dark_spots_pct=float(stats.get("dark_spots_pct", 0.0)),
                holes=int(stats.get("holes_count", 0)),
                reconstructed_pct=float(stats.get("reconstructed_pct", 0.0)),
            )

            recon_pct_val = round(discolor_metrics.reconstructed_pct)
            recorder.record(
                tool="measure_discoloration",
                reason="Measure discoloration components across valid leaf tissue",
                result=(
                    f"Discoloration measured (yellowing: {discolor_metrics.yellowing_pct}%, "
                    f"browning: {discolor_metrics.browning_pct}%, dark spots: {discolor_metrics.dark_spots_pct}%). "
                    f"{recon_pct_val}% of the leaf was reconstructed and excluded from measurements."
                ),
                status="ok",
                duration_ms=disc_dur,
            )

            t_patch = time.perf_counter()
            patch_res = TOOLS["find_worst_patch"]["fn"](
                current_image_id,
                mask_image_id=mask_id,
                reconstructed_mask_id=reconstructed_mask_id,
            )
            patch_dur = int((time.perf_counter() - t_patch) * 1000)
            worst_patch = patch_res.get("worst_patch", {})
            recorder.record(
                tool="find_worst_patch",
                reason="Locate area of highest discoloration density for detail inspection",
                result=f"Worst patch located at bbox {worst_patch.get('bbox', [])} with score {worst_patch.get('discoloration_score', 0.0)}",
                status="ok",
                duration_ms=patch_dur,
            )

            affected_area = (
                discolor_metrics.yellowing_pct
                + discolor_metrics.browning_pct
                + discolor_metrics.dark_spots_pct
            )

            if UNCERTAIN_BAND[0] <= affected_area <= UNCERTAIN_BAND[1]:
                bx, by, bw, bh = worst_patch.get("bbox", [0, 0, 0, 0])
                if bw > 0 and bh > 0:
                    t_zoom = time.perf_counter()
                    leaf_img = get_image(current_image_id)
                    patch_img = leaf_img[by : by + bh, bx : bx + bw]
                    p_img_id = register_image(patch_img)
                    patch_mask = mask_arr[by : by + bh, bx : bx + bw]
                    p_mask_id = register_image(patch_mask)
                    p_recon_id = None
                    if reconstructed_mask_id:
                        p_recon = get_image(reconstructed_mask_id)[by : by + bh, bx : bx + bw]
                        p_recon_id = register_image(p_recon)
                    zoom_disc = TOOLS["measure_discoloration"]["fn"](
                        p_img_id,
                        mask_image_id=p_mask_id,
                        reconstructed_mask_id=p_recon_id,
                    )
                    zoom_stats = zoom_disc["stats"]
                    zoom_dur = int((time.perf_counter() - t_zoom) * 1000)
                    recon_share = round(float(zoom_stats.get("reconstructed_pct", 0.0)))
                    recorder.record(
                        tool="measure_discoloration",
                        reason="Zoomed re-measure on worst patch due to borderline discoloration",
                        result=(
                            f"Zoomed patch: yellowing={zoom_stats.get('yellowing_pct', 0.0)}%, "
                            f"browning={zoom_stats.get('browning_pct', 0.0)}%, "
                            f"dark spots={zoom_stats.get('dark_spots_pct', 0.0)}%. "
                            f"{recon_share}% of the leaf was reconstructed and excluded from measurements."
                        ),
                        status="ok",
                        duration_ms=zoom_dur,
                    )
    else:
        discolor_metrics = DiscolorationMetrics(
            healthy_pct=0.0,
            yellowing_pct=0.0,
            browning_pct=0.0,
            dark_spots_pct=0.0,
            holes=0,
            reconstructed_pct=0.0,
        )

    state.discoloration = discolor_metrics
    state.trace = recorder.get_steps()

    final_fixed_img = get_image(current_image_id)
    final_overlay_img = get_image(overlay_id) if overlay_id else final_fixed_img

    if interpreter_override:
        final_verdict, interpretation, user_action = interpreter_override(
            final_fixed_img,
            discolor_metrics,
            latest_report,
            verdict,
            retake_message,
        )
    else:
        final_verdict, interpretation, user_action = interpret(
            final_fixed_img,
            discolor_metrics,
            latest_report,
            initial_verdict=verdict,
            retake_message=retake_message,
            api_key=api_key,
            model_name=model_name,
        )

    state.verdict = final_verdict
    state.user_action = user_action

    orig_b64 = encode_image_to_data_url(get_image(original_id))
    fixed_b64 = encode_image_to_data_url(final_fixed_img)
    overlay_b64 = encode_image_to_data_url(final_overlay_img)

    return AnalyzeResponse(
        verdict=final_verdict,
        summary=interpretation.summary,
        confidence_note=interpretation.confidence_note,
        possible_causes=interpretation.possible_causes,
        care_tips=interpretation.care_tips,
        images=Images(
            original=orig_b64,
            fixed=fixed_b64,
            overlay=overlay_b64,
        ),
        quality_before=quality_before,
        quality_after=quality_after,
        fixes_applied=fixes_applied,
        discoloration=discolor_metrics,
        trace=recorder.get_steps(),
        needs_user_action=user_action,
    )
