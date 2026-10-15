import cv2
import numpy as np
import pytest

from app.agent import run_agent
from app.schemas import AnalyzeResponse


def test_dark_image_leads_to_fix_exposure_and_better_quality():
    img = np.full((300, 300, 3), 10, dtype=np.uint8)
    cv2.circle(img, (150, 150), 90, (20, 35, 20), -1)

    res = run_agent(img)

    assert isinstance(res, AnalyzeResponse)
    assert "fix_exposure" in res.fixes_applied
    assert res.quality_after.mean_brightness > res.quality_before.mean_brightness


def test_unfixable_blur_leads_to_retake_needed():
    img = np.full((300, 300, 3), (35, 160, 45), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (20, 180, 20), -1)
    blurred = cv2.GaussianBlur(img, (51, 51), 25.0)

    res = run_agent(blurred)

    assert res.verdict == "retake_needed"
    assert res.needs_user_action is not None
    assert res.needs_user_action.type == "retake"
    assert "blurry" in res.needs_user_action.message.lower()


def test_unknown_planner_tool_is_rejected_and_logged():
    img = np.full((200, 200, 3), 20, dtype=np.uint8)
    cv2.circle(img, (100, 100), 70, (35, 180, 45), -1)

    calls = 0

    def fake_planner(quality_report, fixes_applied, current_image_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "tool": "invalid_magic_tool",
                "args": {},
                "reason": "Test unknown tool rejection",
            }
        return {"tool": "finish", "args": {}, "reason": "Complete"}

    res = run_agent(img, planner_override=fake_planner)

    error_steps = [s for s in res.trace if s.tool == "invalid_magic_tool"]
    assert len(error_steps) == 1
    assert error_steps[0].status == "error"
    assert "not in allowed tools" in error_steps[0].result


def test_excessive_reconstruction_leads_to_retake_needed():
    img = np.full((300, 300, 3), 10, dtype=np.uint8)
    cv2.circle(img, (150, 150), 90, (35, 180, 45), -1)
    cv2.circle(img, (150, 150), 45, (255, 255, 255), -1)

    res = run_agent(img)

    assert res.verdict == "retake_needed"
    assert res.needs_user_action is not None
    assert res.needs_user_action.type == "retake"
    assert "glare" in res.needs_user_action.message.lower()

    tool_names = [t.tool for t in res.trace]
    assert "segment_leaf" not in tool_names
    assert "measure_discoloration" not in tool_names
