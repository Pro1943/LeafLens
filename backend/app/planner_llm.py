import concurrent.futures
import json
from typing import Any
from google import genai
from google.genai import types

FUNCTION_DECLARATIONS = [
    {
        "name": "fix_exposure",
        "description": "Adjust contrast and exposure using CLAHE and gamma correction.",
        "parameters": {
            "type": "object",
            "properties": {
                "clip_limit": {"type": "number", "description": "CLAHE clip limit"},
                "reason": {"type": "string", "description": "One sentence reason for calling this tool"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "sharpen",
        "description": "Apply unsharp mask filter to sharpen image details.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Sharpening intensity amount"},
                "reason": {"type": "string", "description": "One sentence reason for calling this tool"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "remove_glare",
        "description": "Detect and inpaint saturated glare reflections on the leaf.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One sentence reason for calling this tool"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "crop_to_leaf",
        "description": "Crop image to leaf bounding box with margin padding.",
        "parameters": {
            "type": "object",
            "properties": {
                "padding": {"type": "integer", "description": "Pixel padding around leaf"},
                "reason": {"type": "string", "description": "One sentence reason for calling this tool"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "finish",
        "description": "Finish planning fixes and proceed to leaf measurement.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One sentence reason for finishing"},
            },
            "required": ["reason"],
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You are a plant leaf image analysis planner. Your goal is to get the photo into an optimal measurable "
    "state, then call finish so the leaf can be segmented and measured. Allowed tools: fix_exposure, sharpen, "
    "remove_glare, crop_to_leaf, finish. You must provide a one-sentence reason with every call explaining "
    "your decision based on the quality metrics. If quality is acceptable or needed fixes are already applied, "
    "call finish."
)


def _call_gemini_planner(
    quality_report: dict[str, Any],
    fixes_applied: list[str],
    current_image_id: str,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))
    prompt = (
        f"Current Image ID: {current_image_id}\n"
        f"Quality Report: {json.dumps(quality_report)}\n"
        f"Fixes Already Applied: {json.dumps(fixes_applied)}\n"
        f"Select the next action."
    )
    tools = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    if response.function_calls:
        fn_call = response.function_calls[0]
        tool_name = fn_call.name
        args = dict(fn_call.args or {})
        reason = str(args.pop("reason", "Decided by LLM planner based on quality report."))
        return {"tool": tool_name, "args": args, "reason": reason}

    return {
        "tool": "finish",
        "args": {},
        "reason": "LLM planner concluded without requesting additional tools.",
    }


def plan_next_action(
    quality_report: dict[str, Any],
    fixes_applied: list[str],
    current_image_id: str,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _call_gemini_planner,
            quality_report,
            fixes_applied,
            current_image_id,
            api_key,
            model_name,
        )
        return future.result(timeout=10.0)
