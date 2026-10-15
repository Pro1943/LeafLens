import asyncio
import json
import logging
import os
import time
from typing import Optional
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from app.agent import AgentState, encode_image_to_data_url, run_agent
from app.schemas import (
    AnalyzeResponse,
    DiscolorationMetrics,
    Images,
    PossibleCause,
    QualityMetrics,
    UserAction,
)
from leaflens.config import APP_NAME
from leaflens.tools import clear_store

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))

logger = logging.getLogger("leaflens")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def _build_timeout_partial_response(state: AgentState) -> AnalyzeResponse:
    orig_b64 = encode_image_to_data_url(state.original_img)
    fixed_b64 = encode_image_to_data_url(state.current_img)
    overlay_b64 = encode_image_to_data_url(state.overlay_img)
    q_before = state.quality_before or QualityMetrics(
        blur_score=0.0,
        mean_brightness=0.0,
        glare_ratio=0.0,
        coverage=0.0,
        needs_fix=[],
    )
    q_after = state.quality_after or q_before
    discolor = state.discoloration or DiscolorationMetrics(
        healthy_pct=0.0,
        yellowing_pct=0.0,
        browning_pct=0.0,
        dark_spots_pct=0.0,
        holes=0,
        reconstructed_pct=0.0,
    )
    return AnalyzeResponse(
        verdict="uncertain",
        summary="Analysis timed out before completing full pipeline execution. Partial results returned.",
        confidence_note="Request timeout reached. Partial results should be visually verified.",
        possible_causes=[
            PossibleCause(
                name="Incomplete Analysis Due to Timeout",
                likelihood="low",
                note="Agent was unable to finish all evaluation steps within configured timeout.",
            ),
            PossibleCause(
                name="Borderline Image Conditions",
                likelihood="low",
                note="Iterative enhancements required extended processing time.",
            ),
        ],
        care_tips=[
            "Retake photo in clear, bright natural lighting to speed up processing.",
            "Verify leaf appearance manually.",
        ],
        images=Images(
            original=orig_b64,
            fixed=fixed_b64,
            overlay=overlay_b64,
        ),
        quality_before=q_before,
        quality_after=q_after,
        fixes_applied=list(state.fixes_applied),
        discoloration=discolor,
        trace=list(state.trace),
        needs_user_action=UserAction(
            type="confirm",
            message="Analysis exceeded the time limit. Please confirm if leaf shows noticeable symptoms in person.",
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": APP_NAME}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
    start_time = time.perf_counter()
    fallback_used = False
    verdict = "uncertain"
    steps_count = 0
    state: Optional[AgentState] = None

    try:
        content_type = (image.content_type or "").lower().split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported image format. Allowed formats: jpeg, png, webp.",
            )

        contents = await image.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Image size exceeds maximum limit of {MAX_UPLOAD_MB}MB.",
            )

        img_arr = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_arr is None or img_arr.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image data could not be decoded. Ensure the file is a valid image.",
            )

        state = AgentState(img_arr)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    run_agent,
                    img_arr,
                    api_key=GEMINI_API_KEY if GEMINI_API_KEY else None,
                    model_name=GEMINI_MODEL,
                    max_steps=MAX_AGENT_STEPS,
                    state=state,
                ),
                timeout=REQUEST_TIMEOUT_S,
            )
            verdict = response.verdict
            steps_count = len(response.trace)
            fallback_used = state.fallback_planner_used
            return response
        except asyncio.TimeoutError:
            verdict = "uncertain"
            steps_count = len(state.trace)
            fallback_used = state.fallback_planner_used
            return _build_timeout_partial_response(state)
    finally:
        clear_store()
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        log_entry = {
            "duration_ms": duration_ms,
            "verdict": verdict,
            "steps_taken": steps_count,
            "fallback_planner_used": fallback_used,
        }
        logger.info(json.dumps(log_entry))
