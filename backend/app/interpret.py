import concurrent.futures
import json
from typing import Any, Optional
import cv2
import numpy as np
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.schemas import DiscolorationMetrics, PossibleCause, UserAction, Verdict
from leaflens.config import HEALTHY_THRESHOLD_PCT, ISSUE_THRESHOLD_PCT


class LLMInterpretation(BaseModel):
    summary: str
    confidence_note: str
    possible_causes: list[PossibleCause]
    care_tips: list[str]


SYSTEM_PROMPT = (
    "You are an expert botanical image analyst. Analyze the provided leaf image alongside measured "
    "discoloration statistics.\n"
    "Crucial guidelines:\n"
    "- These are possible causes from image analysis, not a medical or definitive agricultural diagnosis.\n"
    "- Never name a single certain disease; present potential conditions with appropriate likelihood.\n"
    "- Lower likelihoods when evidence is weak or localized.\n"
    "- The measured percentages exclude reconstructed pixels from specular glare repairs.\n"
    "- Provide between 2 and 4 possible causes with name, likelihood ('low', 'medium', or 'high'), and note.\n"
    "- Provide between 2 and 4 actionable care tips.\n"
    "- Output strict JSON matching the schema."
)


def _rule_based_interpretation(
    discoloration: DiscolorationMetrics,
    verdict: Verdict,
    retake_message: Optional[str] = None,
) -> LLMInterpretation:
    affected_area = round(
        discoloration.yellowing_pct + discoloration.browning_pct + discoloration.dark_spots_pct,
        2,
    )

    if verdict == "retake_needed":
        return LLMInterpretation(
            summary=retake_message or "Image quality prevents reliable leaf assessment.",
            confidence_note="Analysis stopped because quality thresholds were not satisfied.",
            possible_causes=[
                PossibleCause(
                    name="Image Quality Limitation",
                    likelihood="low",
                    note="Blur, glare, or framing prevented definitive colorimetric analysis.",
                ),
                PossibleCause(
                    name="Unconfirmed Foliar Health",
                    likelihood="low",
                    note="Retake the photograph under diffuse lighting to assess true leaf condition.",
                ),
            ],
            care_tips=[
                "Hold the camera steady and tap to focus directly on the leaf blade.",
                "Avoid harsh direct flash or strong specular light reflection.",
                "Ensure the leaf fills most of the frame against a contrasting background.",
            ],
        )

    if verdict == "healthy":
        return LLMInterpretation(
            summary=f"The leaf appears healthy with {discoloration.healthy_pct}% green tissue and only {affected_area}% discoloration.",
            confidence_note="Measurements reflect standard chlorophyll levels excluding reconstructed pixels.",
            possible_causes=[
                PossibleCause(
                    name="Normal Foliar Vitality",
                    likelihood="low",
                    note="Coloration and surface consistency fall within normal physiological parameters.",
                ),
                PossibleCause(
                    name="Minor Natural Aging",
                    likelihood="low",
                    note="Slight pigment fluctuation consistent with normal leaf maturity.",
                ),
            ],
            care_tips=[
                "Maintain consistent watering when topsoil feels dry to the touch.",
                "Ensure sufficient indirect sunlight appropriate for the plant species.",
                "Periodically check leaf undersides to monitor for emerging pests.",
            ],
        )

    if verdict == "possible_issue":
        causes: list[PossibleCause] = []
        if discoloration.yellowing_pct >= 8.0:
            causes.append(
                PossibleCause(
                    name="Chlorosis / Moisture or Nutrient Imbalance",
                    likelihood="high" if discoloration.yellowing_pct > 25.0 else "medium",
                    note=f"Yellowing covers {discoloration.yellowing_pct}% of the leaf surface.",
                )
            )
        if (discoloration.browning_pct + discoloration.dark_spots_pct) >= 5.0:
            causes.append(
                PossibleCause(
                    name="Necrotic Spotting / Fungal or Bacterial Wear",
                    likelihood="high" if (discoloration.browning_pct + discoloration.dark_spots_pct) > 15.0 else "medium",
                    note=f"Brown areas and dark spots account for {round(discoloration.browning_pct + discoloration.dark_spots_pct, 1)}% of surface.",
                )
            )
        if discoloration.holes > 0:
            causes.append(
                PossibleCause(
                    name="Foliar Perforation / Pest or Mechanical Injury",
                    likelihood="medium",
                    note=f"Detected {discoloration.holes} internal tissue void(s).",
                )
            )
        if len(causes) < 2:
            causes.append(
                PossibleCause(
                    name="Environmental Stress",
                    likelihood="medium",
                    note="Localized pigment anomalies indicate possible light, heat, or humidity stress.",
                )
            )
        return LLMInterpretation(
            summary=f"Elevated discoloration detected across {affected_area}% of the leaf surface.",
            confidence_note="Discoloration metrics exceed normal baseline thresholds.",
            possible_causes=causes[:4],
            care_tips=[
                "Check soil drainage and avoid overwatering or water accumulation.",
                "Isolate the plant if spotting spreads to adjacent foliage.",
                "Remove severely damaged leaves using sanitized pruning shears.",
                "Avoid overhead watering that keeps foliage wet for extended periods.",
            ],
        )

    return LLMInterpretation(
        summary=f"Borderline discoloration observed ({affected_area}% total affected area).",
        confidence_note="Measurements fall in the borderline range between healthy baseline and active stress.",
        possible_causes=[
            PossibleCause(
                name="Early Stage Physiological Stress",
                likelihood="low",
                note="Subtle discoloration that may indicate initial moisture or temperature fluctuations.",
            ),
            PossibleCause(
                name="Cultivar Variation or Lighting Artifact",
                likelihood="low",
                note="Mild color gradients that could stem from natural pigmentation or camera white balance.",
            ),
        ],
        care_tips=[
            "Observe leaf coloration over 3 to 5 days under consistent natural lighting.",
            "Verify soil moisture depth before modifying irrigation routines.",
            "Ensure stable ambient temperature away from heating or cooling vents.",
        ],
    )


def _call_gemini_interpreter(
    fixed_img: np.ndarray,
    discoloration: DiscolorationMetrics,
    quality_report: dict[str, Any],
    api_key: str,
    model_name: str,
) -> LLMInterpretation:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
    success, buffer = cv2.imencode(".jpg", fixed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("Failed to encode image for interpreter")

    image_part = types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
    user_prompt = (
        f"Quantitative Discoloration Data:\n"
        f"- Healthy tissue: {discoloration.healthy_pct}%\n"
        f"- Yellowing: {discoloration.yellowing_pct}%\n"
        f"- Browning: {discoloration.browning_pct}%\n"
        f"- Dark spots: {discoloration.dark_spots_pct}%\n"
        f"- Internal holes count: {discoloration.holes}\n"
        f"- Reconstructed glare area (excluded from above): {discoloration.reconstructed_pct}%\n\n"
        f"Quality Metrics: {json.dumps(quality_report)}\n\n"
        f"Provide the botanical interpretation as strict JSON."
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=LLMInterpretation,
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, user_prompt],
        config=config,
    )

    parsed = LLMInterpretation.model_validate_json(response.text)
    if not (2 <= len(parsed.possible_causes) <= 4):
        raise ValueError("Invalid number of possible causes returned")
    if not (2 <= len(parsed.care_tips) <= 4):
        raise ValueError("Invalid number of care tips returned")
    return parsed


def interpret(
    fixed_img: np.ndarray,
    discoloration: DiscolorationMetrics,
    quality_report: dict[str, Any],
    initial_verdict: Optional[Verdict] = None,
    retake_message: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> tuple[Verdict, LLMInterpretation, Optional[UserAction]]:
    if initial_verdict == "retake_needed":
        interp = _rule_based_interpretation(discoloration, "retake_needed", retake_message)
        action = UserAction(
            type="retake",
            message=retake_message or "A clearer photo is required to accurately assess leaf condition.",
        )
        return "retake_needed", interp, action

    affected_area = (
        discoloration.yellowing_pct
        + discoloration.browning_pct
        + discoloration.dark_spots_pct
    )

    if affected_area < HEALTHY_THRESHOLD_PCT:
        base_verdict: Verdict = "healthy"
    elif affected_area > ISSUE_THRESHOLD_PCT:
        base_verdict = "possible_issue"
    else:
        base_verdict = "uncertain"

    interpretation: Optional[LLMInterpretation] = None

    if api_key:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _call_gemini_interpreter,
                    fixed_img,
                    discoloration,
                    quality_report,
                    api_key,
                    model_name,
                )
                interpretation = future.result(timeout=15.0)
        except Exception:
            interpretation = None

    if interpretation is None:
        interpretation = _rule_based_interpretation(discoloration, base_verdict)

    final_verdict: Verdict = base_verdict
    if all(cause.likelihood == "low" for cause in interpretation.possible_causes):
        final_verdict = "uncertain"

    if final_verdict == "uncertain":
        action = UserAction(
            type="confirm",
            message="Discoloration is borderline or subtle. Please confirm whether symptoms are noticeable in person.",
        )
    elif final_verdict == "retake_needed":
        action = UserAction(
            type="retake",
            message=retake_message or "A new photo is required.",
        )
    else:
        action = None

    return final_verdict, interpretation, action
