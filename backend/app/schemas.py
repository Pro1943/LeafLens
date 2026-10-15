from typing import Literal, Optional
from pydantic import BaseModel

Verdict = Literal["healthy", "possible_issue", "uncertain", "retake_needed"]
Likelihood = Literal["low", "medium", "high"]
TraceStatus = Literal["ok", "warning", "error"]
UserActionType = Literal["retake", "confirm"]


class QualityMetrics(BaseModel):
    blur_score: float
    mean_brightness: float
    glare_ratio: float
    coverage: float
    needs_fix: list[str]


class TraceStep(BaseModel):
    step: int
    tool: str
    reason: str
    result: str
    status: TraceStatus
    duration_ms: int


class PossibleCause(BaseModel):
    name: str
    likelihood: Likelihood
    note: str


class Images(BaseModel):
    original: str
    fixed: str
    overlay: str


class DiscolorationMetrics(BaseModel):
    healthy_pct: float
    yellowing_pct: float
    browning_pct: float
    dark_spots_pct: float
    holes: int
    reconstructed_pct: float


class UserAction(BaseModel):
    type: UserActionType
    message: str


class AnalyzeResponse(BaseModel):
    verdict: Verdict
    summary: str
    confidence_note: str
    possible_causes: list[PossibleCause]
    care_tips: list[str]
    images: Images
    quality_before: QualityMetrics
    quality_after: QualityMetrics
    fixes_applied: list[str]
    discoloration: DiscolorationMetrics
    trace: list[TraceStep]
    needs_user_action: Optional[UserAction] = None
