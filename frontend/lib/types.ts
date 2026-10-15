export type Verdict = "healthy" | "possible_issue" | "uncertain" | "retake_needed"

export type QualityMetrics = {
  blur_score: number
  mean_brightness: number
  glare_ratio: number
  coverage: number
  needs_fix: string[]
}

export type TraceStep = {
  step: number
  tool: string
  reason: string
  result: string
  status: "ok" | "warning" | "error"
  duration_ms: number
}

export type AnalyzeResponse = {
  verdict: Verdict
  summary: string
  confidence_note: string
  possible_causes: { name: string; likelihood: "low" | "medium" | "high"; note: string }[]
  care_tips: string[]
  images: { original: string; fixed: string; overlay: string }
  quality_before: QualityMetrics
  quality_after: QualityMetrics
  fixes_applied: string[]
  discoloration: {
    healthy_pct: number
    yellowing_pct: number
    browning_pct: number
    dark_spots_pct: number
    holes: number
    reconstructed_pct: number
  }
  trace: TraceStep[]
  needs_user_action: { type: "retake" | "confirm"; message: string } | null
}
