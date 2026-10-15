import { API_URL, USE_MOCK, mockResponse, traceSteps } from "./config"
import type { AnalyzeResponse, TraceStep } from "./types"

const TIMEOUT_MS = 90_000

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function isAnalyzeResponse(v: unknown): v is AnalyzeResponse {
  if (!v || typeof v !== "object") return false
  const r = v as Record<string, unknown>
  return (
    typeof r.verdict === "string" &&
    typeof r.summary === "string" &&
    typeof r.confidence_note === "string" &&
    Array.isArray(r.possible_causes) &&
    Array.isArray(r.care_tips) &&
    r.images !== null && typeof r.images === "object" &&
    r.quality_before !== null && typeof r.quality_before === "object" &&
    r.quality_after !== null && typeof r.quality_after === "object" &&
    Array.isArray(r.fixes_applied) &&
    r.discoloration !== null && typeof r.discoloration === "object" &&
    Array.isArray(r.trace)
  )
}

function friendlyError(err: unknown, status?: number): string {
  if (err instanceof DOMException && err.name === "AbortError") {
    return "The analysis took too long (90 s). Please try again with a clearer photo."
  }
  if (status !== undefined) {
    if (status >= 400 && status < 500) {
      return `The server rejected the request (${status}). Check that your photo is a JPG, PNG, or WEBP under 10 MB.`
    }
    if (status >= 500) {
      return `The analysis service encountered an error (${status}). Please try again in a moment.`
    }
  }
  if (err instanceof TypeError) {
    return "Could not reach the analysis service. Check your internet connection and try again."
  }
  return err instanceof Error ? err.message : "We could not analyze that photo."
}

export async function checkHealth(): Promise<boolean> {
  if (!API_URL) return false
  try {
    const res = await fetch(`${API_URL}/health`, { method: "GET" })
    return res.ok
  } catch {
    return false
  }
}

let activeController: AbortController | null = null

export function cancelAnalysis() {
  activeController?.abort()
  activeController = null
}

export async function analyzeImage(
  file: File,
  onStep?: (step: TraceStep) => void,
): Promise<AnalyzeResponse> {
  if (USE_MOCK) {
    for (const [index, item] of traceSteps.entries()) {
      await wait(600 + index * 70)
      onStep?.({
        step: index + 1,
        tool: item.tool,
        reason: item.reason,
        result: "Working through the image.",
        status: "ok",
        duration_ms: 600 + index * 70,
      })
    }
    const url = URL.createObjectURL(file)
    return { ...mockResponse, images: { original: url, fixed: url, overlay: url } }
  }

  if (!API_URL) throw new Error("The analysis service is not configured yet.")

  activeController = new AbortController()
  const { signal } = activeController
  const timer = setTimeout(() => activeController?.abort(), TIMEOUT_MS)

  try {
    const body = new FormData()
    body.append("image", file)

    const response = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      body,
      signal,
    })

    if (!response.ok) {
      let detail = ""
      try {
        const json = await response.json()
        if (json && typeof json.detail === "string") detail = json.detail
      } catch {}
      throw Object.assign(
        new Error(detail || friendlyError(null, response.status)),
        { status: response.status },
      )
    }

    const raw = await response.json()

    if (!isAnalyzeResponse(raw)) {
      throw new Error("The server returned an unexpected response shape. Please try again.")
    }

    if (onStep && Array.isArray(raw.trace)) {
      for (const step of raw.trace) {
        await wait(180)
        onStep(step)
      }
    }

    return raw
  } catch (err: unknown) {
    const status = (err as { status?: number }).status
    throw new Error(friendlyError(err, status))
  } finally {
    clearTimeout(timer)
    activeController = null
  }
}
