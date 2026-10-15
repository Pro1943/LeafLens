export const APP_NAME = "LeafLens"
export const APP_TAGLINE = "A clearer look at plant health."
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? ""
export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false"

export const navLinks = [
  { label: "How it works", href: "#how-it-works" },
  { label: "Features", href: "#features" },
  { label: "FAQ", href: "#faq" },
]

export const sampleImage = "https://images.unsplash.com/photo-1591958911259-bee2173bdccc?auto=format&fit=crop&w=1200&q=85"
export const sampleImageAlt = "Close-up of a green plant leaf"

export const motionEase = [0.22, 1, 0.36, 1] as const
export const motionDuration = 0.45

export const qualityLabels = [
  { key: "blur_score", label: "Blur" },
  { key: "mean_brightness", label: "Brightness" },
  { key: "glare_ratio", label: "Glare" },
  { key: "coverage", label: "Leaf coverage" },
] as const

export const discolorationLabels = [
  { key: "healthy_pct", label: "Healthy green", color: "bg-emerald-600" },
  { key: "yellowing_pct", label: "Yellowing", color: "bg-amber-400" },
  { key: "browning_pct", label: "Browning", color: "bg-orange-700" },
  { key: "dark_spots_pct", label: "Dark spots", color: "bg-stone-800" },
] as const

export const traceSteps = [
  { tool: "OpenCV / quality check", reason: "Checking whether the photo is usable." },
  { tool: "OpenCV / image repair", reason: "Balancing light and reducing blur." },
  { tool: "Leaf segmentation", reason: "Separating the leaf from its background." },
  { tool: "Color analysis", reason: "Measuring visible discoloration patterns." },
  { tool: "Evidence summary", reason: "Turning measurements into a readable explanation." },
]

export const featureItems = [
  ["Quality-aware", "Rejects or repairs blurry, dark, and glared photos.", "Sparkles"],
  ["Transparent trace", "Every step and reason is shown, not hidden.", "ListChecks"],
  ["Honest uncertainty", "Asks for a retake instead of guessing.", "ShieldQuestion"],
  ["Before and after", "Compare the original and fixed photo side by side.", "Split"],
  ["Measured evidence", "Percentages for yellowing, browning, spots, and holes.", "ScanLine"],
  ["Works from your phone camera", "Capture a leaf wherever you are.", "Smartphone"],
] as const

export const faqItems = [
  ["What kinds of plants work?", "LeafLens works best with a single, visible leaf photographed in even light. It can inspect many common houseplants, garden plants, and nursery plants, but image analysis is not a substitute for species-specific expertise."],
  ["Is my photo stored?", "In mock mode, your image stays in your browser. A connected analysis service may process the image to return results; check that service's retention policy before uploading sensitive photos."],
  ["How accurate is it?", "LeafLens reports visual evidence and uncertainty rather than promising a diagnosis. Results depend on the plant, lighting, focus, and how much of the leaf is visible."],
  ["Why did it ask me to retake the photo?", "A blurry, dark, reflective, or mostly hidden leaf can make measurements unreliable. A retake is safer than presenting a confident guess."],
  ["Does it work on a phone?", "Yes. The analyzer accepts a camera capture on supported mobile browsers and is designed for one-handed use."],
  ["Is it free?", "This prototype includes a mock analysis mode. Any production service pricing would depend on the analysis provider."],
] as const

export const mockResponse = {
  verdict: "possible_issue" as const,
  summary: "The leaf shows mild yellowing with a few darker spots.",
  confidence_note: "The photo is clear enough for a useful visual review, but this is not a diagnosis.",
  possible_causes: [
    { name: "Watering stress", likelihood: "medium" as const, note: "Uneven yellowing can appear when watering is inconsistent." },
    { name: "Natural leaf aging", likelihood: "low" as const, note: "Older leaves can yellow before they are shed." },
    { name: "Nutrient imbalance", likelihood: "low" as const, note: "A broader pattern across the plant would make this more likely." },
  ],
  care_tips: ["Check the soil moisture before watering again.", "Inspect nearby leaves for the same pattern.", "Avoid treating until the pattern is confirmed by a plant expert."],
  images: { original: "", fixed: "", overlay: "" },
  quality_before: { blur_score: 61, mean_brightness: 48, glare_ratio: 14, coverage: 76, needs_fix: ["low brightness", "mild glare"] },
  quality_after: { blur_score: 78, mean_brightness: 68, glare_ratio: 7, coverage: 76, needs_fix: [] },
  fixes_applied: ["Lifted shadows", "Reduced glare", "Sharpened edges"],
  discoloration: { healthy_pct: 68, yellowing_pct: 19, browning_pct: 8, dark_spots_pct: 5, holes: 1, reconstructed_pct: 0 },
  trace: [
    { step: 1, tool: "OpenCV / quality check", reason: "The image needed a quick quality check before inspection.", result: "Usable with light correction.", status: "ok" as const, duration_ms: 482 },
    { step: 2, tool: "OpenCV / image repair", reason: "Brightness and glare could hide subtle color changes.", result: "Adjusted exposure and reduced glare.", status: "ok" as const, duration_ms: 735 },
    { step: 3, tool: "Leaf segmentation", reason: "The leaf needed to be isolated from the background.", result: "Leaf coverage measured at 76%.", status: "ok" as const, duration_ms: 614 },
    { step: 4, tool: "Color analysis", reason: "Color regions reveal visible stress patterns.", result: "Mild yellowing and dark spots detected.", status: "warning" as const, duration_ms: 918 },
    { step: 5, tool: "Evidence summary", reason: "Measurements need a clear, cautious explanation.", result: "Possible issue, not a diagnosis.", status: "ok" as const, duration_ms: 352 },
  ],
  needs_user_action: null,
}
