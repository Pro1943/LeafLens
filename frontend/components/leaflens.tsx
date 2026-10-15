"use client"

import { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { AnimatePresence, motion, useScroll, useSpring } from "framer-motion"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { AlertCircle, ArrowRight, Check, ChevronRight, CircleHelp, Droplets, ImagePlus, Leaf, ListChecks, Menu, Moon, ScanSearch, ScanLine, ShieldQuestion, Smartphone, Sparkles, Split, Sun, Upload, Wand2, X } from "lucide-react"
import { APP_NAME, APP_TAGLINE, discolorationLabels, faqItems, featureItems, motionEase, navLinks, qualityLabels, sampleImage, sampleImageAlt, traceSteps } from "@/lib/config"
import { analyzeImage, cancelAnalysis, checkHealth } from "@/lib/api"
import type { AnalyzeResponse, TraceStep } from "@/lib/types"
import { fade, fadeUp, MotionSection, stagger } from "@/lib/motion"

function resolveImage(src: string, fallback: string): string {
  if (!src) return fallback
  if (src.startsWith("data:") || src.startsWith("http") || src.startsWith("/")) return src
  return `data:image/jpeg;base64,${src}`
}

function Brand() { return <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight"><span className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground"><Leaf /></span><span>{APP_NAME}</span></Link> }

function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.2 })
  return <motion.div aria-hidden="true" className="fixed inset-x-0 top-0 z-[60] h-0.5 origin-left bg-emerald-500" style={{ scaleX }} />
}

export function Navbar() {
  const pathname = usePathname()
  const [dark, setDark] = useState(false); const [open, setOpen] = useState(false)
  const toggle = () => { document.documentElement.classList.toggle("dark"); setDark(!dark) }
  const sectionHref = (href: string) => pathname === "/" ? href : `/${href}`
  return <><ScrollProgress /><header className="fixed inset-x-0 top-0 z-50 px-4 pt-4"><div className="mx-auto flex max-w-6xl items-center justify-between rounded-2xl border border-border/70 bg-background/80 px-4 py-3 shadow-sm backdrop-blur-xl"><Brand /><nav className="hidden items-center gap-7 text-sm text-muted-foreground md:flex">{navLinks.map((link) => <a key={link.href} href={sectionHref(link.href)} className="transition-colors hover:text-foreground">{link.label}</a>)}</nav><div className="flex items-center gap-2"><Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">{dark ? <Sun /> : <Moon />}</Button><Button asChild className="hidden min-w-32 gap-2 rounded-xl whitespace-nowrap md:inline-flex"><Link href="/analyze">Try {APP_NAME}<ArrowRight data-icon="inline-end" /></Link></Button><Button variant="ghost" size="icon" className="md:hidden" onClick={() => setOpen(!open)} aria-label="Open menu">{open ? <X /> : <Menu />}</Button></div></div>{open && <nav className="mx-auto mt-2 flex max-w-6xl flex-col gap-1 rounded-2xl border bg-background p-3 shadow-lg md:hidden">{navLinks.map((link) => <a key={link.href} href={sectionHref(link.href)} onClick={() => setOpen(false)} className="rounded-xl px-3 py-3 text-sm hover:bg-muted">{link.label}</a>)}<Button asChild className="mt-1 rounded-xl"><Link href="/analyze">Try {APP_NAME}</Link></Button></nav>}</header></>
}

export function HeroPreview() { const [position, setPosition] = useState(52); return <motion.div variants={fadeUp} className="relative mx-auto max-w-md"><div className="absolute -inset-8 rounded-full bg-emerald-500/10 blur-3xl" /><Card className="relative overflow-hidden rounded-3xl border-border/70 bg-card/90 shadow-2xl shadow-emerald-950/10"><div className="relative aspect-[4/3] overflow-hidden"><img src={sampleImage} alt={sampleImageAlt} className="size-full object-cover" /><div className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-white/80" style={{ width: `${position}%` }}><img src={sampleImage} alt="Sharpened leaf preview" className="size-full max-w-none object-cover" style={{ width: `${100 / (position / 100)}%` }} /></div><input aria-label="Preview before and after" type="range" min="15" max="85" value={position} onChange={(e) => setPosition(Number(e.target.value))} className="absolute inset-0 size-full cursor-ew-resize opacity-0" /><div className="absolute bottom-3 left-3 rounded-full bg-black/55 px-3 py-1 text-xs text-white">fixed photo</div><div className="absolute bottom-3 right-3 rounded-full bg-white/80 px-3 py-1 text-xs text-stone-900">original</div></div><CardContent className="p-5"><div className="mb-4 flex items-center justify-between"><span className="text-sm font-medium">Live analysis trace</span><Badge variant="secondary" className="gap-1"><span className="size-1.5 rounded-full bg-emerald-500" />running</Badge></div><div className="flex flex-col gap-3">{traceSteps.slice(0, 3).map((step, i) => <motion.div key={step.tool} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 + i * 0.45, duration: 0.4, ease: motionEase }} className="flex items-center gap-3 text-xs text-muted-foreground"><span className="flex size-6 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-700"><Check /></span>{step.tool}</motion.div>)}</div><div className="mt-5 flex items-center justify-between rounded-xl bg-emerald-500/10 px-3 py-2.5 text-sm"><span>Possible issue</span><span className="font-medium text-emerald-700">mild yellowing</span></div></CardContent></Card></motion.div> }

export function Hero() { return <section className="relative overflow-hidden px-4 pb-20 pt-36 sm:pb-28 sm:pt-48"><div className="absolute left-1/2 top-24 size-[34rem] -translate-x-1/2 rounded-full bg-emerald-500/10 blur-3xl" /><div className="relative mx-auto grid max-w-6xl items-center gap-14 lg:grid-cols-[1fr_0.85fr]"><motion.div initial="hidden" animate="visible" variants={stagger}><motion.div variants={fadeUp} className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-600/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-800 dark:text-emerald-300"><Sparkles /> Visual plant care, explained</motion.div><motion.h1 variants={fadeUp} className="max-w-3xl text-5xl font-semibold leading-[1.02] tracking-[-0.06em] sm:text-7xl">See what&apos;s wrong with your plant, <span className="text-muted-foreground">and how we found out.</span></motion.h1><motion.p variants={fadeUp} className="mt-6 max-w-xl text-lg leading-8 text-muted-foreground">Upload a leaf photo. {APP_NAME} fixes the picture, measures the leaf, and shows every step it took.</motion.p><motion.div variants={fadeUp} className="mt-8 flex flex-col gap-3 sm:flex-row"><Button asChild size="lg" className="gap-2 rounded-xl whitespace-nowrap"><Link href="/analyze">Analyze a leaf<ArrowRight data-icon="inline-end" /></Link></Button><Button asChild size="lg" variant="outline" className="rounded-xl"><a href="#how-it-works">See how it works<ChevronRight data-icon="inline-end" /></a></Button></motion.div></motion.div><HeroPreview /></div></section> }

export function HowItWorks() { const steps = [[Upload, "Upload", "Choose a clear photo from your camera roll."], [Wand2, "Fix", "The agent improves light and sharpness."], [ScanSearch, "Inspect", "It measures visible color and shape changes."], [ListChecks, "Explain", "You get the evidence, not just a label."]]; return <MotionSection id="how-it-works" className="mx-auto max-w-6xl px-4 py-20 sm:py-28"><div className="max-w-xl"><p className="eyebrow">How it works</p><h2 className="section-title">A small trace for a better answer.</h2></div><div className="relative mt-12 grid gap-8 md:grid-cols-4">{steps.map(([Icon, title, text], i) => <motion.div variants={fadeUp} key={String(title)} className="relative"><div className="mb-5 flex size-12 items-center justify-center rounded-2xl bg-emerald-500/10 text-emerald-700"><Icon /></div><p className="mb-2 text-sm font-semibold">0{i + 1} / {title}</p><p className="text-sm leading-6 text-muted-foreground">{text}</p>{i < 3 && <div className="absolute left-12 top-6 hidden h-px w-[calc(100%-2rem)] bg-gradient-to-r from-emerald-500/50 to-border md:block" />}</motion.div>)}</div></MotionSection> }

export function FeatureGrid() { return <MotionSection id="features" className="border-y bg-muted/30 px-4 py-20 sm:py-28"><div className="mx-auto max-w-6xl"><div className="max-w-xl"><p className="eyebrow">Features</p><h2 className="section-title">Careful by design.</h2></div><div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{featureItems.map(([title, text, iconName]) => { const Icon = ({ Sparkles, ListChecks, ShieldQuestion, Split, ScanLine, Smartphone } as Record<string, typeof Sparkles>)[iconName]; return <motion.div variants={fadeUp} key={title} className="rounded-2xl border bg-background p-6 transition-shadow hover:-translate-y-0.5 hover:shadow-lg"><Icon className="mb-8 text-emerald-700" /><h3 className="font-medium">{title}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{text}</p></motion.div> })}</div></div></MotionSection> }

export function DemoSlider() { const [pos, setPos] = useState(50); return <MotionSection className="mx-auto max-w-6xl px-4 py-20 sm:py-28"><div className="grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]"><div><p className="eyebrow">See the difference</p><h2 className="section-title">The fix is part of the evidence.</h2><p className="mt-5 max-w-lg leading-7 text-muted-foreground">Compare the source image with the corrected version before you trust what the agent measured.</p></div><div><div className="relative aspect-[4/3] overflow-hidden rounded-3xl border shadow-lg"><img src={sampleImage} alt="Original leaf photo" className="size-full object-cover brightness-75 blur-[1px]" /><div className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-white" style={{ width: `${pos}%` }}><img src={sampleImage} alt="Fixed leaf photo" className="size-full max-w-none object-cover saturate-125 brightness-110" style={{ width: `${100 / (pos / 100)}%` }} /></div><input aria-label="Compare original and fixed image" type="range" min="10" max="90" value={pos} onChange={(e) => setPos(Number(e.target.value))} className="absolute inset-0 size-full cursor-ew-resize opacity-0" /><div className="absolute bottom-3 left-3 rounded-full bg-black/55 px-3 py-1 text-xs text-white">fixed</div><div className="absolute bottom-3 right-3 rounded-full bg-white/85 px-3 py-1 text-xs text-stone-900">original</div></div><div className="mt-4 grid grid-cols-4 gap-2">{[["Blur", "61 \u2192 78"], ["Brightness", "48 \u2192 68"], ["Glare", "14 \u2192 7"], ["Coverage", "76%"]].map(([a, b]) => <div key={a} className="rounded-xl border bg-card p-3"><p className="text-[11px] text-muted-foreground">{a}</p><p className="mt-1 text-sm font-medium">{b}</p></div>)}</div></div></div></MotionSection> }

export function MeasuresSection() { return <MotionSection className="border-y bg-muted/30 px-4 py-20 sm:py-28"><div className="mx-auto grid max-w-6xl gap-12 lg:grid-cols-2"><div><p className="eyebrow">What it measures</p><h2 className="section-title">Numbers with context.</h2><div className="mt-8 flex flex-col gap-4">{["Blur", "Exposure", "Glare", "Leaf coverage"].map((x, i) => <div key={x} className="flex items-center justify-between border-b pb-4 text-sm"><span>{x}</span><span className="text-muted-foreground">{["61 / 100", "48 / 100", "14%", "76%"][i]}</span></div>)}</div></div><div className="rounded-3xl border bg-background p-6 sm:p-8"><div className="flex items-center gap-2"><ScanLine className="text-emerald-700" /><h3 className="font-medium">Leaf condition</h3></div><div className="mt-7 flex flex-col gap-5">{[["Healthy green", 68, "bg-emerald-600"], ["Yellowing", 19, "bg-amber-400"], ["Browning", 8, "bg-orange-700"], ["Dark spots", 5, "bg-stone-800"]].map(([label, value, color]) => <div key={String(label)}><div className="mb-2 flex justify-between text-sm"><span>{label}</span><span className="text-muted-foreground">{value}%</span></div><div className="h-2 overflow-hidden rounded-full bg-muted"><motion.div initial={{ width: 0 }} whileInView={{ width: `${value}%` }} viewport={{ once: true }} transition={{ duration: 0.8, ease: motionEase }} className={`h-full rounded-full ${color}`} /></div></div>)}</div></div></div></MotionSection> }

export function AudienceCards() { return <MotionSection className="mx-auto max-w-6xl px-4 py-20 sm:py-28"><p className="eyebrow">Made for curious caretakers</p><div className="mt-8 grid gap-4 md:grid-cols-3">{[[Droplets, "Home gardeners", "Get a second set of eyes before you change a care routine."], [Leaf, "Plant nurseries", "Give customers a transparent visual starting point."], [CircleHelp, "Students and hobbyists", "Turn a leaf photo into a small, explainable investigation."]].map(([Icon, title, text]) => <motion.div variants={fadeUp} key={String(title)} className="rounded-2xl border p-6"><Icon className="mb-10 text-emerald-700" /><h3 className="font-medium">{String(title)}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{String(text)}</p></motion.div>)}</div></MotionSection> }

export function LimitsCard() { return <MotionSection className="mx-auto max-w-6xl px-4 pb-20"><div className="rounded-3xl border border-emerald-700/20 bg-emerald-500/5 p-7 sm:p-10"><ShieldQuestion className="text-emerald-700" /><h2 className="mt-5 text-2xl font-semibold tracking-tight">What {APP_NAME} is not.</h2><p className="mt-4 max-w-3xl leading-7 text-muted-foreground">It provides possible causes from image analysis, not a diagnosis. Results depend on photo quality, plant context, and what is visible in the frame. Check with a local nursery or plant expert before treating a plant.</p></div></MotionSection> }

export function Faq() { return <MotionSection id="faq" className="mx-auto max-w-3xl px-4 py-20 sm:py-28"><div className="text-center"><p className="eyebrow">FAQ</p><h2 className="section-title">Good questions are part of good care.</h2></div><Accordion type="single" className="mt-10">{faqItems.map(([q, a], i) => <AccordionItem key={q} value={`item-${i}`}><AccordionTrigger className="text-left">{q}</AccordionTrigger><AccordionContent className="leading-7 text-muted-foreground">{a}</AccordionContent></AccordionItem>)}</Accordion></MotionSection> }

export function CtaBand() { return <section className="px-4 pb-20"><div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 rounded-3xl bg-primary p-8 text-primary-foreground sm:flex-row sm:items-center sm:p-12"><div><h2 className="text-3xl font-semibold tracking-tight">Check your first leaf in under a minute.</h2><p className="mt-2 text-primary-foreground/70">Start with one clear photo.</p></div><Button asChild variant="secondary" size="lg" className="gap-2 rounded-xl whitespace-nowrap"><Link href="/analyze">Analyze a leaf<ArrowRight data-icon="inline-end" /></Link></Button></div></section> }

export function Footer() { return <footer className="border-t px-4 py-10"><div className="mx-auto flex max-w-6xl flex-col gap-8 sm:flex-row sm:items-end sm:justify-between"><div><Brand /><p className="mt-3 text-sm text-muted-foreground">{APP_TAGLINE}</p></div><div className="flex flex-wrap gap-5 text-sm text-muted-foreground"><a href="https://github.com/Pro1943/LeafLens" target="_blank" rel="noreferrer">GitHub</a><a href="/#how-it-works">How it works</a><a href="/#faq">FAQ</a></div></div><p className="mx-auto mt-8 max-w-6xl text-xs text-muted-foreground">Visual analysis can be wrong. Consult a plant expert before treating.</p></footer> }

function StepList({ steps, active }: { steps: TraceStep[]; active: number }) { return <div className="flex flex-col gap-3" aria-live="polite">{steps.map((step, i) => <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} key={step.step} className="flex gap-3 rounded-2xl border bg-card p-4"><span className={`mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full ${i < active ? "bg-emerald-500/15 text-emerald-700" : "bg-muted text-muted-foreground"}`}>{i < active ? <Check /> : step.step}</span><div><p className="text-sm font-medium">{step.tool}</p><p className="mt-1 text-xs leading-5 text-muted-foreground">{step.reason}</p></div></motion.div>)}</div> }

function BeforeAfter({ images }: { images: AnalyzeResponse["images"] }) {
  const [pos, setPos] = useState(50)
  const [overlay, setOverlay] = useState(false)
  const original = resolveImage(images.original, sampleImage)
  const fixed = resolveImage(images.fixed, sampleImage)
  const overlayImg = resolveImage(images.overlay, "")
  return <Card className="overflow-hidden rounded-3xl"><CardHeader className="flex-row items-center justify-between"><CardTitle>Before and after</CardTitle><Button variant="outline" size="sm" onClick={() => setOverlay(!overlay)}>{overlay ? "Hide overlay" : "Show overlay"}</Button></CardHeader><CardContent><div className="relative aspect-[4/3] overflow-hidden rounded-2xl"><img src={original} alt="Original uploaded leaf" className="size-full object-cover" /><div className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-white" style={{ width: `${pos}%` }}><img src={fixed} alt="Fixed leaf" className="size-full max-w-none object-cover" style={{ width: `${100 / (pos / 100)}%` }} />{overlay && overlayImg && <img src={overlayImg} alt="Discoloration overlay" className="absolute inset-0 size-full object-cover opacity-60 mix-blend-multiply" />}</div><input type="range" aria-label="Compare original and fixed images" min="10" max="90" value={pos} onChange={(e) => setPos(Number(e.target.value))} className="absolute inset-0 size-full cursor-ew-resize opacity-0" /></div></CardContent></Card>
}

function QualityPanel({ data }: { data: AnalyzeResponse }) { return <Card className="rounded-3xl"><CardHeader><CardTitle>Photo quality</CardTitle></CardHeader><CardContent className="flex flex-col gap-5">{qualityLabels.map(({ key, label }) => { const before = data.quality_before[key]; const after = data.quality_after[key]; return <div key={key}><div className="mb-2 flex justify-between text-sm"><span>{label}</span><span className="text-muted-foreground">{before} &rarr; {after}</span></div><div className="flex gap-2"><Progress value={before} className="h-2" /><Progress value={after} className="h-2 [&>div]:bg-emerald-600" /></div></div> })}<div className="flex flex-wrap gap-2 pt-1">{data.fixes_applied.map((fix) => <Badge key={fix} variant="secondary">{fix}</Badge>)}</div></CardContent></Card> }

function DiscolorationBar({ data }: { data: AnalyzeResponse["discoloration"] }) {
  const reconstructed = typeof data.reconstructed_pct === "number" ? data.reconstructed_pct : 0
  return (
    <Card className="rounded-3xl">
      <CardHeader><CardTitle>Leaf condition</CardTitle></CardHeader>
      <CardContent>
        <div className="flex h-4 overflow-hidden rounded-full">
          {discolorationLabels.map(({ key, color }) => (
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${data[key]}%` }}
              transition={{ duration: 0.8, ease: motionEase }}
              key={key}
              className={color}
            />
          ))}
        </div>
        <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
          {discolorationLabels.map(({ key, label, color }) => (
            <div key={key} className="flex items-center gap-2">
              <span className={`size-2 rounded-full ${color}`} />
              <span className="text-muted-foreground">{label}</span>
              <strong className="ml-auto">{data[key]}%</strong>
            </div>
          ))}
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-stone-400" />
            <span className="text-muted-foreground">Holes</span>
            <strong className="ml-auto">{data.holes}</strong>
          </div>
          {reconstructed > 0 && (
            <div className="flex items-center gap-2">
              <span className="size-2 rounded-full bg-yellow-400" />
              <span className="text-muted-foreground">Reconstructed</span>
              <strong className="ml-auto">{reconstructed}%</strong>
            </div>
          )}
        </div>
        {reconstructed > 0 && (
          <p className="mt-4 text-xs leading-5 text-muted-foreground">
            {reconstructed}% of the leaf was reconstructed and excluded from these measurements.
            Reconstructed areas are shown in gold on the overlay.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function TraceTimeline({ trace }: { trace: TraceStep[] }) { return <Card className="rounded-3xl"><CardHeader><CardTitle>Agent trace</CardTitle></CardHeader><CardContent className="flex flex-col gap-5">{trace.map((step) => <details key={step.step} className="group border-l-2 border-emerald-500/30 pl-4"><summary className="flex cursor-pointer list-none items-start gap-3"><span className={`mt-1 flex size-6 shrink-0 items-center justify-center rounded-full ${step.status === "warning" ? "bg-amber-500/15 text-amber-700" : step.status === "error" ? "bg-destructive/15 text-destructive" : "bg-emerald-500/15 text-emerald-700"}`}>{step.status === "warning" || step.status === "error" ? <AlertCircle /> : <Check />}</span><span className="flex-1"><strong className="block text-sm">{step.tool}</strong><span className="mt-1 block text-xs text-muted-foreground">{step.result}</span></span><span className="text-xs text-muted-foreground">{step.duration_ms}ms</span></summary><p className="mt-3 pl-9 text-xs leading-5 text-muted-foreground">Why: {step.reason}</p></details>)}</CardContent></Card> }

function CausesList({ data }: { data: AnalyzeResponse }) { return <Card className="rounded-3xl"><CardHeader><CardTitle>What this could mean</CardTitle><p className="text-sm leading-6 text-muted-foreground">These are possible causes based on image analysis, not a diagnosis. Check with a local nursery or plant expert before treating.</p></CardHeader><CardContent><div className="flex flex-col gap-5">{data.possible_causes.map((cause) => <div key={cause.name} className="border-b pb-4 last:border-0"><div className="flex items-center justify-between gap-3"><strong className="text-sm">{cause.name}</strong><Badge variant={cause.likelihood === "high" ? "destructive" : "secondary"}>{cause.likelihood}</Badge></div><p className="mt-2 text-sm leading-6 text-muted-foreground">{cause.note}</p></div>)}</div><h3 className="mt-4 text-sm font-semibold">Care tips</h3><ul className="mt-3 flex flex-col gap-2 text-sm text-muted-foreground">{data.care_tips.map((tip) => <li key={tip} className="flex gap-2"><Check className="mt-0.5 text-emerald-700" />{tip}</li>)}</ul></CardContent></Card> }

function VerdictCard({ data, onRetake }: { data: AnalyzeResponse; onRetake: () => void }) {
  const labels = { healthy: "Looks healthy", possible_issue: "Possible issue", uncertain: "Uncertain", retake_needed: "Retake needed" }
  const isRetake = data.verdict === "retake_needed" || data.needs_user_action !== null
  if (isRetake) {
    const message = data.needs_user_action?.message ?? "This photo could not be analyzed reliably. Please retake with better lighting or focus."
    return (
      <Card className="rounded-3xl border-amber-500/30 bg-amber-500/10">
        <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-start">
          <span className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-amber-500/20 text-amber-700">
            <AlertCircle />
          </span>
          <div className="flex-1">
            <p className="text-lg font-semibold">{labels[data.verdict]}</p>
            <p className="mt-1 leading-6 text-muted-foreground">{message}</p>
            {data.confidence_note && <p className="mt-3 text-xs text-muted-foreground">{data.confidence_note}</p>}
          </div>
          <Button onClick={onRetake} variant="outline" className="shrink-0 rounded-xl">
            Retake photo
          </Button>
        </CardContent>
      </Card>
    )
  }
  return (
    <Card className="rounded-3xl border-emerald-700/20 bg-emerald-500/5">
      <CardContent className="flex gap-4 p-6">
        <span className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-700">
          {data.verdict === "healthy" ? <Check /> : <ScanSearch />}
        </span>
        <div>
          <p className="text-lg font-semibold">{labels[data.verdict]}</p>
          <p className="mt-1 leading-6 text-muted-foreground">{data.summary}</p>
          <p className="mt-3 text-xs text-muted-foreground">{data.confidence_note}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState("")
  const [state, setState] = useState<"upload" | "analyzing" | "result" | "error">("upload")
  const [steps, setSteps] = useState<TraceStep[]>([])
  const [data, setData] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState("")
  const [wakingUp, setWakingUp] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview) }
  }, [preview])

  const choose = (next: File | undefined) => {
    if (!next) return
    if (!next.type.match(/image\/(jpeg|png|webp)/) || next.size > 10 * 1024 * 1024) {
      setError("Choose a JPG, PNG, or WEBP image under 10 MB.")
      return
    }
    setError("")
    setFile(next)
    setPreview(URL.createObjectURL(next))
  }

  const run = async () => {
    if (!file) return
    setState("analyzing")
    setSteps([])
    setWakingUp(false)

    const healthTimer = setTimeout(async () => {
      const alive = await checkHealth()
      if (!alive) setWakingUp(true)
    }, 3000)

    try {
      const result = await analyzeImage(file, (step) => {
        setWakingUp(false)
        setSteps((current) => [...current, step])
      })
      setData(result)
      setState("result")
    } catch (e) {
      setError(e instanceof Error ? e.message : "We could not analyze that photo.")
      setState("error")
    } finally {
      clearTimeout(healthTimer)
      setWakingUp(false)
    }
  }

  const reset = () => {
    cancelAnalysis()
    setState("upload")
    setData(null)
    setFile(null)
    setPreview("")
    setSteps([])
    setError("")
    setWakingUp(false)
  }

  const isRetake = data !== null && (data.verdict === "retake_needed" || data.needs_user_action !== null)

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="mx-auto max-w-6xl px-4 pb-20 pt-32 sm:pt-40">
        <div className="mb-10 flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Leaf analysis</p>
            <h1 className="text-4xl font-semibold tracking-[-0.05em] sm:text-6xl">A closer look at your leaf.</h1>
            <p className="mt-4 max-w-xl leading-7 text-muted-foreground">Upload one clear leaf photo. We&apos;ll improve the image, inspect what is visible, and show the work.</p>
          </div>
          <Button asChild variant="ghost" className="hidden sm:inline-flex"><Link href="/">Back home</Link></Button>
        </div>
        <AnimatePresence mode="wait">
          {state === "upload" && (
            <motion.div key="upload" initial="hidden" animate="visible" variants={fade}>
              <Card className="rounded-3xl border-dashed p-5 sm:p-10">
                <div onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files[0]) }} className="flex min-h-80 flex-col items-center justify-center rounded-2xl border-2 border-dashed border-emerald-700/30 bg-emerald-500/5 p-8 text-center">
                  <div className="mb-5 flex size-14 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-700"><ImagePlus /></div>
                  <h2 className="text-xl font-medium">Drop a leaf photo here</h2>
                  <p className="mt-2 text-sm text-muted-foreground">JPG, PNG, or WEBP up to 10 MB</p>
                  <div className="mt-6 flex flex-col gap-2 sm:flex-row">
                    <Button onClick={() => inputRef.current?.click()} className="rounded-xl"><Upload data-icon="inline-start" />Choose photo</Button>
                    <label className="sm:hidden">
                      <Button asChild variant="outline" className="w-full rounded-xl"><span>Take a photo</span></Button>
                      <input type="file" accept="image/jpeg,image/png,image/webp" capture="environment" onChange={(e) => choose(e.target.files?.[0])} className="sr-only" />
                    </label>
                    <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => choose(e.target.files?.[0])} className="hidden" />
                  </div>
                  {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
                </div>
                {preview && (
                  <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center">
                    <img src={preview} alt="Selected leaf preview" className="h-28 w-40 rounded-2xl object-cover" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">Ready to inspect</p>
                      <p className="mt-1 text-sm text-muted-foreground">{file?.name}</p>
                    </div>
                    <Button onClick={run} className="rounded-xl">Analyze photo<ArrowRight data-icon="inline-end" /></Button>
                  </div>
                )}
              </Card>
            </motion.div>
          )}
          {state === "analyzing" && (
            <motion.div key="analyzing" initial="hidden" animate="visible" variants={fade} className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
              <Card className="overflow-hidden rounded-3xl">
                <img src={preview} alt="Uploaded leaf being analyzed" className="aspect-[4/3] w-full object-cover" />
                <CardContent className="p-6">
                  <Badge variant="secondary">Analyzing photo</Badge>
                  {wakingUp && <p className="mt-3 text-sm text-muted-foreground">Waking up the server, this may take a moment...</p>}
                  <p className="mt-4 leading-7 text-muted-foreground">Each step is visible so you can see what changed and why.</p>
                  {steps.length === 0 && !wakingUp && <p className="mt-5 text-sm text-muted-foreground">Starting the quality check...</p>}
                </CardContent>
              </Card>
              <div>
                <StepList steps={steps.length ? steps : traceSteps.map((x, i) => ({ step: i + 1, tool: x.tool, reason: x.reason, result: "Waiting", status: "ok" as const, duration_ms: 0 }))} active={steps.length} />
                <Button variant="outline" onClick={reset} className="mt-5 rounded-xl">Cancel</Button>
              </div>
            </motion.div>
          )}
          {state === "result" && data && (
            <motion.div key="result" initial="hidden" animate="visible" variants={fade} className="flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">Analysis complete</p>
                <Button variant="outline" onClick={reset} className="rounded-xl">Analyze another photo</Button>
              </div>
              <VerdictCard data={data} onRetake={reset} />
              {!isRetake && (
                <>
                  <div className="grid gap-6 lg:grid-cols-2">
                    <BeforeAfter images={data.images} />
                    <QualityPanel data={data} />
                  </div>
                  <div className="grid gap-6 lg:grid-cols-2">
                    <DiscolorationBar data={data.discoloration} />
                    <CausesList data={data} />
                  </div>
                  <TraceTimeline trace={data.trace} />
                </>
              )}
            </motion.div>
          )}
          {state === "error" && !data && (
            <Card className="rounded-3xl">
              <CardContent className="flex flex-col items-center p-12 text-center">
                <AlertCircle className="text-destructive" />
                <h2 className="mt-4 text-xl font-semibold">We couldn&apos;t finish that analysis.</h2>
                <p className="mt-2 text-muted-foreground">{error}</p>
                <div className="mt-6 flex gap-3">
                  <Button onClick={run} className="rounded-xl">Try again</Button>
                  <Button onClick={reset} variant="outline" className="rounded-xl">Start over</Button>
                </div>
              </CardContent>
            </Card>
          )}
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  )
}

export function LandingPage() { return <div className="min-h-screen bg-background"><Navbar /><main><Hero /><HowItWorks /><FeatureGrid /><DemoSlider /><MeasuresSection /><AudienceCards /><LimitsCard /><Faq /><CtaBand /></main><Footer /></div> }
