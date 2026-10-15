import { motion } from "framer-motion"
import { motionDuration, motionEase } from "./config"

export const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: motionDuration, ease: motionEase } },
}

export const fade = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.3, ease: motionEase } },
}

export const stagger = { visible: { transition: { staggerChildren: 0.06 } } }

export const MotionSection = ({ children, className = "", id }: { children: React.ReactNode; className?: string; id?: string }) => (
  <motion.section id={id} initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-80px" }} variants={stagger} className={className}>{children}</motion.section>
)

export { motion }
