import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { APP_NAME, APP_TAGLINE } from "@/lib/config"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })

export const metadata: Metadata = {
  title: `${APP_NAME} — ${APP_TAGLINE}`,
  description: "A transparent visual review for plant leaves. Fix the photo, measure what is visible, and understand every step.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" suppressHydrationWarning><body className={`${inter.variable} font-sans antialiased`}>{children}</body></html>
}
