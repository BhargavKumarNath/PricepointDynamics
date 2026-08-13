import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Nav } from "@/components/Nav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PricePoint Dynamics",
  description: "An interactive dashboard for the UK supermarket competitive landscape.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      {/* No bg-page utility here -- globals.css's `body { background: var(--page-gradient) }`
          tag-selector rule provides both the base page color and the subtle
          gradient wash; a `.bg-page` class would out-specificity it and
          silently flatten the gradient back to a solid color. */}
      <body className="min-h-full flex flex-col text-text-primary">
        <Nav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-border px-4 py-4 text-center text-xs text-text-muted">
          Data: 5 UK supermarkets, 9.5M+ price records. Built with FastAPI + Next.js.
        </footer>
      </body>
    </html>
  );
}
