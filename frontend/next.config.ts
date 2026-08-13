import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Zero-wait design (project_refactor.md §25.4): every page except the
  // Predictor is fully static -- `output: "export"` produces plain
  // HTML/CSS/JS with no Node server in the request path, deployed as-is
  // to Vercel's edge CDN. The Predictor's live prediction call happens
  // client-side against the FastAPI service, not through Next.js at all.
  output: "export",
};

export default nextConfig;
