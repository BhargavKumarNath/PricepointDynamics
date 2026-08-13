"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/market-overview", label: "Market Overview" },
  { href: "/basket-analysis", label: "Basket Analysis" },
  { href: "/predictor", label: "Price Predictor" },
  { href: "/model-insights", label: "Model Insights" },
  { href: "/market-dynamics", label: "Market Dynamics" },
] as const;

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-30 border-b border-glass-border bg-glass-surface-strong backdrop-blur-glass">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-1 px-4 py-3">
        <span className="mr-4 font-semibold text-text-primary">PricePoint Dynamics</span>
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-bento-sm px-3 py-1.5 text-sm transition-colors ${
                active ? "bg-brand text-page" : "text-text-secondary hover:text-text-primary hover:bg-brand-soft"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
