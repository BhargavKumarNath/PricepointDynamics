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
    <nav className="border-b border-border bg-surface">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-1 px-4 py-3">
        <span className="mr-4 font-semibold text-text-primary">PricePoint Dynamics</span>
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded px-3 py-1.5 text-sm transition-colors ${
                active ? "bg-text-primary text-page" : "text-text-secondary hover:text-text-primary"
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
