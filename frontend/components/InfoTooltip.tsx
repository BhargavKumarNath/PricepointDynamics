"use client";

import { useEffect, useId, useRef, useState } from "react";
import { InfoIcon } from "@/components/icons";

interface InfoTooltipProps {
  label: string;
  align?: "start" | "end";
  children: React.ReactNode;
}

/**
 * The "How to read this chart" popover, used on every chart card.
 * No new dependency: a controlled disclosure (not native `<details>`,
 * which doesn't close on outside-click) built from `useState` + one
 * `useEffect` for outside-pointerdown/Escape handling.
 *
 * Content here is always supplementary reading guidance, never the
 * only place a number lives -- every value referenced must also be
 * visible in the chart/table itself.
 */
export function InfoTooltip({ label, align = "start", children }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
        className="flex h-4 w-4 items-center justify-center rounded-full text-text-muted transition-colors hover:text-brand"
      >
        <InfoIcon size={14} />
        <span className="sr-only">{label}</span>
      </button>
      {open && (
        <div
          id={panelId}
          role="group"
          aria-label={label}
          className={`absolute top-full z-20 mt-2 w-72 rounded-bento-sm border border-glass-border bg-glass-surface-strong p-3 text-xs leading-relaxed text-text-secondary backdrop-blur-glass-sm ${
            align === "end" ? "right-0" : "left-0"
          }`}
          style={{ boxShadow: "var(--shadow-glass)" }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
