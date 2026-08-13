type Span = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

interface GlassCardProps {
  children: React.ReactNode;
  span?: Span;
  rowSpan?: 1 | 2;
  variant?: "default" | "strong" | "flat";
  padding?: "sm" | "md" | "lg";
  accent?: boolean;
  as?: "div" | "section" | "article";
  className?: string;
}

// Literal class strings throughout -- Tailwind's content scanner needs
// to see them verbatim in source, not built from a runtime template.
const SPAN_CLASSES: Record<Span, string> = {
  1: "sm:col-span-1",
  2: "sm:col-span-2",
  3: "sm:col-span-3",
  4: "sm:col-span-4",
  5: "sm:col-span-5",
  6: "sm:col-span-6",
  7: "sm:col-span-7",
  8: "sm:col-span-8",
  9: "sm:col-span-9",
  10: "sm:col-span-10",
  11: "sm:col-span-11",
  12: "sm:col-span-12",
};

const ROW_SPAN_CLASSES: Record<1 | 2, string> = {
  1: "",
  2: "sm:row-span-2",
};

const PADDING_CLASSES: Record<"sm" | "md" | "lg", string> = {
  sm: "p-4",
  md: "p-5",
  lg: "p-8",
};

/**
 * The bento cell primitive -- replaces every ad hoc
 * `rounded-lg border border-border bg-surface p-5` card shell.
 *
 * `variant="flat"` opts out of the blur (solid `bg-surface` instead):
 * mandatory for dense numeric tables, where blurring behind rows of
 * numbers hurts scan-ability and wastes paint for no visual benefit.
 *
 * `accent` adds the brand glow ring -- use on at most one "hero" card
 * per page, never as a general-purpose highlight.
 */
export function GlassCard({
  children,
  span = 12,
  rowSpan = 1,
  variant = "default",
  padding = "md",
  accent = false,
  as: Tag = "div",
  className = "",
}: GlassCardProps) {
  const surface =
    variant === "flat"
      ? "bg-surface"
      : variant === "strong"
        ? "bg-glass-surface-strong backdrop-blur-glass"
        : "bg-glass-surface backdrop-blur-glass";

  return (
    <Tag
      className={`rounded-bento border border-glass-border ${surface} ${PADDING_CLASSES[padding]} ${SPAN_CLASSES[span]} ${ROW_SPAN_CLASSES[rowSpan]} ${className}`}
      style={{
        boxShadow: accent ? "var(--shadow-glass), var(--glow-brand)" : "var(--shadow-glass)",
      }}
    >
      {children}
    </Tag>
  );
}
