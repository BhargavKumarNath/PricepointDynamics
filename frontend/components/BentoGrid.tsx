interface BentoGridProps {
  children: React.ReactNode;
  columns?: 6 | 12;
  className?: string;
}

// Literal class strings, not template-interpolated -- Tailwind's content
// scanner can't see a runtime-built `sm:grid-cols-${columns}` string.
const COLUMN_CLASSES: Record<12 | 6, string> = {
  12: "sm:grid-cols-12",
  6: "sm:grid-cols-6",
};

/** Page-level bento layout wrapper -- replaces the old `space-y-10`
 * stacked-card convention. Children are `GlassCard`s using `span` to
 * claim a fraction of this grid's column count. */
export function BentoGrid({ children, columns = 12, className = "" }: BentoGridProps) {
  return <div className={`grid grid-cols-1 gap-4 ${COLUMN_CLASSES[columns]} ${className}`}>{children}</div>;
}
