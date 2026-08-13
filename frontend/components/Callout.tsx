import { CheckIcon, CriticalIcon, SparkleIcon, WarningIcon } from "@/components/icons";

export type CalloutTone = "insight" | "good" | "warning" | "critical" | "neutral";

interface CalloutProps {
  tone?: CalloutTone;
  title?: string;
  children: React.ReactNode;
}

const TONE_CONFIG: Record<
  CalloutTone,
  { defaultTitle: string; border: string; wash: string; icon: React.ReactNode | null }
> = {
  insight: { defaultTitle: "Insight", border: "border-l-brand", wash: "bg-brand-soft", icon: <SparkleIcon size={14} /> },
  good: {
    defaultTitle: "Good",
    border: "border-l-status-good",
    wash: "bg-status-good/10",
    icon: <CheckIcon size={14} />,
  },
  warning: {
    defaultTitle: "Warning",
    border: "border-l-status-warning",
    wash: "bg-status-warning/10",
    icon: <WarningIcon size={14} />,
  },
  critical: {
    defaultTitle: "Critical",
    border: "border-l-status-critical",
    wash: "bg-status-critical/10",
    icon: <CriticalIcon size={14} />,
  },
  neutral: { defaultTitle: "Note", border: "border-l-text-muted", wash: "bg-text-muted/8", icon: null },
};

/** Replaces every plain `<strong>Insight:</strong>` paragraph. Status
 * colors never carry meaning alone -- each tone pairs an icon with a
 * title, never color-only. */
export function Callout({ tone = "insight", title, children }: CalloutProps) {
  const config = TONE_CONFIG[tone];
  return (
    <div className={`rounded-bento-sm border-l-4 p-3 text-xs text-text-secondary ${config.border} ${config.wash}`}>
      <div className="mb-1 flex items-center gap-1.5 font-medium text-text-primary">
        {config.icon}
        {title ?? config.defaultTitle}
      </div>
      {children}
    </div>
  );
}
