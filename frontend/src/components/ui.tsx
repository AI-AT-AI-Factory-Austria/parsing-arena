import clsx from "clsx";
import { Check, LoaderCircle, X } from "lucide-react";
import type { ExperimentStatus, RunStatus } from "@/lib/types";

type Tone = "neutral" | "brand" | "mint" | "amber" | "danger" | "outline";

const TONES: Record<Tone, string> = {
  neutral: "bg-fog text-muted",
  brand: "bg-brand-soft text-brand-strong",
  mint: "bg-mint-soft text-leaf-strong",
  amber: "bg-amber-wash text-amber",
  danger: "bg-danger-wash text-danger",
  outline: "border border-line text-muted",
};

export function Chip({
  tone = "neutral",
  className,
  title,
  children,
}: {
  tone?: Tone;
  className?: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      title={title}
      className={clsx(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dot({ className }: { className?: string }) {
  return <span aria-hidden className={clsx("size-1.5 rounded-full bg-current", className)} />;
}

export function Spinner({ className }: { className?: string }) {
  return <LoaderCircle aria-hidden className={clsx("animate-spin", className ?? "size-3.5")} />;
}

const STATUS: Record<
  RunStatus | ExperimentStatus,
  { label: string; tone: Tone; icon: "spin" | "check" | "x" | "dot" }
> = {
  queued: { label: "Queued", tone: "neutral", icon: "dot" },
  loading: { label: "Loading model", tone: "brand", icon: "spin" },
  running: { label: "Running", tone: "brand", icon: "spin" },
  completed: { label: "Done", tone: "mint", icon: "check" },
  failed: { label: "Failed", tone: "danger", icon: "x" },
};

export function StatusChip({
  status,
  label,
  title,
}: {
  status: RunStatus | ExperimentStatus;
  label?: string;
  title?: string;
}) {
  const s = STATUS[status];
  return (
    <Chip tone={s.tone} title={title}>
      {s.icon === "spin" && <Spinner className="size-3" />}
      {s.icon === "check" && <Check aria-hidden className="size-3" strokeWidth={2.5} />}
      {s.icon === "x" && <X aria-hidden className="size-3" strokeWidth={2.5} />}
      {s.icon === "dot" && <Dot />}
      {label ?? s.label}
    </Chip>
  );
}

export function ProgressBar({
  value,
  indeterminate = false,
  tone = "brand",
  className,
  label,
}: {
  value?: number;
  indeterminate?: boolean;
  tone?: "brand" | "mint" | "ink";
  className?: string;
  label?: string;
}) {
  const fill = { brand: "bg-brand", mint: "bg-leaf", ink: "bg-ink" }[tone];
  const pct = Math.round(Math.min(1, Math.max(0, value ?? 0)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : pct}
      className={clsx("h-1.5 overflow-hidden rounded-full bg-fog", className)}
    >
      <div
        className={clsx(
          "h-full rounded-full transition-[width] duration-500 ease-out",
          fill,
          indeterminate && "stripes animate-stripes",
        )}
        style={{ width: indeterminate ? "100%" : pct > 0 ? `${Math.max(pct, 2)}%` : "0%" }}
      />
    </div>
  );
}
