import Link from "next/link";
import clsx from "clsx";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" className={className}>
      <rect width="64" height="64" rx="16" fill="#005AF0" />
      <g fill="none" stroke="#fff" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 24v-4a4 4 0 0 1 4-4h4M40 16h4a4 4 0 0 1 4 4v4M48 40v4a4 4 0 0 1-4 4h-4M24 48h-4a4 4 0 0 1-4-4v-4" />
        <path d="M24 27h16M24 32h11M24 37h16" />
      </g>
    </svg>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <Link
      href="/"
      aria-label="OCR Arena home"
      className={clsx(
        "group inline-flex items-center gap-3 rounded-2xl bg-white/95 p-1.5 pr-4 shadow-float ring-1 ring-black/5 backdrop-blur",
        className,
      )}
    >
      <LogoMark className="size-10 transition-transform duration-200 group-hover:-rotate-3" />
      <span className="font-mono text-[11px] font-medium uppercase leading-[1.2] tracking-[0.08em] text-ink">
        OCR
        <br />
        Arena
      </span>
    </Link>
  );
}
