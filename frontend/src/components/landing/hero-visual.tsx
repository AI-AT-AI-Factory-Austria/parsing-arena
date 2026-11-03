const LANES = [
  { name: "rakedoc-nano", meta: "1.2B", width: "92%", duration: "3.4s", delay: "0s" },
  { name: "Aeddix Alpine OCR", meta: "1.2B", width: "86%", duration: "3.8s", delay: "0.35s" },
  { name: "LlamaParse Agentic Plus", meta: "API", width: "100%", duration: "4.4s", delay: "0.7s" },
];

/** Decorative: a document being read and several parsers racing on it. */
export function HeroVisual() {
  return (
    <div aria-hidden className="relative mx-auto aspect-[6/5] w-full max-w-[620px] select-none">
      <div className="absolute right-0 top-0 h-[64%] w-[80%] rounded-[30px] bg-gradient-to-br from-[#c2c2c2] via-[#d8d8d8] to-[#ededed]" />
      <div className="absolute bottom-[2%] left-0 h-[34%] w-[52%] rounded-[30px] bg-white/70" />
      <div className="absolute left-[3%] top-[2%] aspect-square w-[26%] rounded-full bg-mint" />
      <div className="absolute bottom-[5%] right-[5%] aspect-square w-[13%] rounded-full bg-brand" />

      {/* The document */}
      <div className="card absolute left-[9%] top-[11%] flex aspect-[3/4] w-[46%] -rotate-2 flex-col gap-[4.5%] overflow-hidden p-[5.5%]">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[9px] text-faint sm:text-[10px]">q3-report.pdf</span>
          <span className="rounded-full bg-fog px-1.5 font-mono text-[8px] text-muted sm:text-[9px]">
            p. 1
          </span>
        </div>
        <div className="h-[4%] w-3/4 rounded-full bg-ink/85" />
        <div className="space-y-[5%]">
          <div className="h-1.5 w-full rounded-full bg-ash/70" />
          <div className="h-1.5 w-11/12 rounded-full bg-ash/70" />
          <div className="h-1.5 w-4/5 rounded-full bg-ash/70" />
        </div>
        <div className="grid flex-1 grid-cols-3 overflow-hidden rounded-lg border border-line">
          {Array.from({ length: 12 }, (_, i) => (
            <div
              key={i}
              className={`border-b border-r border-line ${i < 3 ? "bg-smoke" : ""} ${i % 3 === 2 ? "border-r-0" : ""}`}
            >
              <div
                className={`m-[14%] h-1 rounded-full ${i < 3 ? "bg-ink/50" : "bg-ash/60"}`}
                style={{ width: `${45 + ((i * 17) % 40)}%` }}
              />
            </div>
          ))}
        </div>
        <div className="flex items-end gap-[5%]">
          {[38, 62, 48, 80, 70, 92].map((h, i) => (
            <div
              key={i}
              className={`flex-1 rounded-t ${i % 2 ? "bg-brand" : "bg-brand-soft"}`}
              style={{ height: `${h * 0.32}px` }}
            />
          ))}
        </div>
        <div className="absolute inset-x-[5%] h-[2px] animate-scan rounded-full bg-brand shadow-[0_0_18px_5px_rgba(0,90,240,0.28)]" />
      </div>

      {/* The parsers */}
      <div className="absolute right-[3%] top-[20%] flex w-[47%] flex-col gap-3">
        {LANES.map((lane) => (
          <div key={lane.name} className="card rounded-2xl px-3.5 py-3">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-mono text-[10px] text-ink sm:text-[11px]">
                {lane.name}
              </span>
              <span
                className={`shrink-0 rounded-full px-1.5 py-px font-mono text-[9px] sm:text-[10px] ${
                  lane.meta === "API" ? "bg-mint text-leaf-strong" : "bg-brand-wash text-brand"
                }`}
              >
                {lane.meta}
              </span>
            </div>
            <div className="mt-2.5 h-1.5 rounded-full bg-fog">
              <div className="h-full rounded-full bg-fog" style={{ width: lane.width }}>
                <div
                  className={`h-full origin-left animate-race rounded-full ${lane.meta === "API" ? "bg-leaf" : "bg-brand"}`}
                  style={{ animationDuration: lane.duration, animationDelay: lane.delay }}
                />
              </div>
            </div>
          </div>
        ))}
        <div className="ml-auto flex items-center gap-2 rounded-full bg-ink px-3 py-1.5 font-mono text-[10px] text-white shadow-float sm:text-[11px]">
          <span className="size-1.5 rounded-full bg-mint" /> side-by-side results
        </div>
      </div>
    </div>
  );
}
