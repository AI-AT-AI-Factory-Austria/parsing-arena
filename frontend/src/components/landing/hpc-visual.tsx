const RACKS = [0.86, 1, 0.92, 0.8];
const BLADES = 9;

/** Decorative: a stack of documents flowing into a row of HPC racks. */
export function HpcVisual() {
  return (
    <div aria-hidden className="relative mx-auto aspect-[5/4] w-full max-w-[560px] select-none">
      <div className="absolute -right-[6%] -top-[10%] aspect-square w-[22%] rounded-full bg-mint" />
      <div className="absolute bottom-[3%] left-[3%] aspect-square w-[15%] rounded-full border-2 border-white/30" />

      <div className="absolute bottom-[9%] right-[3%] flex h-[80%] w-[62%] items-end gap-[3.5%]">
        {RACKS.map((height, rack) => (
          <div
            key={rack}
            className="flex flex-1 flex-col gap-[3.5%] rounded-2xl bg-white/10 p-[7%] ring-1 ring-white/25"
            style={{ height: `${height * 100}%` }}
          >
            <div className="mb-[4%] h-1 w-1/2 rounded-full bg-white/40" />
            {Array.from({ length: BLADES }, (_, blade) => (
              <div
                key={blade}
                className="flex flex-1 items-center gap-[10%] rounded-[5px] bg-white/12 px-[10%]"
              >
                <span
                  className="aspect-square w-[14%] shrink-0 animate-blink rounded-full bg-mint"
                  style={{ animationDelay: `${((rack * BLADES + blade) * 263) % 1900}ms` }}
                />
                <span className="h-[2px] flex-1 rounded-full bg-white/30" />
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="absolute left-[3%] top-[14%] aspect-[3/4] w-[33%]">
        {[2, 1, 0].map((i) => (
          <div
            key={i}
            className="card absolute inset-0 flex flex-col gap-[7%] p-[11%]"
            style={{ transform: `translate(${i * 9}%, ${i * 7}%) rotate(${-7 + i * 4}deg)` }}
          >
            <div className="h-[5%] w-3/4 rounded-full bg-ink/80" />
            <div className="h-[3%] w-full rounded-full bg-ash/70" />
            <div className="h-[3%] w-11/12 rounded-full bg-ash/70" />
            <div className="h-[3%] w-4/5 rounded-full bg-ash/70" />
            <div className="mt-[4%] grid flex-1 grid-cols-3 gap-[6%]">
              {Array.from({ length: 6 }, (_, cell) => (
                <div key={cell} className={cell < 3 ? "rounded bg-brand-soft" : "rounded bg-fog"} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="absolute bottom-[1%] left-[22%] flex items-center gap-2 rounded-full bg-ink px-3 py-1.5 font-mono text-[10px] text-white shadow-float sm:text-[11px]">
        <span className="size-1.5 rounded-full bg-mint" /> corpus → Markdown at scale
      </div>
    </div>
  );
}
