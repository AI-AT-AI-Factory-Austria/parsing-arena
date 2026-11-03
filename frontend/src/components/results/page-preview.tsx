"use client";

import { useState } from "react";
import clsx from "clsx";
import type { PageElement } from "@/lib/types";
import { categoryColor, categoryLabel } from "./categories";

export function PagePreview({
  src,
  alt,
  elements,
}: {
  src: string;
  alt: string;
  elements: PageElement[] | null;
}) {
  const [loaded, setLoaded] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div className="relative overflow-hidden rounded-xl border border-line bg-white">
      {!loaded && <div className="aspect-[1/1.414] w-full animate-pulse bg-smoke" />}
      <img
        src={src}
        alt={alt}
        onLoad={() => setLoaded(true)}
        className={clsx("block h-auto w-full", !loaded && "absolute inset-0 opacity-0")}
      />
      {loaded &&
        elements?.map((el, i) => {
          const [x1, y1, x2, y2] = el.bbox;
          const color = categoryColor(el.category);
          return (
            <div
              key={i}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
              title={`${el.layout_order + 1}. ${categoryLabel(el.category)}`}
              className="absolute rounded-[3px] border-[1.5px] transition-colors"
              style={{
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
                borderColor: color,
                backgroundColor: hovered === i ? `${color}26` : `${color}0d`,
              }}
            >
              <span
                className="absolute -top-[1.5px] left-[-1.5px] -translate-y-full rounded-t-[3px] px-1 font-mono text-[9px] leading-[14px] text-white"
                style={{ backgroundColor: color }}
              >
                {el.layout_order + 1}
              </span>
            </div>
          );
        })}
    </div>
  );
}

export function CategoryLegend({ elements }: { elements: PageElement[] }) {
  const counts = new Map<string, number>();
  for (const el of elements) counts.set(el.category, (counts.get(el.category) ?? 0) + 1);
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1.5 text-[11px] text-muted">
      {[...counts].map(([category, count]) => (
        <li key={category} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="size-2.5 rounded-[3px] border-[1.5px]"
            style={{ borderColor: categoryColor(category) }}
          />
          {categoryLabel(category)} <span className="text-stone">{count}</span>
        </li>
      ))}
    </ul>
  );
}
