"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const LINKS = [
  { href: "/arena", label: "Arena", match: ["/arena", "/experiments"] },
  { href: "/#models", label: "Models", match: [] },
  { href: "/#how-it-works", label: "How it works", match: [] },
];

export function NavLinkList({ pathname }: { pathname: string | null }) {
  return (
    <ul className="hidden items-center gap-0.5 md:flex">
      {LINKS.map((link) => {
        const active = pathname !== null && link.match.some((m) => pathname.startsWith(m));
        return (
          <li key={link.href}>
            <Link
              href={link.href}
              aria-current={active ? "page" : undefined}
              className={clsx(
                "rounded-xl px-3.5 py-2 text-sm transition-colors",
                active ? "bg-fog text-ink" : "text-body hover:bg-smoke hover:text-ink",
              )}
            >
              {link.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

export function NavLinks() {
  return <NavLinkList pathname={usePathname()} />;
}
