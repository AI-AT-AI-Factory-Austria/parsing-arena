import Link from "next/link";
import { Suspense } from "react";
import { ArrowRight } from "lucide-react";
import { Logo } from "./logo";
import { NavLinkList, NavLinks } from "./nav-links";

export function SiteHeader() {
  return (
    // Zero height so the floating bar overlays the page like on ai-at.eu.
    <header className="pointer-events-none sticky top-0 z-50 h-0">
      <div className="mx-auto grid max-w-[1320px] grid-cols-[1fr_auto] items-center gap-4 px-4 pt-3 sm:px-6 sm:pt-4 md:grid-cols-[1fr_auto_1fr]">
        <Logo className="pointer-events-auto justify-self-start" />
        <nav
          aria-label="Main"
          className="pointer-events-auto flex items-center gap-1 rounded-2xl bg-white/95 p-1.5 shadow-float ring-1 ring-black/5 backdrop-blur"
        >
          <Suspense fallback={<NavLinkList pathname={null} />}>
            <NavLinks />
          </Suspense>
          <Link href="/arena" className="btn btn-primary btn-sm md:hidden">
            Try it <ArrowRight className="size-3.5" />
          </Link>
        </nav>
        <Link
          href="/arena"
          className="btn btn-primary pointer-events-auto hidden justify-self-end shadow-float md:inline-flex"
        >
          Try it with your documents <ArrowRight className="size-4" />
        </Link>
      </div>
    </header>
  );
}
