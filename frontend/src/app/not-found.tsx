import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-1 items-center bg-fog">
      <div className="mx-auto w-full max-w-[1320px] px-4 pb-24 pt-36 sm:px-6">
        <p className="eyebrow">404</p>
        <h1 className="mt-4 max-w-xl text-4xl tracking-[-0.02em] sm:text-5xl">
          This page isn&apos;t part of the arena.
        </h1>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/arena" className="btn btn-primary">
            Open the arena
          </Link>
          <Link href="/" className="btn btn-outline">
            Home
          </Link>
        </div>
      </div>
    </div>
  );
}
