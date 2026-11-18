import Link from "next/link";
import { Suspense } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Columns3,
  FileUp,
  Gauge,
  LockKeyhole,
  PackageOpen,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { HeroVisual } from "@/components/landing/hero-visual";
import { HpcVisual } from "@/components/landing/hpc-visual";
import { ModelsTable, ModelsTableSkeleton } from "@/components/landing/models-table";

const HPC_URL = "https://ai-at.eu/en/services/exclusive-access-to-the-european-hpc-infrastructure/";

const PRINCIPLES = [
  {
    icon: ShieldCheck,
    title: "Made for sensitive documents",
    text: "Contracts, medical records, internal reports: the open models run on the arena's own hardware. A document only goes to an external API if you add the LlamaParse reference to a run yourself.",
  },
  {
    icon: PackageOpen,
    title: "Open OCR models only",
    text: "Every parser in the line-up is open source with published weights and a license, so the one you pick can later run on your own servers or on HPC, without vendor lock-in.",
  },
  {
    icon: Gauge,
    title: "Test before you build",
    text: "Compare output quality page by page and runtime per page on identical hardware, then choose the model that fits your application's accuracy and speed requirements.",
  },
];

const STEPS = [
  {
    icon: FileUp,
    title: "Drop in a document",
    text: "PDF, PNG, JPG, WebP or TIFF. Test the whole file or pick the pages that matter, such as the ones with tables, charts or formulas.",
  },
  {
    icon: Timer,
    title: "Run the arena",
    text: "The open models parse the pages one after another on the same hardware, so their runtimes are comparable. Optionally add LlamaParse Agentic Plus as a commercial reference.",
  },
  {
    icon: Columns3,
    title: "Compare side by side",
    text: "See every model's output next to the original page, rendered or as raw Markdown, with seconds per page and the detected layout.",
  },
];

const HPC_STEPS = [
  "Test open models on a sample of your documents here",
  "Pick the model that fits your application",
  "Parse the full corpus on HPC with AI Factory Austria",
];

export default function HomePage() {
  return (
    <>
      <section className="relative overflow-hidden bg-fog">
        <div className="mx-auto grid max-w-[1320px] items-center gap-14 px-4 pb-20 pt-32 sm:px-6 lg:grid-cols-[1.02fr_1fr] lg:gap-10 lg:pb-28 lg:pt-40">
          <div className="max-w-2xl">
            <p className="eyebrow">Parsing Arena · by AI Factory Austria</p>
            <h1 className="mt-5 text-[2.6rem] leading-[1.04] tracking-[-0.025em] text-ink sm:text-6xl lg:text-[4.1rem]">
              Find the open OCR model for your sensitive documents
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-body">
              A testing ground for choosing the OCR model behind your application. Run open-source
              models of different sizes on your own documents, compare their output side by side
              and measure the runtime per page before you commit to one.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/arena" className="btn btn-primary btn-lg">
                Try it with your documents <ArrowRight className="size-[18px]" />
              </Link>
              <Link href="#models" className="btn btn-outline btn-lg">
                See the models
              </Link>
            </div>
            <p className="mt-5 flex max-w-xl items-start gap-2 text-sm leading-relaxed text-muted">
              <LockKeyhole aria-hidden className="mt-0.5 size-4 shrink-0 text-leaf" />
              Open models run on the arena&apos;s own hardware. Nothing is sent to a third-party
              API unless you switch on the optional LlamaParse reference.
            </p>
            <dl className="mt-10 grid max-w-2xl grid-cols-3 gap-6 border-t border-ash/70 pt-6">
              <div>
                <dt className="eyebrow text-[11px]">Line-up</dt>
                <dd className="mt-1 text-2xl tracking-tight">24 open parsers</dd>
              </div>
              <div>
                <dt className="eyebrow text-[11px]">Compared on</dt>
                <dd className="mt-1 text-2xl tracking-tight">Quality & speed</dd>
              </div>
              <div>
                <dt className="eyebrow text-[11px]">External APIs</dt>
                <dd className="mt-1 text-2xl tracking-tight">Opt-in only</dd>
              </div>
            </dl>
          </div>
          <HeroVisual />
        </div>
      </section>

      <section aria-labelledby="principles-heading" className="bg-white">
        <div className="mx-auto max-w-[1320px] px-4 pt-20 sm:px-6 lg:pt-28">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="eyebrow">Why Parsing Arena</p>
              <h2
                id="principles-heading"
                className="mt-4 max-w-2xl text-3xl leading-tight tracking-[-0.02em] sm:text-[2.6rem]"
              >
                Choose an OCR model with evidence from your own documents
              </h2>
            </div>
            <p className="max-w-md leading-relaxed text-muted">
              Leaderboard averages don&apos;t tell you how a model handles your invoices, forms or
              scans. The arena does, without your data leaving controlled infrastructure.
            </p>
          </div>
          <ul className="mt-12 grid gap-5 md:grid-cols-3">
            {PRINCIPLES.map((item) => (
              <li
                key={item.title}
                className="rounded-[var(--radius-tile)] border border-line p-7 transition-colors hover:border-ash"
              >
                <span className="grid size-11 place-items-center rounded-2xl bg-mint-soft text-leaf-strong">
                  <item.icon aria-hidden className="size-5" />
                </span>
                <h3 className="mt-8 text-xl tracking-tight">{item.title}</h3>
                <p className="mt-3 leading-relaxed text-muted">{item.text}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section id="how-it-works" className="bg-white">
        <div className="mx-auto max-w-[1320px] px-4 py-20 sm:px-6 lg:py-28">
          <p className="eyebrow">How it works</p>
          <h2 className="mt-4 max-w-2xl text-3xl leading-tight tracking-[-0.02em] sm:text-[2.6rem]">
            From a dropped file to a side-by-side verdict in three steps
          </h2>
          <ol className="mt-12 grid gap-5 md:grid-cols-3">
            {STEPS.map((step, i) => (
              <li key={step.title} className="rounded-[var(--radius-tile)] bg-smoke p-7">
                <div className="flex items-center justify-between">
                  <span className="grid size-11 place-items-center rounded-2xl bg-brand-soft text-brand">
                    <step.icon aria-hidden className="size-5" />
                  </span>
                  <span className="font-mono text-sm text-stone">0{i + 1}</span>
                </div>
                <h3 className="mt-8 text-xl tracking-tight">{step.title}</h3>
                <p className="mt-3 leading-relaxed text-muted">{step.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section id="models" className="bg-smoke">
        <div className="mx-auto max-w-[1320px] px-4 py-20 sm:px-6 lg:py-28">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="eyebrow">The line-up</p>
              <h2 className="mt-4 text-3xl leading-tight tracking-[-0.02em] sm:text-[2.6rem]">
                Open models in the arena
              </h2>
            </div>
            <p className="max-w-xl leading-relaxed text-muted">
              From multi-page document models to lightweight text extraction. Explore each
              method&apos;s focus, ordered roughly by size, and try the ones that fit your documents.
            </p>
          </div>
          <div className="mt-10">
            <Suspense fallback={<ModelsTableSkeleton />}>
              <ModelsTable />
            </Suspense>
          </div>
          <p className="mt-5 text-sm text-faint">
            M / B = million / billion parameters. Pipeline sizes are approximate; + indicates
            auxiliary models. Each method uses its own extraction settings. Compare the results
            on your documents to find the best fit.
          </p>
        </div>
      </section>

      <section id="hpc" aria-labelledby="hpc-heading" className="bg-white">
        <div className="mx-auto max-w-[1320px] px-4 py-20 sm:px-6 lg:py-28">
          <div className="grid items-center gap-12 overflow-hidden rounded-[var(--radius-panel)] bg-brand px-6 py-12 text-white sm:px-10 lg:grid-cols-[1.05fr_1fr] lg:gap-8 lg:px-14 lg:py-16">
            <div>
              <p className="eyebrow text-white/70">Scale up with AI Factory Austria</p>
              <h2
                id="hpc-heading"
                className="mt-4 text-3xl leading-tight tracking-[-0.02em] sm:text-[2.6rem]"
              >
                Find the right model. Scale up with HPC.
              </h2>
              <p className="mt-5 max-w-xl text-lg leading-relaxed text-white/85">
                Once you know which model reads your documents best, AI Factory Austria (AI:AT)
                can support you in applying for access to European high-performance computing.
                Onboarding and technical support help you prepare to run your chosen model over
                large document collections.
              </p>
              <ol className="mt-8 space-y-3">
                {HPC_STEPS.map((step, i) => (
                  <li key={step} className="flex items-center gap-3">
                    <span className="grid size-7 shrink-0 place-items-center rounded-full bg-white/15 font-mono text-xs">
                      {i + 1}
                    </span>
                    <span className="text-white/90">{step}</span>
                  </li>
                ))}
              </ol>
              <div className="mt-10 flex flex-wrap items-center gap-3">
                <a
                  href={HPC_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-lg bg-white text-brand hover:bg-brand-wash"
                >
                  Ask about HPC access <ArrowUpRight aria-hidden className="size-[18px]" />
                </a>
                <Link
                  href="/arena"
                  className="btn btn-lg border-white/60 text-white hover:border-white hover:bg-white/10"
                >
                  Start testing first
                </Link>
              </div>
              <p className="mt-4 text-xs text-white/60">
                Access to EuroHPC resources through AI:AT is subject to the programme&apos;s
                conditions.
              </p>
            </div>
            <HpcVisual />
          </div>
        </div>
      </section>

      <section className="bg-mint">
        <div className="mx-auto flex max-w-[1320px] flex-col items-start gap-8 px-4 py-16 sm:px-6 md:flex-row md:items-center md:justify-between lg:py-20">
          <div>
            <p className="eyebrow text-leaf-strong">Your documents, your verdict</p>
            <h2 className="mt-3 max-w-2xl text-3xl leading-tight tracking-[-0.02em] sm:text-4xl">
              Ready to see how the open models handle your files?
            </h2>
          </div>
          <Link href="/arena" className="btn btn-dark btn-lg">
            Open the arena <ArrowRight className="size-[18px]" />
          </Link>
        </div>
      </section>
    </>
  );
}
