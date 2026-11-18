import { LogoMark } from "./logo";

export function SiteFooter() {
  return (
    <footer className="border-t border-line bg-white">
      <div className="mx-auto flex max-w-[1320px] flex-col gap-4 px-4 py-8 text-sm sm:px-6 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <LogoMark className="size-7" />
          <p className="text-muted">
            Parsing Arena by{" "}
            <a
              href="https://ai-at.eu/"
              className="underline decoration-ash underline-offset-2 hover:text-ink"
              target="_blank"
              rel="noreferrer"
            >
              AI Factory Austria (AI:AT)
            </a>
            {" "}· test open OCR models on your own documents.
          </p>
        </div>
        <p className="text-faint">
          Optional reference: LlamaParse Agentic Plus
        </p>
      </div>
    </footer>
  );
}
