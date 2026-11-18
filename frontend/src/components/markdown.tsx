"use client";

import { memo } from "react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import katex from "katex";
import type { Element, ElementContent } from "hast";
import { ImageOff } from "lucide-react";
import { runFileUrl } from "@/lib/api";
import "katex/dist/katex.min.css";

const schema = {
  ...defaultSchema,
  // Preserve table captions from parser HTML instead of leaving bare text inside <table>.
  tagNames: [...(defaultSchema.tagNames ?? []), "caption"],
  attributes: {
    ...defaultSchema.attributes,
    code: [["className", /^language-./, "math-inline", "math-display"]],
  },
};

const LATEX_LANGUAGES = new Set(["language-latex", "language-tex", "language-katex"]);

function textOf(node: ElementContent): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(textOf).join("");
  return "";
}

function stripDelimiters(tex: string): string {
  const t = tex.trim();
  for (const [open, close] of [
    ["$$", "$$"],
    ["\\[", "\\]"],
    ["\\(", "\\)"],
    ["$", "$"],
  ]) {
    if (t.startsWith(open) && t.endsWith(close) && t.length > open.length + close.length) {
      return t.slice(open.length, t.length - close.length).trim();
    }
  }
  return t;
}

function Formula({ source }: { source: string }) {
  const html = katex.renderToString(stripDelimiters(source), {
    displayMode: true,
    throwOnError: false,
    strict: "ignore",
  });
  return <div className="katex-block" dangerouslySetInnerHTML={{ __html: html }} />;
}

function latexSource(node: Element | undefined): string | null {
  const code = node?.children.find((c): c is Element => c.type === "element");
  if (!code || code.tagName !== "code") return null;
  const classes = code.properties?.className;
  const list = Array.isArray(classes) ? classes.map(String) : [];
  return list.some((c) => LATEX_LANGUAGES.has(c)) ? textOf(code) : null;
}

function buildComponents(): Components {
  return {
    pre({ node, children, ...props }) {
      const latex = latexSource(node);
      if (latex !== null) return <Formula source={latex} />;
      return <pre {...props}>{children}</pre>;
    },
    table({ node, ...props }) {
      return (
        <div className="table-wrap">
          <table {...props} />
        </div>
      );
    },
    a({ node, ...props }) {
      return <a {...props} target="_blank" rel="noreferrer" />;
    },
    img({ src, alt }) {
      if (typeof src === "string" && src.startsWith("/api/")) {
        return <img src={src} alt={alt ?? ""} loading="lazy" />;
      }
      return (
        <span className="inline-flex items-center gap-1.5 rounded-lg border border-dashed border-ash px-2 py-1 text-xs text-faint">
          <ImageOff aria-hidden className="size-3.5" />
          {alt ? `Image: ${alt}` : "Image"} (not included in the output)
        </span>
      );
    },
  };
}

const components = buildComponents();

export const Markdown = memo(function Markdown({
  markdown,
  runId,
}: {
  markdown: string;
  runId?: string;
}) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, schema], rehypeKatex]}
        components={components}
        urlTransform={(url, key) => {
          if (key === "src" && runId && /^(\.\/)?artifacts\//.test(url)) {
            return runFileUrl(runId, url);
          }
          return defaultUrlTransform(url);
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
});
