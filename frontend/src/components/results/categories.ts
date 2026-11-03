const COLORS: Record<string, string> = {
  Title: "#7c3aed",
  "Section-header": "#7c3aed",
  Text: "#005af0",
  "List-item": "#0e7490",
  Table: "#48992e",
  Chart: "#d97706",
  Picture: "#d97706",
  Figure: "#d97706",
  Formula: "#db2777",
  Caption: "#616161",
  Footnote: "#616161",
  "Page-header": "#a1a1a1",
  "Page-footer": "#a1a1a1",
};

export function categoryColor(category: string): string {
  return COLORS[category] ?? "#616161";
}

export function categoryLabel(category: string): string {
  return category.replace(/-/g, " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());
}
