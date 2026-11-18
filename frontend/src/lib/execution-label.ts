import type { Availability } from "./types";

export function executionLabel(availability: Availability): string {
  if (availability.execution_label) return availability.execution_label;
  // Older backends may not yet report hardware; never infer CPU from "local".
  switch (availability.backend) {
    case "api": return "Hosted API";
    case "endpoint": return "Remote model server";
    case "cpu": return "Local CPU";
    default: return "Local runtime";
  }
}
