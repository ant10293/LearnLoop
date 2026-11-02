/** Model scores are stored from 0 to 1. The overlay shows them as a percent. */
export function formatConfidence(score: number): string {
  return `${Math.round(score * 100)}%`;
}
