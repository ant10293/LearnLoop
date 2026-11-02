/** Column for a label sitting on a box. It may hang past the box, and it stops short of a neighbor. */
export function labelSpan(
  x: number,
  width: number,
  others: { x: number; width: number }[],
  past = 14,
  gap = 4,
): { x: number; width: number } {
  let left = x - past / 2;
  let right = x + width + past;
  for (const other of others) {
    const otherRight = other.x + other.width;
    if (otherRight <= x) left = Math.max(left, otherRight + gap);
    else if (other.x >= x + width) right = Math.min(right, other.x - gap);
  }
  left = Math.min(left, x);
  right = Math.max(right, x + width);
  return { x: left, width: Math.max(right - left, 1) };
}

/** Greedy word wrap. A line stays within maxWidth, except a single word that is wider than the column. */
export function wrapLines(text: string, maxWidth: number, measure: (value: string) => number): string[] {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  if (!(maxWidth > 0)) return [words.join(" ")];

  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (!current || measure(next) <= maxWidth) {
      current = next;
      continue;
    }
    lines.push(current);
    current = word;
  }
  if (current) lines.push(current);
  return lines;
}
