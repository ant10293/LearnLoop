/** Turn a raw model response into indented lines for the log view. */
export function formatLogText(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const parsed = parseJson(stripMarkdownFence(trimmed));
  if (parsed !== undefined) return JSON.stringify(parsed, null, 2);
  return value;
}

function stripMarkdownFence(text: string): string {
  const match = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return match ? match[1].trim() : text;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}
