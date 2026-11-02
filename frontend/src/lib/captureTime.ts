/** Local value for `<input type="datetime-local">`, and the offset form the API stores. */

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

function normalizeStamp(raw: string): { text: string; aware: boolean } | null {
  let text = raw.trim();
  if (!text) return null;
  if (text.length >= 19 && text[10] === " ") text = `${text.slice(0, 10)}T${text.slice(11)}`;
  text = text.replace(/(\.\d{3})\d+/, "$1");
  text = text.replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
  const aware = /Z$/i.test(text) || /[+-]\d{2}:\d{2}$/.test(text);
  return { text, aware };
}

function wallClock(text: string): string {
  const match = text.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?/);
  if (!match) return "";
  return `${match[1]}T${match[2]}:${match[3] ?? "00"}`;
}

function formatLocal(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function captureToLocalInput(raw: string | null | undefined, utc?: string | null): string {
  const source = (utc && utc.trim()) || raw || "";
  const parsed = normalizeStamp(source);
  if (!parsed) return "";
  if (!parsed.aware) return wallClock(parsed.text);
  const date = new Date(parsed.text);
  if (Number.isNaN(date.getTime())) return wallClock(parsed.text);
  return formatLocal(date);
}

export function localInputToCapture(value: string): string {
  if (!value) return "";
  const withSeconds = value.length === 16 ? `${value}:00` : value;
  const date = new Date(withSeconds);
  if (Number.isNaN(date.getTime())) return value;
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const abs = Math.abs(offset);
  return `${withSeconds}${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}
