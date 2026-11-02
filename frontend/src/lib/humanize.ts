const ACRONYMS = new Set(["ai", "cv", "api", "json", "sam", "iou", "gpt", "fps", "vad", "url", "vlm"]);

const WORDS: Record<string, string> = {
  openai: "OpenAI",
};

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** Turn camelCase, kebab-case, and snake_case identifiers into a short label.
 *  A lone word is capitalized. Sentences, JSON, and ids are returned unchanged.
 */
export function humanize(value: string): string {
  if (!value || /[\s"'`{}[\]()@/\\,;:!?]/.test(value)) return value;
  if (UUID.test(value)) return value;
  if (/^[A-Z][A-Z0-9]+$/.test(value)) return formatWord(value, true);
  if (!/[_-]/.test(value) && !/[a-z\d][A-Z]/.test(value) && !/[A-Z]{2,}[a-z]/.test(value)) return capitalizeWord(value);

  const spaced = value
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .replace(/(\d)-(?=\d)/g, "$1\u0000")
    .replace(/-/g, " ")
    .replace(/\u0000/g, "-");
  const words = spaced.split(/\s+/).filter(Boolean);
  if (!words.length) return value;
  return words
    .map((word, index) => formatWord(word, index === 0))
    .join(" ")
    .replace(/head mounted/gi, (match) => (match.startsWith("H") ? "Head-mounted" : "head-mounted"));
}

/** "Object 1" beside a name becomes "Object 1: bolt". The id stands alone when there is no name. */
export function labeledId(localId: string, name?: string): string {
  const id = humanize(localId);
  const label = name ? humanize(name).trim() : "";
  if (!label) return id;
  return `${id}: ${label}`;
}

const TOKEN_EDGE = /[ "'`{}[\]()@/\\,;:!?]/;

/** Format identifier tokens inside a label, and leave ordinary words in place. */
export function humanizeText(value: string): string {
  return value
    .split(/(\s+)/)
    .map((part) => (/^\s+$/.test(part) ? part : rewriteToken(part)))
    .join("");
}

function rewriteToken(part: string): string {
  let start = 0;
  let end = part.length;
  while (start < end && TOKEN_EDGE.test(part[start])) start += 1;
  while (end > start && TOKEN_EDGE.test(part[end - 1])) end -= 1;
  const core = part.slice(start, end);
  if (!core || !shouldRewriteToken(core)) return part;
  return part.slice(0, start) + humanize(core) + part.slice(end);
}

const OBJECT_REF = /(?<![A-Za-z0-9_])Object_\d+(?:\/Object_\d+)*(?:'s)?(?![A-Za-z0-9_])/g;

/** Verb plus the parts it acts on, so two lifts in one second are not the same row. */
export function actionPhrase(
  action: { action_type: string; object_local_ids?: readonly string[] },
  labels: ReadonlyMap<string, string>,
): string {
  const names = (action.object_local_ids ?? [])
    .map((id) => {
      const label = labels.get(id)?.trim();
      return label ? humanizeText(label) : humanize(id);
    })
    .filter((name, index, all) => name && all.indexOf(name) === index);
  const verb = humanize(action.action_type);
  return names.length ? `${verb} · ${names.join(", ")}` : verb;
}

export function objectLabels(objects: readonly { local_id: string; label: string }[]): Map<string, string> {
  return new Map(objects.map((object) => [object.local_id, object.label]));
}

/** Replace Object_6 in a sentence with that part's name. A bare id stays "Object 6". */
export function nameObjects(value: string, labels: ReadonlyMap<string, string> | Record<string, string>): string {
  if (!value || !value.includes("Object_")) return value;
  const lookup = labels instanceof Map ? labels : new Map(Object.entries(labels));
  return value.replace(OBJECT_REF, (token, offset: number) => {
    const possessive = token.endsWith("'s");
    const stem = possessive ? token.slice(0, -2) : token;
    const before = value.slice(0, offset);
    const hasArticle = /(?:^|\s)(?:the|a|an)\s+$/i.test(before);
    const ids = stem.split("/");
    const pieces = ids.map((id) => {
      const label = lookup.get(id)?.trim();
      return label ? { text: label, named: true } : { text: humanize(id), named: false };
    });
    let phrase: string;
    if (hasArticle) phrase = pieces.map((piece) => piece.text).join(" and ");
    else if (pieces.length === 1) phrase = pieces[0].named ? `the ${pieces[0].text}` : pieces[0].text;
    else phrase = pieces.map((piece) => (piece.named ? `the ${piece.text}` : piece.text)).join(" and ");
    if (possessive) phrase += "'s";
    if (/^\s*$/.test(before) || /[.!?]\s*$/.test(before)) phrase = phrase.charAt(0).toUpperCase() + phrase.slice(1);
    return phrase;
  });
}

function shouldRewriteToken(part: string): boolean {
  if (/_/.test(part) || /[a-z\d][A-Z]/.test(part) || /^[A-Z][A-Z0-9]+$/.test(part)) return true;
  return part.includes("-") && (/\d/.test(part) || part.split("-").length >= 3);
}

function capitalizeWord(value: string): string {
  if (!/^[a-z]/.test(value)) return value;
  const lower = value.toLowerCase();
  if (ACRONYMS.has(lower)) return lower.toUpperCase();
  if (WORDS[lower]) return WORDS[lower];
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatWord(word: string, first: boolean): string {
  if (/^\d/.test(word)) return word;
  const lower = word.toLowerCase();
  if (ACRONYMS.has(lower)) return lower.toUpperCase();
  if (WORDS[lower]) return WORDS[lower];
  if (!first) return lower;
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}
