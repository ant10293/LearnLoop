type PathFile = File & { path?: unknown };

function isAbsolute(path: string): boolean {
  return path.startsWith("/") || /^[A-Za-z]:[\\/]/.test(path);
}

function decodePath(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function fileUrlToPath(value: string): string {
  const text = value.trim();
  if (!text || text.startsWith("#")) return "";
  if (/^file:/i.test(text)) {
    try {
      let path = decodePath(new URL(text).pathname);
      if (/^\/[A-Za-z]:\//.test(path)) path = path.slice(1);
      return isAbsolute(path) ? path : "";
    } catch {
      return "";
    }
  }
  return isAbsolute(text) ? text : "";
}

function readList(transfer: { getData(type: string): string }, type: string): string {
  try {
    return transfer.getData(type);
  } catch {
    return "";
  }
}

function listedPaths(transfer: { getData(type: string): string }): string[] {
  const raw = readList(transfer, "text/uri-list") || readList(transfer, "text/plain");
  return raw.split(/\r?\n/).map(fileUrlToPath).filter(Boolean);
}

/** Absolute path when the drop already includes one. Browsers often omit it. */
export function droppedPath(file: File): string {
  const path = (file as PathFile).path;
  if (typeof path !== "string") return "";
  const trimmed = path.trim();
  return isAbsolute(trimmed) ? trimmed : "";
}

/** One absolute path per file, or "" when the drop did not include that file's path. */
export function pathsForDrop(files: File[], transfer: { getData(type: string): string }): string[] {
  const listed = listedPaths(transfer);
  return files.map((file, index) => {
    const direct = droppedPath(file);
    if (direct) return direct;
    if (listed.length === files.length) return listed[index] ?? "";
    const name = file.name;
    return listed.find((path) => path.endsWith(`/${name}`) || path.endsWith(`\\${name}`)) ?? "";
  });
}
