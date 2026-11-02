import { useEffect, useRef, useState, type ReactNode } from "react";
import { pathsForDrop } from "../lib/droppedPath";

type Item = { name: string; url: string; file: File };

export type DropRow = {
  id: string;
  name: string;
  url: string;
  subtitle: string;
  progress: number | null;
  failed?: boolean;
};

type Props = {
  accept: string;
  multiple?: boolean;
  prompt: string;
  chooseLabel: string;
  supported: string;
  hint?: string;
  pending?: boolean;
  pendingLabel?: string;
  error?: string;
  onFiles: (files: File[], paths?: string[]) => void;
  defer?: boolean;
  onStagedChange?: (files: File[]) => void;
  settings?: ReactNode;
  rows?: DropRow[];
  onRemove?: (id: string) => void;
  chooser?: boolean;
  libraryLabel?: string;
  onChooseLibrary?: () => void;
};

export function AssetDropZone({
  accept,
  multiple = false,
  prompt,
  chooseLabel,
  supported,
  hint,
  pending = false,
  pendingLabel,
  error,
  onFiles,
  defer = false,
  onStagedChange,
  settings,
  rows,
  onRemove,
  chooser = true,
  libraryLabel,
  onChooseLibrary,
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const wasPending = useRef(false);
  const itemsRef = useRef<Item[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [items, setItems] = useState<Item[]>([]);
  const controlled = rows !== undefined;
  itemsRef.current = items;

  useEffect(() => {
    if (controlled || !wasPending.current || pending || error) {
      wasPending.current = pending;
      return;
    }
    setItems((current) => {
      for (const item of current) URL.revokeObjectURL(item.url);
      return [];
    });
    wasPending.current = pending;
  }, [controlled, pending, error]);

  useEffect(() => {
    return () => {
      if (controlled) return;
      for (const item of itemsRef.current) URL.revokeObjectURL(item.url);
    };
  }, [controlled]);

  function publish(next: Item[]) {
    onStagedChange?.(next.map((item) => item.file));
  }

  function isFileDrag(event: { dataTransfer: DataTransfer | null }) {
    return Array.from(event.dataTransfer?.types ?? []).includes("Files");
  }

  function take(list: FileList | null, hintedPaths?: string[]) {
    const files = [...(list ?? [])];
    const chosen = multiple ? files : files.slice(0, 1);
    if (!chosen.length) return;
    const paths = hintedPaths?.slice(0, chosen.length);
    if (controlled) {
      onFiles(chosen, paths);
      return;
    }
    const added = chosen.map((file) => ({ name: file.name, url: URL.createObjectURL(file), file }));
    setItems((current) => {
      const next = defer && multiple ? [...current, ...added] : added;
      if (!(defer && multiple)) {
        for (const item of current) URL.revokeObjectURL(item.url);
      }
      publish(next);
      return next;
    });
    if (!defer) onFiles(chosen, paths);
  }

  function remove(url: string) {
    setItems((current) => {
      const next = current.filter((item) => item.url !== url);
      const gone = current.find((item) => item.url === url);
      if (gone) URL.revokeObjectURL(gone.url);
      publish(next);
      return next;
    });
  }

  const shown: DropRow[] = controlled
    ? rows
    : items.map((item) => ({
        id: item.url,
        name: item.name,
        url: item.url,
        subtitle: pending ? pendingLabel || "Working…" : error || "Ready",
        progress: null,
        failed: Boolean(error),
      }));
  const showZone = chooser || Boolean(onChooseLibrary);

  return (
    <div>
      {showZone && <>
      <div
        className={`drop-zone${dragOver ? " is-dragover" : ""}${chooser ? "" : " is-compact"}`}
        tabIndex={0}
        onDragEnter={(event) => {
          if (!chooser || !isFileDrag(event)) return;
          event.preventDefault();
          setDragOver(true);
        }}
        onDragOver={(event) => {
          if (!chooser || !isFileDrag(event)) return;
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragOver(false);
        }}
        onDrop={(event) => {
          const files = event.dataTransfer.files;
          const hinted = pathsForDrop([...files], event.dataTransfer);
          event.preventDefault();
          setDragOver(false);
          if (!chooser || !isFileDrag(event)) return;
          take(files, hinted);
        }}
        onKeyDown={(event) => {
          if (!chooser) return;
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            input.current?.click();
          }
        }}
      >
        {chooser && <div className="drop-zone-icon" aria-hidden="true" />}
        <p className="drop-zone-copy">
          {[
            prompt ? `${prompt} ` : null,
            chooser ? (
              <button key="files" type="button" className="linkish" onClick={() => input.current?.click()}>
                {chooseLabel}
              </button>
            ) : null,
            chooser && onChooseLibrary ? ", or " : null,
            onChooseLibrary ? (
              <button key="library" type="button" className="linkish" onClick={onChooseLibrary}>
                {libraryLabel}
              </button>
            ) : null,
          ]}
        </p>
        {chooser && <p className="drop-zone-supported">{supported}</p>}
        {chooser && (
          <input
            ref={input}
            type="file"
            hidden
            multiple={multiple}
            accept={accept}
            onChange={(event) => {
              take(event.target.files);
              event.currentTarget.value = "";
            }}
          />
        )}
      </div>
      {hint ? (
        <div className="drop-zone-foot">
          <span>{hint}</span>
        </div>
      ) : null}
      </>}
      {shown.map((item) => {
        const percent = item.progress == null ? null : Math.round(item.progress);
        const removable = Boolean(onRemove) || (!controlled && defer);
        return (
          <div key={item.id} className="drop-item">
            <div className="drop-item-main">
              {item.url?.startsWith("blob:") ? (
                <video className="drop-item-thumb" src={item.url} muted playsInline preload="metadata" />
              ) : item.url ? (
                <img className="drop-item-thumb" src={item.url} alt="" />
              ) : (
                <span className="drop-item-thumb" aria-hidden="true" />
              )}
              <div className="drop-item-meta">
                <strong>{item.name}</strong>
                {item.subtitle ? <span className={item.failed ? "is-error" : undefined}>{item.subtitle}</span> : null}
              </div>
              {removable && (
                <button
                  type="button"
                  className="drop-item-remove"
                  aria-label={`Remove ${item.name}`}
                  title="Remove"
                  onClick={() => (onRemove ? onRemove(item.id) : remove(item.id))}
                >
                  ×
                </button>
              )}
            </div>
            {percent != null && (
              <div
                className="drop-progress"
                role="progressbar"
                aria-valuenow={percent}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={item.subtitle}
              >
                <div className="drop-progress-track">
                  <div className="drop-progress-fill" style={{ width: `${percent}%` }} />
                </div>
                <span className="mono">{percent}%</span>
              </div>
            )}
          </div>
        );
      })}
      {!shown.length && pending && pendingLabel && <p className="drop-zone-status">{pendingLabel}</p>}
      {!shown.length && error && <p className="drop-zone-status is-error">{error}</p>}
      {settings && <div className="drop-zone-settings">{settings}</div>}
    </div>
  );
}
