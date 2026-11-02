export function ZoomStepper({
  canZoomOut,
  canZoomIn,
  onZoomOut,
  onZoomIn,
}: {
  canZoomOut: boolean;
  canZoomIn: boolean;
  onZoomOut: () => void;
  onZoomIn: () => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-[var(--line)] bg-[var(--panel-2)] text-sm" role="group" aria-label="Zoom">
      <button
        type="button"
        className="px-3.5 py-1 hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Zoom out"
        disabled={!canZoomOut}
        onClick={onZoomOut}
      >
        −
      </button>
      <button
        type="button"
        className="border-l border-[var(--line)] px-3.5 py-1 hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Zoom in"
        disabled={!canZoomIn}
        onClick={onZoomIn}
      >
        +
      </button>
    </div>
  );
}
