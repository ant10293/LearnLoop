/** Largest box with the image's own aspect ratio that fits the expanded preview. */
export function fittedPreviewSize(
  naturalWidth: number,
  naturalHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): { width: number; height: number } | null {
  if (naturalWidth <= 0 || naturalHeight <= 0 || viewportWidth <= 0 || viewportHeight <= 0) return null;
  const scale = Math.min((viewportWidth * 0.9) / naturalWidth, (viewportHeight * 0.85) / naturalHeight);
  return { width: naturalWidth * scale, height: naturalHeight * scale };
}
