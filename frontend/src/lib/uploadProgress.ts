/** Each step fills the bar on its own, from 0 to 100. The next step starts over. */

export function progressStepCount(spansDays: boolean): number {
  return spansDays ? 3 : 2;
}

export function progressLabel(step: number, total: number, detail: string): string {
  return `Step ${step}/${total} · ${detail}`;
}

export const RECONCILE_DETAIL = "Reconciling the project reading";

export function uploadPercent(ratio: number): number {
  const clamped = Math.max(0, Math.min(1, ratio));
  return Math.round(clamped * 100);
}

export function analysisPercent(serverProgress: number): number {
  const clamped = Math.max(0, Math.min(100, serverProgress));
  return Math.round(clamped);
}

export function videoReady(video: { ingest_status?: string }): boolean {
  return !video.ingest_status || video.ingest_status === "ready";
}
