import type { CSSProperties } from "react";

/** Placeholder sized by the text around it. Add `bone-fill` to stretch into a media box. From vision-test-bench. */
export function Bone({ className = "", width }: { className?: string; width?: string }) {
  const style: CSSProperties | undefined = width ? { width } : undefined;
  return <span className={className ? `bone ${className}` : "bone"} style={style} aria-hidden="true" />;
}

/** A real, empty control, so its box comes from the same styles and a loading form does not jump when its values arrive. */
export function FieldBone({ select = false, className = "" }: { select?: boolean; className?: string }) {
  const names = className ? `bone field ${className}` : "bone field";
  return select ? <select className={names} disabled tabIndex={-1} aria-hidden="true" /> : <input className={names} disabled tabIndex={-1} aria-hidden="true" />;
}

/** Stands in for a ReviewTag. */
export function TagBone() {
  return <span className="review"><Bone width="4rem" /></span>;
}
