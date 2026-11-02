import type { CSSProperties } from "react";

/** Placeholder sized by the text around it. Add `bone-fill` to stretch into a media box. */
export function Bone({ className = "", width }: { className?: string; width?: string }) {
  const style: CSSProperties | undefined = width ? { width } : undefined;
  return <span className={className ? `bone ${className}` : "bone"} style={style} aria-hidden="true" />;
}

/** The media block a project card shows while its preview is not ready. */
export function MediaPlaceholder() {
  return (
    <div className="thumb-grid is-single">
      <Bone className="bone-fill" />
    </div>
  );
}

/**
 * A field-shaped bone. Matches a select by default; `input` matches a text input,
 * `textarea` a default two-row text area, `tall` a prompt editor, and `compact` an inline select.
 */
export function FieldBone({
  stretch = false,
  tall = false,
  textarea = false,
  compact = false,
  input = false,
  className = "",
}: {
  stretch?: boolean;
  tall?: boolean;
  textarea?: boolean;
  compact?: boolean;
  input?: boolean;
  className?: string;
}) {
  const names = ["bone", "field", "is-placeholder"];
  if (stretch) names.push("stretch");
  if (tall) names.push("is-tall");
  if (textarea) names.push("is-textarea");
  if (compact) names.push("is-compact");
  if (input) names.push("is-input");
  if (className) names.push(className);
  return <div className={names.join(" ")} aria-hidden="true" />;
}
