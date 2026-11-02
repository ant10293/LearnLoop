import type { CSSProperties, SyntheticEvent } from "react";
import { formatConfidence } from "../lib/confidence";

/** An object still with its model score in the top-right corner. */
export function ShotFrame({
  src,
  alt = "",
  confidence,
  className,
  imgClassName,
  imgStyle,
  frameStyle,
  draggable,
  onLoad,
}: {
  src: string;
  alt?: string;
  confidence?: number | null;
  className?: string;
  imgClassName?: string;
  imgStyle?: CSSProperties;
  frameStyle?: CSSProperties;
  draggable?: boolean;
  onLoad?: (event: SyntheticEvent<HTMLImageElement>) => void;
}) {
  return (
    <span className={className ? `shot-frame ${className}` : "shot-frame"} style={frameStyle}>
      <img src={src} alt={alt} className={imgClassName} style={imgStyle} draggable={draggable} onLoad={onLoad} />
      {confidence != null && <span className="shot-score">{formatConfidence(confidence)}</span>}
    </span>
  );
}
