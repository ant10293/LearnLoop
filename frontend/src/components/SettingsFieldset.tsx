import type { ReactNode } from "react";

export function SettingsFieldset({
  legend,
  children,
  disabled = false,
}: {
  legend: string;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <fieldset className="panel space-y-2 p-3 pt-2" disabled={disabled}>
      <legend className="px-1 text-sm font-medium">{legend}</legend>
      {children}
    </fieldset>
  );
}
