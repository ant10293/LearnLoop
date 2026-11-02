import { createContext, useContext } from "react";
import type { Bundle } from "./types";

export const BundleContext = createContext<Bundle | null>(null);

export function useOptionalBundle() {
  return useContext(BundleContext);
}

export function useBundle() {
  const bundle = useOptionalBundle();
  if (!bundle) throw new Error("Analysis is not loaded");
  return bundle;
}
