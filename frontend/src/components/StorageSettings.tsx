import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type VideoStorage } from "../lib/api";
import { SettingsFieldset } from "./SettingsFieldset";

export function StorageSettings() {
  const client = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const save = useMutation({
    mutationFn: (videoStorage: VideoStorage) => api.saveSettings({ video_storage: videoStorage }),
    onSuccess: (data) => client.setQueryData(["settings"], data),
  });
  const mode = settings.data?.video_storage;
  const pending = save.isPending || settings.isLoading;

  return (
    <SettingsFieldset legend="Video storage" disabled={pending}>
      <label className="flex items-start gap-2 text-sm">
        <input
          type="radio"
          name="video-storage"
          className="mt-1"
          checked={mode === "copy"}
          onChange={() => save.mutate("copy")}
        />
        <span>
          <span className="block">Copy into the lab</span>
          <span className="block text-xs text-[var(--muted)]">The file is copied into the lab. The original can be moved or deleted.</span>
        </span>
      </label>
      <label className="flex items-start gap-2 text-sm">
        <input
          type="radio"
          name="video-storage"
          className="mt-1"
          checked={mode === "reference"}
          onChange={() => save.mutate("reference")}
        />
        <span>
          <span className="block">Reference the original path</span>
          <span className="block text-xs text-[var(--muted)]">The lab reads the file where it is. Moving or deleting that file means it will not load.</span>
        </span>
      </label>
      {save.isError && <p className="text-sm text-[var(--rose)]">{(save.error as Error).message}</p>}
      <p className="text-xs text-[var(--muted)]">Choose how a new video is added. This applies to the next file you add.</p>
    </SettingsFieldset>
  );
}
