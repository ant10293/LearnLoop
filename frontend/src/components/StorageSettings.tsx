import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type VideoStorage } from "../lib/learnloop";
import { ErrorNote } from "./ui";

const CHOICES: { mode: VideoStorage; label: string; detail: string }[] = [
  { mode: "copy", label: "Copy into LearnLoop", detail: "The file is copied in. The original can be moved or deleted." },
  { mode: "reference", label: "Reference the original path", detail: "Faster. LearnLoop reads the file where it is, so moving or deleting it means it will not load." },
];

/** How the next file is added, from vision-test-bench. */
export function StorageSettings() {
  const client = useQueryClient();
  const settings = useQuery({ queryKey: ["storage"], queryFn: api.storage });
  const save = useMutation({
    mutationFn: api.saveStorage,
    onSuccess: (data) => client.setQueryData(["storage"], data),
  });
  return (
    <fieldset className="storage" disabled={save.isPending || settings.isPending}>
      <legend>Video storage</legend>
      {CHOICES.map((choice) => (
        <label key={choice.mode} className="storage-choice">
          <input type="radio" name="video-storage" checked={settings.data?.video_storage === choice.mode} onChange={() => save.mutate(choice.mode)} />
          <span>
            <strong>{choice.label}</strong>
            <span className="muted small">{choice.detail}</span>
          </span>
        </label>
      ))}
      <ErrorNote message={save.error ? (save.error as Error).message : ""} />
    </fieldset>
  );
}
