import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Bone, FieldBone } from "../components/Skeleton";
import { SettingsFieldset } from "../components/SettingsFieldset";
import { api } from "../lib/api";
import { humanize } from "../lib/humanize";
import { PERCEPTION_OPTIONS, PIPELINE_OPTIONS, PROMPT_STRATEGY_OPTIONS, SEGMENTATION_OPTIONS, SEMANTIC_OPTIONS } from "../lib/runOptions";
import type { Capability } from "../lib/types";
import { videoReady } from "../lib/uploadProgress";

const MODES = [
  ["fixed_camera", "Fixed camera"],
  ["constrained_head_mounted", "Constrained head-mounted"],
  ["unconstrained_head_mounted", "Unconstrained head-mounted"],
  ["auto", "Auto-characterize"],
] as const;

export function ExperimentPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const videos = useQuery({ queryKey: ["videos"], queryFn: () => api.videos() });
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const prompts = useQuery({ queryKey: ["prompts"], queryFn: api.prompts });
  const [videoId, setVideoId] = useState(params.get("video") ?? "");
  const [providerId, setProviderId] = useState("meta");
  const [modelId, setModelId] = useState("muse-wonder-1.0");
  const [mode, setMode] = useState("fixed_camera");
  const [preferFrames, setPreferFrames] = useState(false);
  const [preset, setPreset] = useState("standard");
  const [pass, setPass] = useState("single");
  const [semanticMode, setSemanticMode] = useState("combined");
  const [segmentation, setSegmentation] = useState("boxes_only");
  const [promptStrategy, setPromptStrategy] = useState("concept");
  const [multiSeed, setMultiSeed] = useState(false);
  const [temperature, setTemperature] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [transcribe, setTranscribe] = useState(false);
  const [useTranscript, setUseTranscript] = useState(false);
  const [whisperModel, setWhisperModel] = useState("base");
  const [language, setLanguage] = useState("");
  const [vad, setVad] = useState(true);
  const [words, setWords] = useState(false);
  const [windowSec, setWindowSec] = useState("4");
  const [perceptionPreset, setPerceptionPreset] = useState("B_temporal");
  const [error, setError] = useState("");
  const whisper = useQuery({ queryKey: ["audio-models"], queryFn: api.audioModels });
  const models = useMemo(() => {
    const provider = providers.data?.find((item) => item.id === providerId);
    return (provider?.models ?? []).filter((model) => !model.model_id.startsWith("sam"));
  }, [providers.data, providerId]);
  const meta = providers.data?.find((item) => item.id === "meta");
  const samCaps = meta?.sam;
  const selected: Capability | undefined = models.find((item) => item.model_id === modelId) ?? models[0];
  const modality = selected?.native_video && !(preferFrames && selected.image) ? "native_video" : "sampled_frames";
  const codexRun = providerId === "openai" && providers.data?.find((item) => item.id === "openai")?.codex?.transport === "codex";
  const reasoningOptions = selected?.reasoning_efforts ?? [];
  const effort = reasoningOptions.includes(reasoning) ? reasoning : "";
  const videosPending = videos.data === undefined && !videos.isError;
  const providersPending = providers.data === undefined && !providers.isError;
  const promptsPending = prompts.data === undefined && !prompts.isError;

  const create = useMutation({
    mutationFn: () =>
      api.createRun({
        name: `${humanize(mode)} · ${humanize(selected?.model_id ?? modelId)}`,
        video_id: videoId,
        provider_id: providerId,
        model_id: selected?.model_id ?? modelId,
        research_mode: mode,
        modality,
        sampling_preset: preset,
        analysis_pass: pass,
        semantic_mode: semanticMode,
        segmentation_mode: segmentation,
        segmentation_model_id: segmentation === "sam" || segmentation === "sam_video" ? "sam-3.1" : "",
        sam_prompt_strategy: segmentation === "sam_video" ? promptStrategy : "concept",
        sam_multi_seed: segmentation === "sam_video" && promptStrategy !== "concept" && multiSeed,
        temperature: codexRun || temperature === "" ? null : Number(temperature),
        reasoning_effort: effort || null,
        prompt_key: "scene",
        max_frames: 48,
        transcribe,
        use_transcript: transcribe && useTranscript,
        whisper_model: whisperModel,
        whisper_language: language,
        whisper_vad: vad,
        word_timestamps: words,
        align_window_sec: Number(windowSec) || 4,
        perception_preset: perceptionPreset,
      }),
    onSuccess: (run) => navigate(`/runs/${run.id}/pipeline`),
    onError: (err: Error) => setError(err.message),
  });

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-medium">Experiment configuration</h1>
      <p className="text-sm text-[var(--muted)]">
        A model that accepts video receives the video file unless you choose sampled frames. A model that only accepts images receives sampled frames. Unsupported settings stay disabled.
      </p>
      {(videos.isError || providers.isError || prompts.isError) && (
        <p className="text-sm text-[var(--rose)]">
          {(videos.error as Error | null)?.message || (providers.error as Error | null)?.message || (prompts.error as Error | null)?.message}
        </p>
      )}
      <Field label="Video">
        {videosPending ? <FieldBone /> : (
          <select className="field" value={videoId} onChange={(event) => setVideoId(event.target.value)}>
            <option value="">Choose a video</option>
            {videos.data?.filter(videoReady).map((video) => <option key={video.id} value={video.id}>{video.filename}</option>)}
          </select>
        )}
      </Field>
      <Field label="Research mode">
        <select className="field" value={mode} onChange={(event) => setMode(event.target.value)}>
          {MODES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </Field>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Primary provider">
          {providersPending ? <FieldBone stretch /> : (
            <select className="field stretch" value={providerId} onChange={(event) => {
              const next = event.target.value;
              setProviderId(next);
              const first = providers.data?.find((item) => item.id === next)?.models[0];
              if (first) setModelId(first.model_id);
            }}>
              {providers.data?.map((provider) => <option key={provider.id} value={provider.id}>{humanize(provider.id)}</option>)}
            </select>
          )}
        </Field>
        <Field label="Primary model">
          {providersPending ? <FieldBone stretch /> : (
            <select className="field stretch" value={selected?.model_id ?? modelId} onChange={(event) => setModelId(event.target.value)}>
              {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.label || humanize(model.model_id)}</option>)}
            </select>
          )}
        </Field>
      </div>
      {(providersPending || selected) && (
        <div className="panel space-y-1 p-3 text-xs text-[var(--muted)]" aria-hidden={providersPending || undefined}>
          {providersPending ? (
            <>
              <div><Bone width="90%" /></div>
              <div><Bone width="70%" /></div>
              <div><Bone width="80%" /></div>
            </>
          ) : selected && (
            <>
          <div>Image {yes(selected.image)} · Native video {yes(selected.native_video)} · Structured JSON {yes(selected.structured_output)} · Temperature {yes(selected.supports_temperature)}</div>
          <div>Max images per call {selected.max_images}. Reasoning: {selected.reasoning_efforts.map(effortLabel).join(", ") || "not offered"}.</div>
          <div>{selected.notes}</div>
          {codexRun && (
            <div>This run uses the linked ChatGPT plan through Codex in read-only mode. Spend counts against the plan, and the lab records the call as unpriced. Temperature is left unset.</div>
          )}
          {providerId === "anthropic" && providers.data?.find((item) => item.id === "anthropic")?.claude_code?.transport === "claude_code" && (
            <div>This run uses the linked Claude Code subscription. Spend counts against the plan, and the lab records the call as unpriced.</div>
          )}
          {providerId === "cursor" && providers.data?.find((item) => item.id === "cursor")?.cursor_cli?.transport === "login" && (
            <div>This run uses the linked Cursor account through the CLI in ask mode. Spend counts against the plan, and the lab records the call as unpriced.</div>
          )}
          {providerId === "cursor" && providers.data?.find((item) => item.id === "cursor")?.cursor_cli?.transport === "api_key" && (
            <div>This run uses the Cursor API key through the CLI in ask mode. A cost is shown when the catalog has a list rate.</div>
          )}
            </>
          )}
        </div>
      )}
      <Field label="Perception">
        <select className="field" value={perceptionPreset} onChange={(event) => setPerceptionPreset(event.target.value)}>
          {PERCEPTION_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <span className="text-xs text-[var(--muted)]">
          The same footage can be repeated under each preset. Appearance tracking asks for DINOv2 only when that preset is selected, and it does not download weights until a run encodes a crop. No accuracy is assumed.
        </span>
      </Field>
      <Field label="Input modality">
        <select
          className="field"
          value={modality}
          disabled={!selected?.native_video || !selected.image}
          onChange={(event) => setPreferFrames(event.target.value === "sampled_frames")}
        >
          <option value="native_video" disabled={!selected?.native_video}>
            Native video {selected?.native_video ? "(default)" : "(unsupported for this model)"}
          </option>
          <option value="sampled_frames" disabled={!selected?.image}>
            Sampled frames {selected?.image ? "" : "(unsupported for this model)"}
          </option>
        </select>
        <span className="text-xs text-[var(--muted)]">
          {modality === "native_video"
            ? "The model receives the video file. Preview frames stay in the inspector."
            : "The model receives timestamped JPEG frames. Longer clips are split into windows."}
        </span>
      </Field>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Sampling">
          <select className="field stretch" value={preset} onChange={(event) => setPreset(event.target.value)}>
            {["sparse", "standard", "detailed", "high_detail", "intensive", "adaptive"].map((item) => <option key={item} value={item}>{humanize(item)}</option>)}
          </select>
        </Field>
        <Field label="Pipeline">
          <select className="field stretch" value={pass} onChange={(event) => setPass(event.target.value)}>
            {PIPELINE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </Field>
      </div>
      <Field label="Semantic order">
        <select className="field" value={semanticMode} onChange={(event) => setSemanticMode(event.target.value)}>
          {SEMANTIC_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <span className="text-xs text-[var(--muted)]">
          {semanticMode === "grounded"
            ? "Two semantic calls: one names objects, and one interprets actions and workflow after tracking and segmentation are resolved."
            : "One semantic call interprets everything before segmentation. Evidence fusion then resolves geometry without another call."}
        </span>
      </Field>
      <SettingsFieldset legend="Segmentation and tracking">
        <p className="text-xs text-[var(--muted)]">
          The primary model interprets the scene, actions, and workflow. The segmentation model only supplies boxes, masks, and tracks. It does not decide whether a step was completed.
        </p>
        <Field label="Segmentation">
          <select className="field" value={segmentation} onChange={(event) => setSegmentation(event.target.value)}>
            {SEGMENTATION_OPTIONS.map(([value, label]) => (
              <option
                key={value}
                value={value}
                disabled={
                  (value === "sam" && Boolean(samCaps) && !samCaps?.image_segmentation)
                  || (value === "sam_video" && Boolean(samCaps) && !samCaps?.video_tracking)
                }
              >
                {label}
              </option>
            ))}
          </select>
        </Field>
        {(segmentation === "sam" || segmentation === "sam_video") && (
          <p className="text-xs text-[var(--muted)]">
            Secondary model: Meta · sam-3.1. Image segmentation and box prompts are {samCaps?.image_segmentation ? "available" : "not reported"}. Video tracking is {samCaps?.video_tracking ? "available" : "not reported"}.
          </p>
        )}
        {(segmentation === "sam" || segmentation === "sam_video") && meta && !meta.configured && (
          <p className="text-xs text-[var(--amber)]">
            SAM 3.1 needs a Meta Model API key. This run will not switch to GrabCut or another model.
          </p>
        )}
        {segmentation === "sam_video" && (
          <Field label="SAM prompting">
            <select className="field" value={promptStrategy} onChange={(event) => setPromptStrategy(event.target.value)}>
              {PROMPT_STRATEGY_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <span className="text-xs text-[var(--muted)]">
              {promptStrategy === "concept"
                ? "SAM looks for each object by name. This is the default."
                : promptStrategy === "trusted_spatial"
                  ? "SAM follows boxes that a person drew or that two producers agree on. An object with no such box in a chunk stays unobserved, and no name is sent."
                  : "SAM follows verified boxes where they exist. Objects without one are looked up by name in a separate request."}
            </span>
            {promptStrategy !== "concept" && (
              <label className="mt-1 flex items-center gap-2 text-xs">
                <input type="checkbox" checked={multiSeed} onChange={(event) => setMultiSeed(event.target.checked)} />
                Prompt one object on several frames when trusted boxes add a new view or span a gap
              </label>
            )}
          </Field>
        )}
        {segmentation === "sam_video" && (
          <p className="text-xs text-[var(--muted)]">
            Short clips are tracked in one request. Longer recordings are split into chunks. A chunk boundary does not keep the same SAM id, and a frame with no detection is unobserved rather than absent. Unconstrained footage still treats a camera leaving the scene as a loss of view.
          </p>
        )}
      </SettingsFieldset>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Temperature">
          <input className="field stretch" disabled={codexRun || !selected?.supports_temperature} value={temperature} placeholder={codexRun || !selected?.supports_temperature ? "unsupported" : "model default"} onChange={(event) => setTemperature(event.target.value)} />
        </Field>
        <Field label="Reasoning effort">
          <select className="field stretch" value={effort} disabled={!reasoningOptions.length} onChange={(event) => setReasoning(event.target.value)}>
            <option value="">default</option>
            {reasoningOptions.map((option) => <option key={option} value={option}>{effortLabel(option)}</option>)}
          </select>
        </Field>
      </div>
      <Field label="Prompt family">
        {promptsPending ? <FieldBone /> : (
          <select className="field" defaultValue="scene">
            {prompts.data?.filter((item, index, all) => all.findIndex((other) => other.key === item.key) === index).map((item) => (
              <option key={item.key} value={item.key}>{humanize(item.name)}</option>
            ))}
          </select>
        )}
      </Field>
      <SettingsFieldset legend="Audio analysis">
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={transcribe} onChange={(event) => {
            setTranscribe(event.target.checked);
            if (!event.target.checked) setUseTranscript(false);
          }} />
          Enable local transcription
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={useTranscript} disabled={!transcribe} onChange={(event) => setUseTranscript(event.target.checked)} />
          Use transcript for workflow analysis
        </label>
        {transcribe && (
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Whisper model">
              <select className="field stretch" value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)}>
                {(whisper.data?.models ?? [{ model: "base", downloaded: false }]).map((item) => (
                  <option key={item.model} value={item.model}>
                    {humanize(item.model)}{item.downloaded ? "" : " (not downloaded)"}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Language">
              <input className="field stretch" value={language} placeholder="auto" onChange={(event) => setLanguage(event.target.value)} />
            </Field>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={vad} onChange={(event) => setVad(event.target.checked)} />
              Skip silence
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={words} onChange={(event) => setWords(event.target.checked)} />
              Word timestamps
            </label>
            <Field label="Alignment window (seconds)">
              <input className="field" value={windowSec} onChange={(event) => setWindowSec(event.target.value)} />
            </Field>
          </div>
        )}
        {whisper.data && !whisper.data.installed && (
          <p className="text-xs text-[var(--amber)]">{whisper.data.detail}</p>
        )}
        {whisper.data?.acceleration?.note && transcribe && (
          <p className="text-xs text-[var(--muted)]">{whisper.data.acceleration.note}</p>
        )}
        {useTranscript && !["fixture", "deterministic", "manual_json"].includes(providerId) && (
          <p className="text-xs text-[var(--amber)]">
            Transcript text, not the audio track, will be sent to {humanize(providerId)} for comparison with the visual record.
          </p>
        )}
        <p className="text-xs text-[var(--muted)]">
          Off by default. Visual analysis does not wait on speech, and a recording without an audio track still runs.
        </p>
      </SettingsFieldset>
      {error && <p className="text-sm text-[var(--rose)]">{error}</p>}
      <button className="btn primary" disabled={!videoId || create.isPending} onClick={() => create.mutate()}>
        {create.isPending ? "Starting…" : "Start analysis"}
      </button>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field-label">
      <span>{label}</span>
      {children}
    </label>
  );
}

function effortLabel(value: string) {
  if (value === "xhigh") return "Extra high";
  return humanize(value);
}

function yes(value: boolean) {
  return value ? "yes" : "no";
}
