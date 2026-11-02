import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Bone, FieldBone } from "../components/Skeleton";
import { api } from "../lib/api";
import { humanize, humanizeText } from "../lib/humanize";
import type { ClaudeCodeSettings, CliLinkStatus, CodexSettings, CursorCliSettings, ProviderInfo, RunSummary } from "../lib/types";

function LinkTag({ pending, linked }: { pending: boolean; linked: boolean }) {
  if (pending) return <span className="tag" aria-hidden="true"><Bone width="5.5rem" /></span>;
  return <span className={`tag ${linked ? "observed" : "unknown"}`}>{linked ? "linked" : "not linked"}</span>;
}

function AccountIdentity({ status }: { status?: Pick<CliLinkStatus, "logged_in" | "subscription" | "account"> }) {
  if (!status?.logged_in) return null;
  const line = [status.subscription ? humanize(status.subscription) : "", status.account].filter(Boolean).join(" · ");
  if (!line) return null;
  return <p className="text-sm">{line}</p>;
}

function useStoredCli<T extends string>(
  settings: { binary: string; transport: T },
  save: (binary: string, transport: T) => Promise<unknown>,
  statusKey: string,
) {
  const client = useQueryClient();
  const [binary, setBinary] = useState(settings.binary);
  const [transport, setTransport] = useState(settings.transport);
  const [note, setNote] = useState("");
  const tail = useRef(Promise.resolve());
  useEffect(() => {
    setBinary(settings.binary);
    setTransport(settings.transport);
  }, [settings.binary, settings.transport]);

  function persist(nextBinary: string, nextTransport: T): Promise<boolean> {
    const run = tail.current.then(() => write(nextBinary, nextTransport));
    tail.current = run.then(() => undefined, () => undefined);
    return run;
  }

  async function write(nextBinary: string, nextTransport: T): Promise<boolean> {
    const trimmed = nextBinary.trim();
    if (!trimmed) {
      setNote("Executable path is required.");
      return false;
    }
    if (trimmed === settings.binary && nextTransport === settings.transport) return true;
    try {
      await save(trimmed, nextTransport);
      setNote("");
      await client.invalidateQueries({ queryKey: ["providers"] });
      await client.invalidateQueries({ queryKey: [statusKey] });
      return true;
    } catch (error) {
      setNote(error instanceof Error ? error.message : "Could not update this provider.");
      return false;
    }
  }

  return { binary, setBinary, transport, setTransport, note, setNote, persist };
}

function CodexPanel({ settings }: { settings: CodexSettings }) {
  const client = useQueryClient();
  const { binary, setBinary, transport, setTransport, note, setNote, persist } = useStoredCli(settings, api.saveCodex, "codex");
  const [awaiting, setAwaiting] = useState(false);
  const [loginUrl, setLoginUrl] = useState("");
  const status = useQuery({
    queryKey: ["codex"],
    queryFn: api.codex,
    refetchInterval: awaiting ? 3000 : false,
  });
  useEffect(() => {
    if (awaiting && status.data?.logged_in) {
      setAwaiting(false);
      setLoginUrl("");
      setNote("ChatGPT is linked.");
    }
  }, [awaiting, status.data?.logged_in, setNote]);
  const linked = status.data?.logged_in ?? false;
  return (
    <div className="space-y-2 rounded-lg border border-[var(--line)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-medium">Codex</h3>
        <LinkTag pending={status.data === undefined && !status.isError} linked={linked} />
        <span className="tag">{transport === "codex" ? "runs use the ChatGPT plan" : "runs use API key"}</span>
        {status.data?.version && <span className="tag">{status.data.version}</span>}
      </div>
      <p className="text-xs text-[var(--muted)]">
        Analysis runs call the codex executable in read-only mode. Codex can see the frames and is not given permission to edit files. A linked ChatGPT plan, including Plus, is used when runs are set to Codex. The API key above is used only when runs are set to API key.
      </p>
      <AccountIdentity status={status.data} />
      {status.data?.error && !awaiting && <p className="text-sm text-[var(--rose)]">{status.data.error}</p>}
      <div className="grid gap-2 md:grid-cols-2">
        <label className="field-label">
          Executable
          <input className="field" value={binary} spellCheck={false} onChange={(event) => setBinary(event.target.value)} onBlur={() => { void persist(binary, transport); }} />
        </label>
        <label className="field-label">
          OpenAI runs
          <select className="field stretch" value={transport} onChange={async (event) => {
            const next = event.target.value as CodexSettings["transport"];
            setTransport(next);
            if (!(await persist(binary, next))) setTransport(settings.transport);
          }}>
            <option value="api">API key</option>
            <option value="codex">ChatGPT plan</option>
          </select>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {transport === "codex" && (
          <button className="btn" onClick={async () => {
            if (!(await persist(binary, transport))) return;
            try {
              const result = await api.linkCodex(false);
              setLoginUrl(result.login_url || "");
              setAwaiting(Boolean(result.login_started) && !result.logged_in);
              setNote(result.logged_in ? "ChatGPT is linked." : "Open the sign-in page and finish with the ChatGPT account. This page checks until it is linked.");
              await client.invalidateQueries({ queryKey: ["codex"] });
            } catch (error) {
              setAwaiting(false);
              setNote(error instanceof Error ? error.message : "link failed");
            }
          }}>{linked ? "Refresh link" : "Link ChatGPT"}</button>
        )}
        {transport === "codex" && linked && (
          <button className="btn" onClick={async () => {
            if (!(await persist(binary, transport))) return;
            try {
              const result = await api.linkCodex(true);
              setLoginUrl(result.login_url || "");
              setAwaiting(Boolean(result.login_started));
              setNote("Finish the new sign-in in the browser to replace the linked account.");
            } catch (error) {
              setAwaiting(false);
              setNote(error instanceof Error ? error.message : "sign-in failed");
            }
          }}>Sign in again</button>
        )}
        {loginUrl && (
          <a className="btn primary inline-flex items-center" href={loginUrl} target="_blank" rel="noreferrer">Open the ChatGPT sign-in page</a>
        )}
      </div>
      {note && <p className="text-sm">{note}</p>}
    </div>
  );
}

function ClaudeCodePanel({ settings }: { settings: ClaudeCodeSettings }) {
  const client = useQueryClient();
  const { binary, setBinary, transport, setTransport, note, setNote, persist } = useStoredCli(settings, api.saveClaudeCode, "claude-code");
  const [awaiting, setAwaiting] = useState(false);
  const [loginUrl, setLoginUrl] = useState("");
  const [loginCode, setLoginCode] = useState("");
  const status = useQuery({
    queryKey: ["claude-code"],
    queryFn: api.claudeCode,
    refetchInterval: awaiting ? 3000 : false,
  });
  useEffect(() => {
    if (awaiting && status.data?.logged_in) {
      setAwaiting(false);
      setLoginUrl("");
      setLoginCode("");
      setNote("Claude Code is linked.");
    }
  }, [awaiting, status.data?.logged_in, setNote]);
  const linked = status.data?.logged_in ?? false;
  return (
    <div className="space-y-2 rounded-lg border border-[var(--line)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-medium">Claude Code</h3>
        <LinkTag pending={status.data === undefined && !status.isError} linked={linked} />
        <span className="tag">{transport === "claude_code" ? "runs use Claude Code" : "runs use API key"}</span>
        {status.data?.version && <span className="tag">{status.data.version}</span>}
      </div>
      <p className="text-xs text-[var(--muted)]">
        Uses the Claude plan signed in through the Claude Code executable on this machine. The API key above still calls the Anthropic API when runs are set to API key.
      </p>
      <AccountIdentity status={status.data} />
      {status.data?.error && !awaiting && <p className="text-sm text-[var(--rose)]">{status.data.error}</p>}
      <div className="grid gap-2 md:grid-cols-2">
        <label className="field-label">
          Executable
          <input className="field" value={binary} spellCheck={false} onChange={(event) => setBinary(event.target.value)} onBlur={() => { void persist(binary, transport); }} />
        </label>
        <label className="field-label">
          Anthropic runs
          <select className="field stretch" value={transport} onChange={async (event) => {
            const next = event.target.value as ClaudeCodeSettings["transport"];
            setTransport(next);
            if (!(await persist(binary, next))) setTransport(settings.transport);
          }}>
            <option value="api">API key</option>
            <option value="claude_code">Claude Code subscription</option>
          </select>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn" onClick={async () => {
          if (!(await persist(binary, transport))) return;
          try {
            const result = await api.linkClaudeCode(false);
            setLoginUrl(result.login_url || "");
            setAwaiting(Boolean(result.login_started) && !result.logged_in);
            setNote(result.logged_in ? "Claude Code is linked." : "Open the sign-in page and choose the Pro or Max account. Paste the code it shows here.");
            await client.invalidateQueries({ queryKey: ["claude-code"] });
          } catch (error) {
            setAwaiting(false);
            setNote(error instanceof Error ? error.message : "link failed");
          }
        }}>{linked ? "Refresh link" : "Link Claude Code"}</button>
        {linked && (
          <button className="btn" onClick={async () => {
            if (!(await persist(binary, transport))) return;
            try {
              const result = await api.linkClaudeCode(true);
              setLoginUrl(result.login_url || "");
              setAwaiting(false);
              setNote("Finish the new sign-in in the browser to replace the linked account.");
            } catch (error) {
              setAwaiting(false);
              setNote(error instanceof Error ? error.message : "sign-in failed");
            }
          }}>Sign in again</button>
        )}
        {loginUrl && (
          <a className="btn primary inline-flex items-center" href={loginUrl} target="_blank" rel="noreferrer">Open the Claude sign-in page</a>
        )}
      </div>
      {awaiting && (
        <div className="flex flex-wrap items-center gap-2">
          <input className="field" spellCheck={false} placeholder="Paste the sign-in code" value={loginCode} onChange={(event) => setLoginCode(event.target.value)} />
          <button className="btn" disabled={!loginCode.trim()} onClick={async () => {
            try {
              const result = await api.submitClaudeCode(loginCode.trim());
              setLoginCode("");
              if (result.logged_in) {
                setAwaiting(false);
                setLoginUrl("");
                setNote("Claude Code is linked.");
              } else {
                setNote(result.error || "That code was not accepted. Paste the code from the Pro or Max account.");
              }
              await client.invalidateQueries({ queryKey: ["claude-code"] });
            } catch (error) {
              setNote(error instanceof Error ? error.message : "sign-in failed");
            }
          }}>Submit code</button>
        </div>
      )}
      {note && <p className="text-sm">{note}</p>}
    </div>
  );
}

function CursorCliPanel({ settings }: { settings: CursorCliSettings }) {
  const client = useQueryClient();
  const { binary, setBinary, transport, setTransport, note, setNote, persist } = useStoredCli(settings, api.saveCursorCli, "cursor-cli");
  const [awaiting, setAwaiting] = useState(false);
  const [loginUrl, setLoginUrl] = useState("");
  const status = useQuery({
    queryKey: ["cursor-cli"],
    queryFn: api.cursorCli,
    refetchInterval: awaiting ? 3000 : false,
  });
  useEffect(() => {
    if (awaiting && status.data?.logged_in) {
      setAwaiting(false);
      setLoginUrl("");
      setNote("Cursor is linked.");
    }
  }, [awaiting, status.data?.logged_in, setNote]);
  const linked = status.data?.logged_in ?? false;
  return (
    <div className="space-y-2 rounded-lg border border-[var(--line)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-medium">Cursor CLI</h3>
        <LinkTag pending={status.data === undefined && !status.isError} linked={linked} />
        <span className="tag">{transport === "login" ? "runs use the Cursor account" : "runs use the API key"}</span>
        {status.data?.version && <span className="tag">{status.data.version}</span>}
      </div>
      <p className="text-xs text-[var(--muted)]">
        Analysis runs call the agent executable in ask mode. The agent can read the frames and is not given permission to edit them. If the answer is not one JSON object, a follow-up call edits that file until it parses. A linked account uses the Cursor plan. The API key above is used only when runs are set to API key.
      </p>
      <AccountIdentity status={status.data} />
      {status.data?.error && !awaiting && <p className="text-sm text-[var(--rose)]">{status.data.error}</p>}
      <div className="grid gap-2 md:grid-cols-2">
        <label className="field-label">
          Executable
          <input className="field" value={binary} spellCheck={false} onChange={(event) => setBinary(event.target.value)} onBlur={() => { void persist(binary, transport); }} />
        </label>
        <label className="field-label">
          Cursor runs
          <select className="field stretch" value={transport} onChange={async (event) => {
            const next = event.target.value as CursorCliSettings["transport"];
            setTransport(next);
            if (!(await persist(binary, next))) setTransport(settings.transport);
          }}>
            <option value="login">Cursor account</option>
            <option value="api_key">API key</option>
          </select>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {transport === "login" && (
          <button className="btn" onClick={async () => {
            if (!(await persist(binary, transport))) return;
            try {
              const result = await api.linkCursorCli(false);
              setLoginUrl(result.login_url || "");
              setAwaiting(Boolean(result.login_started) && !result.logged_in);
              setNote(result.logged_in ? "Cursor is linked." : "Open the sign-in page and finish with the Cursor account. This page checks until it is linked.");
              await client.invalidateQueries({ queryKey: ["cursor-cli"] });
            } catch (error) {
              setAwaiting(false);
              setNote(error instanceof Error ? error.message : "link failed");
            }
          }}>{linked ? "Refresh link" : "Link Cursor"}</button>
        )}
        {transport === "login" && linked && (
          <button className="btn" onClick={async () => {
            if (!(await persist(binary, transport))) return;
            try {
              const result = await api.linkCursorCli(true);
              setLoginUrl(result.login_url || "");
              setAwaiting(Boolean(result.login_started));
              setNote("Finish the new sign-in in the browser to replace the linked account.");
            } catch (error) {
              setAwaiting(false);
              setNote(error instanceof Error ? error.message : "sign-in failed");
            }
          }}>Sign in again</button>
        )}
        {loginUrl && (
          <a className="btn primary inline-flex items-center" href={loginUrl} target="_blank" rel="noreferrer">Open the Cursor sign-in page</a>
        )}
      </div>
      {note && <p className="text-sm">{note}</p>}
    </div>
  );
}

const MODEL_PLACEHOLDERS = 2;

function ProviderCard({
  provider,
  apiKey,
  base,
  modelDraft,
  onKey,
  onBase,
  onModel,
  onSaveKey,
  onValidate,
  onReady,
  onAddModel,
  onDiscover,
  pendingId,
  pendingLocal,
}: {
  provider: ProviderInfo | null;
  apiKey: string;
  base: string;
  modelDraft: string;
  onKey: (value: string) => void;
  onBase: (value: string) => void;
  onModel: (value: string) => void;
  onSaveKey: () => void;
  onValidate: () => void;
  onReady: () => void;
  onAddModel: () => void;
  onDiscover: () => void;
  pendingId?: string;
  pendingLocal?: boolean;
}) {
  const pending = provider == null;
  const id = provider?.id ?? pendingId ?? "";
  const local = provider?.local ?? pendingLocal ?? false;
  return (
    <section className="panel space-y-2 p-4" aria-hidden={pending || undefined}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg">{id ? humanize(id) : <Bone width="8rem" />}</h2>
        {provider ? (
          <span className={`tag ${provider.validated ? "observed" : "unknown"}`}>{provider.validated ? "validated" : "not validated"}</span>
        ) : (
          <span className="tag"><Bone width="6.5rem" /></span>
        )}
        {provider?.configured && <span className="tag">key {humanize(provider.secret_source)} {provider.hint}</span>}
      </div>
      {provider?.last_error && <p className="text-sm text-[var(--rose)]">{provider.last_error}</p>}
      {pending && <p className="text-sm"><Bone width="70%" /></p>}
      {provider?.id === "openai" && (
        <p className="text-sm text-[var(--muted)]">API key. Analysis runs use it when OpenAI runs is set to API key.</p>
      )}
      {provider?.id === "anthropic" && (
        <p className="text-sm text-[var(--muted)]">API key. Analysis runs use it when Anthropic runs is set to API key.</p>
      )}
      {provider?.id === "cursor" && (
        <p className="text-sm text-[var(--muted)]">API key. Analysis runs use it when Cursor runs is set to API key.</p>
      )}
      {!local && (pending || provider) && (
        <div className="flex flex-wrap items-center gap-2">
          {provider ? (
            <>
              <input className="field" type="password" autoComplete="off" placeholder="API key" value={apiKey} onChange={(event) => onKey(event.target.value)} />
              {provider.id !== "cursor" && (
                <input className="field" placeholder={provider.id === "openai_compatible" ? "Base URL" : "Optional base URL"} value={base} onChange={(event) => onBase(event.target.value)} />
              )}
              <button className="btn" onClick={onSaveKey}>Save key</button>
              <button className="btn" onClick={onValidate}>Validate</button>
            </>
          ) : (
            <>
              <FieldBone input />
              {id !== "cursor" && <FieldBone input />}
              <span className="btn">Save key</span>
              <span className="btn">Validate</span>
            </>
          )}
        </div>
      )}
      {provider?.id === "openai" && provider.codex && <CodexPanel settings={provider.codex} />}
      {provider?.id === "anthropic" && provider.claude_code && <ClaudeCodePanel settings={provider.claude_code} />}
      {provider?.id === "cursor" && provider.cursor_cli && <CursorCliPanel settings={provider.cursor_cli} />}
      {provider && local && (
        <button className="btn" onClick={onReady}>Mark local provider ready</button>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {provider ? (
          <input className="field" placeholder="Add model id" value={modelDraft} onChange={(event) => onModel(event.target.value)} />
        ) : (
          <FieldBone input />
        )}
        {provider ? (
          <>
            <button className="btn shrink-0" onClick={onAddModel}>Add model</button>
            <button className="btn shrink-0" onClick={onDiscover}>Discover</button>
          </>
        ) : (
          <>
            <span className="btn shrink-0">Add model</span>
            <span className="btn shrink-0">Discover</span>
          </>
        )}
      </div>
      <ul className="grid gap-1 text-xs text-[var(--muted)] md:grid-cols-2">
        {provider ? provider.models.map((model) => (
          <li key={model.model_id}>{model.label || humanize(model.model_id)} · video {model.native_video ? "yes" : "no"} · images {model.max_images} · {model.notes}</li>
        )) : Array.from({ length: MODEL_PLACEHOLDERS }, (_, index) => (
          <li key={index}><Bone width="80%" /></li>
        ))}
      </ul>
    </section>
  );
}

export function ProvidersPage() {
  const client = useQueryClient();
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const providerIds = useQuery({ queryKey: ["provider-ids"], queryFn: api.providerIds, staleTime: Infinity });
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [bases, setBases] = useState<Record<string, string>>({});
  const [models, setModels] = useState<Record<string, string>>({});
  const [message, setMessage] = useState("");
  const save = useMutation({
    mutationFn: (id: string) => api.saveKey(id, keys[id] ?? "", bases[id] ?? ""),
    onSuccess: async () => {
      setKeys({});
      setMessage("Key stored on the server. It is not returned to the browser.");
      await client.invalidateQueries({ queryKey: ["providers"] });
    },
    onError: (error: Error) => setMessage(error.message),
  });
  const pending = providers.data === undefined && !providers.isError;
  const catalog = pending ? providerIds.data ?? [] : [];
  const rows: (ProviderInfo | null)[] = pending ? catalog.map(() => null) : providers.data ?? [];
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-medium">Providers</h1>
      <p className="max-w-3xl text-sm text-[var(--muted)]">
        Keys stay on the backend. A provider is connected only after Validate succeeds against the real API.
        Deterministic and fixture providers need no key. Live inference has not been claimed from this screen until that call returns ok.
      </p>
      {message && <p className="text-sm">{message}</p>}
      {providers.isError && <p className="text-sm text-[var(--rose)]">{(providers.error as Error).message}</p>}
      <div className="space-y-3" aria-busy={pending || undefined}>
        {pending && <p className="sr-only">Loading providers</p>}
        {rows.map((provider, row) => (
          <ProviderCard
            key={provider?.id ?? catalog[row]?.id ?? `pending-${row}`}
            provider={provider}
            pendingId={catalog[row]?.id}
            pendingLocal={catalog[row]?.local}
            apiKey={provider ? keys[provider.id] ?? "" : ""}
            base={provider ? bases[provider.id] ?? provider.base_url : ""}
            modelDraft={provider ? models[provider.id] ?? "" : ""}
            onKey={(value) => { if (provider) setKeys({ ...keys, [provider.id]: value }); }}
            onBase={(value) => { if (provider) setBases({ ...bases, [provider.id]: value }); }}
            onModel={(value) => { if (provider) setModels({ ...models, [provider.id]: value }); }}
            onSaveKey={() => { if (provider) save.mutate(provider.id); }}
            onValidate={() => {
              if (!provider) return;
              void (async () => {
                try {
                  await api.validateProvider(provider.id);
                  setMessage(`${provider.id} validation request succeeded.`);
                } catch (error) {
                  setMessage(error instanceof Error ? error.message : "validation failed");
                }
                await client.invalidateQueries({ queryKey: ["providers"] });
              })();
            }}
            onReady={() => {
              if (!provider) return;
              void (async () => {
                await api.validateProvider(provider.id);
                await client.invalidateQueries({ queryKey: ["providers"] });
              })();
            }}
            onAddModel={() => {
              if (!provider) return;
              void (async () => {
                await api.addModel(provider.id, models[provider.id] ?? "");
                await client.invalidateQueries({ queryKey: ["providers"] });
              })();
            }}
            onDiscover={() => {
              if (!provider) return;
              void (async () => {
                const found = await api.discoverModels(provider.id);
                await client.invalidateQueries({ queryKey: ["providers"] });
                setMessage(found.error ? found.error : `Discovered ${found.discovered.length} model ids.`);
              })();
            }}
          />
        ))}
      </div>
    </div>
  );
}

/** Ignore surrounding and trailing whitespace so spacing-only edits are not a new version. */
function comparablePrompt(value: string): string {
  return value
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .trim();
}

type PromptRecord = {
  id: string;
  key: string;
  name: string;
  version: number;
  body: string;
};

/** Seeded pipeline order. Any later key is appended after these. */
const PROMPT_ORDER = [
  "foundational",
  "scene",
  "actions",
  "states",
  "workflow",
  "ambiguity",
  "reinspect",
  "comparison",
];

function latestPrompts(rows: PromptRecord[]): PromptRecord[] {
  const latest = new Map<string, PromptRecord>();
  for (const row of rows) {
    const existing = latest.get(row.key);
    if (!existing || row.version > existing.version) latest.set(row.key, row);
  }
  return [...latest.values()].sort((left, right) => {
    const leftRank = PROMPT_ORDER.indexOf(left.key);
    const rightRank = PROMPT_ORDER.indexOf(right.key);
    const leftOrder = leftRank === -1 ? PROMPT_ORDER.length : leftRank;
    const rightOrder = rightRank === -1 ? PROMPT_ORDER.length : rightRank;
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    return left.key.localeCompare(right.key);
  });
}

function PromptSection({ prompt, pendingKey = "" }: { prompt: PromptRecord | null; pendingKey?: string }) {
  const client = useQueryClient();
  const [draft, setDraft] = useState(prompt?.body ?? "");
  const [savedBody, setSavedBody] = useState(prompt?.body ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const key = prompt?.key ?? pendingKey;
  const title = humanize(key);
  const detail = prompt && prompt.name && prompt.name !== title && prompt.name !== prompt.key ? prompt.name : "";
  const unchanged = comparablePrompt(draft) === comparablePrompt(savedBody);
  const rows = Math.min(24, Math.max(8, draft.split("\n").length));
  const pending = prompt == null;
  return (
    <section className="panel space-y-3 p-4" aria-labelledby={key ? `prompt-${key}` : undefined}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <h2 id={key ? `prompt-${key}` : undefined} className="text-sm font-medium">{title}</h2>
          <span className="tag">{prompt ? `v${prompt.version}` : <Bone width="2ch" />}</span>
        </div>
        <button
          className="btn primary shrink-0"
          disabled={pending || unchanged || saving}
          onClick={async () => {
            if (!prompt || unchanged || saving) return;
            setSaving(true);
            setError("");
            try {
              await api.savePrompt(prompt.key, prompt.name || prompt.key, draft);
              setSavedBody(draft);
              await client.invalidateQueries({ queryKey: ["prompts"] });
            } catch (cause) {
              setError(cause instanceof Error ? cause.message : "Save failed");
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "Saving…" : "Save new version"}
        </button>
      </div>
      {pending ? <p className="text-xs"><Bone width="12rem" /></p> : detail && <p className="text-xs text-[var(--muted)]">{detail}</p>}
      {prompt ? (
        <textarea
          className="field resize-y font-mono text-xs leading-relaxed"
          rows={rows}
          aria-label={`${title} prompt`}
          value={draft}
          spellCheck={false}
          onChange={(event) => setDraft(event.target.value)}
        />
      ) : (
        <FieldBone tall />
      )}
      {error && <p className="text-xs text-[var(--rose)]">{error}</p>}
    </section>
  );
}

export function PromptsPage() {
  const prompts = useQuery({ queryKey: ["prompts"], queryFn: api.prompts });
  const pending = prompts.data === undefined && !prompts.isError;
  const sections = latestPrompts(prompts.data ?? []);
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-medium">Prompt versions</h1>
        <p className="mt-1 max-w-3xl text-sm text-[var(--muted)]">
          Each prompt has its own editor. Saving creates a new version of that prompt. Past experiments keep the snapshot they launched with.
        </p>
      </div>
      {pending && <p className="sr-only">Loading prompts</p>}
      {prompts.isError && <p className="text-sm text-[var(--rose)]">Could not load prompts.</p>}
      {!pending && !prompts.isError && sections.length === 0 && (
        <p className="text-sm text-[var(--muted)]">No prompts yet.</p>
      )}
      <div className="space-y-4" aria-busy={pending || undefined}>
        {pending
          ? PROMPT_ORDER.map((key) => <PromptSection key={key} prompt={null} pendingKey={key} />)
          : sections.map((item) => <PromptSection key={item.id} prompt={item} />)}
      </div>
    </div>
  );
}

export function ComparePage() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => api.runs() });
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [comparisonId, setComparisonId] = useState("");
  const [matchLeft, setMatchLeft] = useState("");
  const [matchRight, setMatchRight] = useState("");
  const pending = runs.data === undefined && !runs.isError;
  const completed = pending ? null : (runs.data ?? []).filter((run) => run.status === "completed");
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-medium">Model comparison laboratory</h1>
      <p className="max-w-3xl text-sm text-[var(--muted)]">
        Object ids from different runs are not the same object. Assisted matches use labels and are marked as assisted. Disagreement is not a verdict.
      </p>
      {runs.isError && <p className="text-sm text-[var(--rose)]">{(runs.error as Error).message}</p>}
      <div className="grid items-stretch gap-4 lg:grid-cols-2" aria-busy={pending || undefined}>
        {pending && <p className="sr-only">Loading runs</p>}
        <RunColumn title="Left run" value={left} onChange={setLeft} runs={completed} />
        <RunColumn title="Right run" value={right} onChange={setRight} runs={completed} />
      </div>
      <button className="btn primary" disabled={!left || !right} onClick={async () => {
        const created = await api.compare(left, right);
        setComparisonId(created.id);
        setReport(created.report);
      }}>Compare</button>
      {report && (
        <div className="space-y-3">
          <pre className="panel mono max-h-[480px] overflow-auto p-3 text-xs">{JSON.stringify(report, null, 2)}</pre>
          <div className="flex flex-wrap gap-2">
            <input className="field max-w-[160px]" placeholder="Left object id" value={matchLeft} onChange={(event) => setMatchLeft(event.target.value)} />
            <input className="field max-w-[160px]" placeholder="Right object id" value={matchRight} onChange={(event) => setMatchRight(event.target.value)} />
            <button className="btn" onClick={async () => {
              const updated = await api.setMatches(comparisonId, [{ left: matchLeft, right: matchRight }]);
              setReport(updated.report);
            }}>Set manual match</button>
          </div>
        </div>
      )}
    </div>
  );
}

const RUN_FACTS: { label: string; truncate?: boolean; value: (run: RunSummary) => string }[] = [
  { label: "Model", value: (run) => humanize(run.model_id) || "unset" },
  { label: "Provider", value: (run) => humanize(run.provider_id) },
  { label: "Video", truncate: true, value: (run) => run.video_filename },
  { label: "Objects / actions", value: (run) => `${run.object_count} / ${run.action_count}` },
];

function RunFacts({ run }: { run: RunSummary }) {
  return (
    <dl className="mt-auto grid grid-cols-2 gap-x-4 gap-y-2 border-t border-[var(--line)] pt-3 text-xs">
      {RUN_FACTS.map((fact) => (
        <div key={fact.label} className={fact.truncate ? "min-w-0" : undefined}>
          <dt className="text-[var(--muted)]">{fact.label}</dt>
          <dd className={fact.truncate ? "truncate" : undefined}>{fact.value(run)}</dd>
        </div>
      ))}
    </dl>
  );
}

function RunColumn({
  title,
  value,
  onChange,
  runs,
}: {
  title: string;
  value: string;
  onChange: (id: string) => void;
  runs: RunSummary[] | null;
}) {
  const pending = runs == null;
  const selected = runs?.find((run) => run.id === value);
  return (
    <section className="panel flex h-full min-h-48 flex-col gap-3 p-4" aria-busy={pending || undefined}>
      <h2 className="text-sm font-medium">{title}</h2>
      {pending ? <FieldBone stretch /> : (
        <select className="field stretch" value={value} onChange={(event) => onChange(event.target.value)} aria-label={title}>
          <option value="">{title}</option>
          {runs.map((run) => <option key={run.id} value={run.id}>{humanizeText(run.name)}</option>)}
        </select>
      )}
      {selected ? <RunFacts run={selected} /> : (
        <p className="mt-auto text-xs text-[var(--muted)]">Select a completed run.</p>
      )}
    </section>
  );
}
