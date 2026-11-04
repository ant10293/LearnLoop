import { useQuery, useQueryClient } from "@tanstack/react-query";
import { EllipsisVertical, Pencil, Trash2 } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";
import { Link, useNavigate } from "react-router-dom";
import { Bone, TagBone } from "../components/Skeleton";
import { ErrorNote, Layout, ReviewTag, VerdictTag } from "../components/ui";
import { api, type ProcedureSummary } from "../lib/learnloop";

const CARD_PLACEHOLDERS = 3;

function ProcedureCardFace({ item }: { item?: ProcedureSummary }) {
  return (
    <>
      {item?.thumbnail_url ? <img src={item.thumbnail_url} alt="" /> : <div className="thumb-empty">{item ? "No demonstration yet" : <Bone className="bone-fill" />}</div>}
      <div className="card-body">
        <h2 title={item?.name}>{item ? item.name : <Bone width="60%" />}</h2>
        <p className="muted small">{item ? <>{item.step_count} expected {item.step_count === 1 ? "step" : "steps"} · {item.execution_count} {item.execution_count === 1 ? "attempt" : "attempts"}</> : <Bone width="80%" />}</p>
        <div className="row">
          {item ? <ReviewTag state={item.review_state} /> : <TagBone />}
          {item?.latest_outcome && <VerdictTag verdict={item.latest_outcome} />}
        </div>
      </div>
    </>
  );
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const procedures = useQuery({ queryKey: ["procedures"], queryFn: api.procedures });
  const [name, setName] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState("");

  async function create(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const created = await api.createProcedure(name);
      navigate(`/procedures/${created.id}/teach`);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <Layout>
      <section className="hero">
        <h1>Learn a task. Verify the next attempt.</h1>
        <p>Record a physical task once. LearnLoop proposes the procedure, you review it, and each new recording is checked against what was accepted, with the evidence shown.</p>
        {adding ? (
          <form className="row create-row" onSubmit={(event) => void create(event)}>
            <input className="field grow" autoFocus aria-label="Procedure name" placeholder="Name this procedure" value={name} onChange={(event) => setName(event.target.value)} />
            <button className="btn primary" disabled={!name.trim()}>Create</button>
            <button className="btn" type="button" onClick={() => setAdding(false)}>Cancel</button>
          </form>
        ) : (
          <button className="btn primary big" onClick={() => setAdding(true)}>Create procedure</button>
        )}
        <ErrorNote message={error} />
      </section>
      <ErrorNote message={procedures.error ? (procedures.error as Error).message : ""} />
      {procedures.isPending && <p className="sr-only">Loading procedures</p>}
      {procedures.data?.length === 0 && <p className="muted">No procedures yet. Create one to teach LearnLoop a task.</p>}
      <ul className="cards" aria-busy={procedures.isPending || undefined}>
        {procedures.isPending && Array.from({ length: CARD_PLACEHOLDERS }, (_, index) => (
          <li key={index}>
            <div className="card"><ProcedureCardFace /></div>
          </li>
        ))}
        {procedures.data?.map((item) => (
          <ProcedureCard key={item.id} item={item} />
        ))}
      </ul>
    </Layout>
  );
}

/** A procedure's card, with a menu to rename or delete it. Renaming opens a dialog over the page. */
function ProcedureCard({ item }: { item: ProcedureSummary }) {
  const client = useQueryClient();
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent ? event.key === "Escape" : !menu.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  async function remove() {
    setOpen(false);
    if (!window.confirm(`Delete “${item.name}”? Its recordings, learned procedures, and verification reports will be removed. This cannot be undone.`)) return;
    setBusy(true);
    setError("");
    try {
      await api.removeProcedure(item.id);
      client.removeQueries({ queryKey: ["procedure", item.id] });
      await client.invalidateQueries({ queryKey: ["procedures"] });
    } catch (exc) {
      setError((exc as Error).message);
      setBusy(false);
    }
  }

  return (
    <li>
      <div className="card-item">
        <Link className="card" to={`/procedures/${item.id}/${item.review_state === "accepted" ? "verify" : item.step_count ? "procedure" : "teach"}`}>
          <ProcedureCardFace item={item} />
        </Link>
        <div className="menu card-menu" ref={menu}>
          <button className="btn icon" aria-label={`Options for ${item.name}`} aria-haspopup="menu" aria-expanded={open} disabled={busy} onClick={() => setOpen((current) => !current)}>
            <EllipsisVertical size={18} aria-hidden="true" />
          </button>
          {open && (
            <div className="menu-list" role="menu">
              <button role="menuitem" className="accent" onClick={() => { setOpen(false); setRenaming(true); }}>Rename<Pencil size={15} aria-hidden="true" /></button>
              <button role="menuitem" className="danger" onClick={() => void remove()}>Delete<Trash2 size={15} aria-hidden="true" /></button>
            </div>
          )}
        </div>
      </div>
      <ErrorNote message={error} />
      {renaming && <RenameDialog item={item} onClose={() => setRenaming(false)} />}
    </li>
  );
}

function RenameDialog({ item, onClose }: { item: ProcedureSummary; onClose: () => void }) {
  const client = useQueryClient();
  const [draft, setDraft] = useState(item.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [onClose]);

  async function rename(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.renameProcedure(item.id, draft);
      await Promise.all([client.invalidateQueries({ queryKey: ["procedures"] }), client.invalidateQueries({ queryKey: ["procedure", item.id] })]);
      onClose();
    } catch (exc) {
      setError((exc as Error).message);
      setBusy(false);
    }
  }

  return createPortal(
    <div className="overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <form className="overlay-panel" role="dialog" aria-modal="true" aria-labelledby={`rename-${item.id}`} onSubmit={(event) => void rename(event)}>
        <h2 id={`rename-${item.id}`}>Rename procedure</h2>
        <input className="field" autoFocus onFocus={(event) => event.target.select()} aria-label="Procedure name" value={draft} onChange={(event) => setDraft(event.target.value)} />
        <ErrorNote message={error} />
        <div className="row overlay-actions">
          <button className="btn" type="button" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || !draft.trim()}>Save</button>
        </div>
      </form>
    </div>,
    document.body,
  );
}
