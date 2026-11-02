import { useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
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
        <h2>{item ? item.name : <Bone width="60%" />}</h2>
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
          <form className="row" onSubmit={(event) => void create(event)}>
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
          <li key={item.id}>
            <Link className="card" to={`/procedures/${item.id}/${item.review_state === "accepted" ? "verify" : item.step_count ? "procedure" : "teach"}`}>
              <ProcedureCardFace item={item} />
            </Link>
          </li>
        ))}
      </ul>
    </Layout>
  );
}
