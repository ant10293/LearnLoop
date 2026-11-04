import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { RecordingAnalysis } from "../components/RecordingAnalysis";
import { RecordingDrop } from "../components/RecordingDrop";
import { Bone, TagBone } from "../components/Skeleton";
import { ErrorNote, Film, Layout, ReviewTag, VerdictTag } from "../components/ui";
import { ObjectViewsNotice } from "../components/RecordingAnalysis";
import { api, basisLabels, clock, errorLabels, frameUrl, humanizeModel, type Answer, type FilmTrack, type ObjectViews, type Procedure, type Recording, type Role, type Run, type StepResult, type Verification, type VideoInfo } from "../lib/learnloop";

const RESULT_PLACEHOLDERS = 3;

export function VerifyPage() {
  const { id = "" } = useParams();
  const client = useQueryClient();
  const procedure = useQuery({
    queryKey: ["procedure", id],
    queryFn: () => api.procedure(id),
    refetchInterval: (query) => (query.state.data?.recordings.some((item) => !["ready", "failed"].includes(item.video.ingest_status)) ? 1000 : false),
  });
  // "" follows the latest attempt; null means the user deselected.
  const [selected, setSelected] = useState<string | null>("");
  const accepted = [...(procedure.data?.templates ?? [])].reverse().find((item) => item.acceptance_status === "accepted");
  const attempts = (procedure.data?.recordings ?? []).filter((item) => item.role === "execution");
  const current = selected === null ? undefined : attempts.find((item) => item.video.id === selected) ?? attempts[attempts.length - 1];

  const check = useCallback(
    async (run: Run) => {
      if (!accepted) return;
      await api.verify(accepted.id, run.id);
      await client.invalidateQueries({ queryKey: ["procedure", id] });
    },
    [accepted, client, id],
  );

  if (procedure.data && !accepted) {
    return (
      <Layout procedureId={id} title={procedure.data.name}>
        <h1>Verify</h1>
        <p className="lead">Review and accept the learned procedure first. A new attempt is checked against an accepted version.</p>
        <Link className="btn primary" to={`/procedures/${id}/procedure`}>Review the procedure</Link>
      </Layout>
    );
  }
  const brief = current ? procedure.data?.verifications.find((item) => item.execution_video_id === current.video.id) : undefined;
  return (
    <Layout procedureId={id} title={procedure.data?.name} titleLoading={procedure.isPending}>
      <div className="heading">
        <div>
          <h1>Verify an attempt</h1>
          {accepted && <p className="muted small">Checking against version {accepted.version} of the accepted procedure.</p>}
          {procedure.isPending && <p className="muted small"><Bone width="20rem" /></p>}
        </div>
        <div className="row">
          {accepted && <ReviewTag state={accepted.review_state} />}
          {procedure.isPending && <TagBone />}
        </div>
      </div>
      <ErrorNote message={procedure.error ? (procedure.error as Error).message : ""} />
      <RecordingDrop
        procedureId={id}
        role="execution"
        prompt="Drag and drop an attempt here or"
        recordings={attempts}
        loading={procedure.isPending}
        onChange={async (created) => {
          if (created) setSelected(created.video.id);
          await client.invalidateQueries({ queryKey: ["procedure", id] });
        }}
      />
      <div className="upload-gap" />
      {procedure.isPending && (
        <>
          <div className="tabs" aria-hidden="true"><span className="tab"><Bone width="7rem" /></span></div>
          <ReportSkeleton />
        </>
      )}
      {attempts.length > 0 && (
        <div className="tabs" role="tablist" aria-label="Attempts">
          {attempts.map((item) => (
            <AttemptTab key={item.video.id} recording={item} procedure={procedure.data as Procedure} active={item.video.id === current?.video.id} onClick={() => setSelected(item.video.id === current?.video.id ? null : item.video.id)} />
          ))}
        </div>
      )}
      {current && !brief && <section className="panel"><RecordingAnalysis key={current.video.id} recording={current} onDone={check} action="Analyze and verify" /></section>}
      {brief && procedure.data && <Report key={brief.id} verificationId={brief.id} procedure={procedure.data} recording={current as Recording} />}
    </Layout>
  );
}

function AttemptTab({ recording, procedure, active, onClick }: { recording: Recording; procedure: Procedure; active: boolean; onClick: () => void }) {
  const brief = procedure.verifications.find((item) => item.execution_video_id === recording.video.id);
  return (
    <button role="tab" aria-selected={active} className={`tab ${active ? "active" : ""}`} onClick={onClick}>
      {recording.video.filename}
      {brief && <VerdictTag verdict={brief.overall} />}
    </button>
  );
}

function Report({ verificationId, procedure, recording }: { verificationId: string; procedure: Procedure; recording: Recording }) {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["verification", verificationId],
    queryFn: () => api.verification(verificationId),
    refetchInterval: (state) => (state.state.data?.report.reinspection?.status === "pending" ? 1500 : false),
  });
  const [stepId, setStepId] = useState("");
  const [tick, setTick] = useState(0);
  const report = query.data?.report;
  const step = report?.steps.find((item) => item.step_id === stepId) ?? report?.steps.find((item) => item.verdict === "FAIL") ?? report?.steps[0];
  useEffect(() => setTick((value) => value + 1), [step?.step_id]);
  const demoRun = procedure.templates.find((item) => item.id === query.data?.template_id)?.source_run_id ?? "";
  const demoViews = useQuery({ queryKey: ["object-views", demoRun], queryFn: () => api.objectViews(demoRun), enabled: Boolean(demoRun) });
  const attemptRun = recording.run?.id ?? "";
  const attemptViews = useQuery({
    queryKey: ["object-views", attemptRun],
    queryFn: () => api.objectViews(attemptRun),
    enabled: Boolean(attemptRun),
    refetchInterval: (state) => (state.state.data?.running ? 1000 : false),
  });
  if (query.error) return <ErrorNote message={(query.error as Error).message} />;
  if (!query.data || !report || !step) return <ReportSkeleton video={recording.video} />;
  const verification = query.data;
  const demo = procedure.recordings.find((item) => item.video.id === report.demonstration_video_id);
  const template = procedure.templates.find((item) => item.id === verification.template_id);
  const evidenceTime = step.execution_time ?? step.evidence[0]?.timestamp ?? null;
  return (
    <ReportView
      overall={verification.overall}
      verdict={<VerdictTag verdict={verification.overall} large />}
      summary={summaryLine(verification)}
      basis={
        <>
          {report.human_assisted ? "Includes human-confirmed observations. Not a fully automatic result." : report.reinspection?.changed?.length ? "Decided by the deterministic verifier from the Meta analysis of this recording and a second Meta look at unclear steps." : "Decided by the deterministic verifier from the Meta analysis of this recording."}
          {" "}{recording.run?.source === "cached" ? "Cached Meta response." : recording.run?.source === "live" ? "Live Meta response." : ""}
        </>
      }
      action={<a className="btn" href={`/api/verifications/${verification.id}/export`} download={`learnloop-report-${verification.id}.json`}>Export report</a>}
      warnings={
        <>
          {report.warnings.map((warning) => <p key={warning} className="warning">{warning}</p>)}
          <EvidenceChecks report={report} />
        </>
      }
      demonstration={<Film src={demo?.video.playback_url ?? ""} time={step.demo_time_start} seek={tick} label="Demonstration" caption={`How “${step.name}” was shown`} tracks={roleTracks(demoViews.data, template?.document.roles ?? [])} />}
      attempt={
        <>
          <Film src={recording.video.playback_url} time={evidenceTime} seek={tick} label="This attempt" caption={evidenceTime == null ? "No timestamp was tied to this step." : "Evidence moment for this step"} tracks={objectTracks(attemptViews.data)} />
          {attemptRun && <ObjectViewsNotice runId={attemptRun} views={attemptViews.data} onStarted={() => void client.invalidateQueries({ queryKey: ["object-views", attemptRun] })} />}
        </>
      }
      results={report.steps.map((item) => (
        <li key={item.step_id}>
          <button className={`result ${item.step_id === step.step_id ? "active" : ""}`} onClick={() => { setStepId(item.step_id); setTick((value) => value + 1); }}>
            <VerdictTag verdict={item.verdict} />
            <span className="grow">{item.name}</span>
            {!item.required && <span className="muted small">Not required</span>}
          </button>
        </li>
      ))}
      detail={<StepDetail step={step} verification={verification} video={recording.video} onChange={() => client.invalidateQueries({ queryKey: ["verification", verificationId] })} onSeek={() => setTick((value) => value + 1)} />}
      diagnostics={
        <dl>
          <dt>Model</dt><dd>{verification.config.provider_id} / {humanizeModel(verification.config.model_id)}, {verification.config.modality}</dd>
          <dt>Source</dt><dd>{recording.run?.source === "cached" ? `Cached Meta response${recording.run.cached_at ? ` from ${recording.run.cached_at}` : ""}` : "Live Meta response"}, {recording.run?.provider_requests ?? 0} requests in this run</dd>
          <dt>Procedure</dt><dd>Version {verification.template_version}, {verification.template_id}</dd>
          <dt>Execution analysis</dt><dd>Run {verification.execution_run_id}, revision {verification.execution_revision_id}</dd>
          {report.roles.map((role) => <span key={role.role_id} className="contents"><dt>{role.label}</dt><dd>{role.status}. {role.note}</dd></span>)}
        </dl>
      }
    />
  );
}

/** The demonstration's boxes, named as the accepted procedure names its roles. */
function roleTracks(views: ObjectViews | undefined, roles: Role[]): FilmTrack[] {
  if (!views || views.running) return [];
  return roles.flatMap((role) => {
    const track = views.objects[role.source_object_ids[0]]?.track;
    return track?.length ? [{ label: role.label, track }] : [];
  });
}

/** The attempt's boxes, named as its own analysis named them. */
function objectTracks(views: ObjectViews | undefined): FilmTrack[] {
  if (!views || views.running) return [];
  return Object.values(views.objects).flatMap((view) => (view.track?.length ? [{ label: view.label, track: view.track }] : []));
}

/** The report's layout, for the loaded report and its skeleton. */
function ReportView({ overall = "", verdict, summary, basis, action, warnings, demonstration, attempt, results, detail, diagnostics }: { overall?: string; verdict: ReactNode; summary: ReactNode; basis: ReactNode; action: ReactNode; warnings?: ReactNode; demonstration: ReactNode; attempt: ReactNode; results: ReactNode; detail: ReactNode; diagnostics?: ReactNode }) {
  return (
    <>
      <section className={`overall ${overall}`}>
        {verdict}
        <div>
          <p className="overall-line">{summary}</p>
          <p className="muted small">{basis}</p>
        </div>
        {action}
      </section>
      {warnings}
      <div className="films">
        {demonstration}
        {attempt}
      </div>
      <div className="split report">
        <ol className="results" aria-label="Step results">{results}</ol>
        {detail}
      </div>
      <details className="panel diagnostics">
        <summary>Diagnostics</summary>
        {diagnostics}
      </details>
    </>
  );
}

function ReportSkeleton({ video }: { video?: VideoInfo }) {
  return (
    <div aria-busy="true">
      <p className="sr-only">Loading the report</p>
      <ReportView
        verdict={<span className="verdict muted large"><Bone width="6rem" /></span>}
        summary={<Bone width="60%" />}
        basis={<Bone width="80%" />}
        action={<Bone width="6rem" />}
        demonstration={<Film label="Demonstration" caption={<Bone width="50%" />} />}
        attempt={<Film label="This attempt" caption={<Bone width="50%" />} />}
        results={Array.from({ length: RESULT_PLACEHOLDERS }, (_, index) => (
          <li key={index}>
            <div className="result">
              <span className="verdict muted"><Bone width="3rem" /></span>
              <span className="grow"><Bone width="60%" /></span>
            </div>
          </li>
        ))}
        detail={
          <StepDetailFrame
            title={<Bone width="40%" />}
            verdict={<span className="verdict muted"><Bone width="3rem" /></span>}
            basis={<Bone width="50%" />}
            expected={<Bone width="70%" />}
            observed={<Bone width="60%" />}
          >
            <div className="frames">
              <FrameFigure video={video} caption={<Bone width="80%" />} />
            </div>
          </StepDetailFrame>
        }
      />
    </div>
  );
}

function summaryLine(verification: Verification): string {
  const { verified, failed, unresolved } = verification.report.score;
  return `${verified} verified, ${failed} failed, ${unresolved} could not be verified.`;
}

function StepDetailFrame({ label, title, verdict, basis, category, expected, observed, why, children }: { label?: string; title: ReactNode; verdict: ReactNode; basis: ReactNode; category?: ReactNode; expected: ReactNode; observed: ReactNode; why?: ReactNode; children?: ReactNode }) {
  return (
    <section className="panel detail" aria-label={label}>
      <div className="row">
        <h2 className="grow">{title}</h2>
        {verdict}
      </div>
      <p className="basis">{basis}</p>
      {category}
      <dl className="facts">
        <dt>Expected</dt><dd>{expected}</dd>
        <dt>Observed</dt><dd>{observed}</dd>
        {why && <><dt>Why</dt><dd>{why}</dd></>}
      </dl>
      {children}
    </section>
  );
}

/** A recorded frame's box, shaped like its video, so the panel keeps its height until the picture arrives. No `src` is the skeleton. */
function FrameImage({ video, src, alt, onClick }: { video?: VideoInfo; src?: string; alt?: string; onClick?: () => void }) {
  const [settled, setSettled] = useState(false);
  const shape = video?.width && video.height ? { aspectRatio: `${video.width} / ${video.height}` } : undefined;
  return (
    <div className="frame-image" style={shape}>
      {src && <img src={src} alt={alt} onClick={onClick} onLoad={() => setSettled(true)} onError={() => setSettled(true)} />}
      {!settled && <Bone className="bone-fill" />}
    </div>
  );
}

function FrameFigure({ video, src, alt, caption, onClick }: { video?: VideoInfo; src?: string; alt?: string; caption: ReactNode; onClick?: () => void }) {
  return (
    <figure>
      <FrameImage video={video} src={src} alt={alt} onClick={onClick} />
      <figcaption className="small">{caption}</figcaption>
    </figure>
  );
}

function StepDetail({ step, verification, video, onChange, onSeek }: { step: StepResult; verification: Verification; video: VideoInfo; onChange: () => Promise<void>; onSeek: () => void }) {
  const moments = step.evidence.filter((item) => item.timestamp != null);
  const observations = verification.human_observations.filter((item) => item.step_id === step.step_id);
  return (
    <StepDetailFrame
      label={`Details for ${step.name}`}
      title={step.name}
      verdict={<VerdictTag verdict={step.verdict} />}
      basis={<>{basisLabels[step.basis]}{step.basis === "human_confirmed" && step.machine_verdict !== step.verdict ? `. Automatic reading: ${step.machine_verdict.replaceAll("_", " ")}` : ""}</>}
      category={step.error_category && <p className="category">{errorLabels[step.error_category] ?? step.error_category}</p>}
      expected={step.expected}
      observed={step.observed}
      why={step.explanation !== step.observed && step.explanation}
    >
      {moments.length > 0 && (
        <div className="frames">
          {moments.slice(0, 3).map((item, index) => (
            <FrameFigure
              key={`${item.timestamp}-${index}`}
              video={video}
              src={frameUrl(video.id, item.timestamp as number)}
              alt={`Frame at ${clock(item.timestamp)}`}
              onClick={onSeek}
              caption={<><span className="mono">{clock(item.timestamp)}</span> {item.provenance === "HUMAN_ANNOTATION" ? "Reviewer" : "Meta"}: {item.observation}</>}
            />
          ))}
        </div>
      )}
      {moments.length === 0 && <p className="muted small">No frame supports this step, so no frame is shown.</p>}
      {observations.map((item) => (
        <p key={item.id} className="edited">{item.reviewer || "Reviewer"} recorded “{item.answer.replaceAll("_", " ")}” at {clock(item.timestamp)}{item.note ? `: ${item.note}` : ""}</p>
      ))}
      {(step.verdict === "CANNOT_VERIFY" || step.basis === "human_confirmed" || step.basis === "not_observed") && <Checkpoint step={step} verification={verification} video={video} onChange={onChange} />}
    </StepDetailFrame>
  );
}

function Checkpoint({ step, verification, video, onChange }: { step: StepResult; verification: Verification; video: VideoInfo; onChange: () => Promise<void> }) {
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const timestamp = Math.max(video.duration_sec - 0.2, 0);

  async function answer(value: Answer) {
    setError("");
    try {
      await api.observe(verification.id, step.step_id, { answer: value, timestamp, note });
      await onChange();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <div className="checkpoint">
      <h3>Review the final frame</h3>
      <FrameImage video={video} src={frameUrl(video.id, timestamp)} alt={`Final frame at ${clock(timestamp)}`} />
      <p className="small">Expected: {step.expected}. Your answer is stored as a human observation at {clock(timestamp)}, and the verifier reads it as evidence.</p>
      <div className="row">
        <input className="field grow" aria-label="Note" placeholder="Note (optional)" value={note} onChange={(event) => setNote(event.target.value)} />
      </div>
      <div className="row">
        <button className="btn" onClick={() => void answer("installed")}>Installed</button>
        <button className="btn" onClick={() => void answer("not_installed")}>Not installed</button>
        <button className="btn" onClick={() => void answer("cannot_determine")}>Cannot determine</button>
      </div>
      <ErrorNote message={error} />
    </div>
  );
}


/** What the automatic checks added: appearance matching, tracking recovery, and a second look at unclear steps. */
function EvidenceChecks({ report }: { report: Verification["report"] }) {
  const lines: string[] = [];
  const reinspection = report.reinspection;
  if (reinspection?.status === "pending") lines.push(`Reinspecting ${reinspection.steps?.length ?? 0} unclear step${reinspection.steps?.length === 1 ? "" : "s"} with a few frames…`);
  else if (reinspection?.status === "completed") {
    const changed = reinspection.changed ?? [];
    lines.push(changed.length ? `Reinspection resolved ${changed.length} step${changed.length === 1 ? "" : "s"}: ${changed.map((item) => `${item.step_id} ${item.before} → ${item.after}`).join(", ")}.` : "Reinspection ran. The unclear steps are still unclear, so their verdicts stand.");
  } else if (reinspection?.status === "unavailable" || reinspection?.status === "failed") lines.push(`Reinspection unavailable. ${reinspection.reason ?? ""}`.trim());
  const appearance = report.appearance;
  if (appearance?.used) {
    const recovered = appearance.recovery?.recovered_sightings ?? 0;
    lines.push(`Appearance matching used the procedure's references (${appearance.gallery.objects_learned ?? 0} of ${appearance.gallery.objects ?? 0} objects).${recovered ? ` Tracking recovery restored ${recovered} boxes.` : ""} Dashed boxes are not confirmed by appearance. Appearance never passes or fails a step.`);
  } else if (appearance && (appearance.gallery.status === "building" || appearance.error)) {
    lines.push(appearance.gallery.status === "building" ? "Object references were still being built, so appearance matching was not used." : "Appearance matching was not used for this recording.");
  }
  if (!lines.length) return null;
  return <div className="muted small">{lines.map((line) => <p key={line}>{line}</p>)}</div>;
}
