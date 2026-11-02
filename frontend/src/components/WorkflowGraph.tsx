import type { ReactNode } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  ReactFlow,
  getBezierPath,
  useStore,
  type Edge,
  type EdgeProps,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { humanize, nameObjects, objectLabels } from "../lib/humanize";
import type { Bundle } from "../lib/types";
import { useInspector } from "../stores/ui";
import { Bone } from "./Skeleton";

const COLORS: Record<string, string> = {
  necessary_physical: "#3ddc97",
  observed_temporal: "#7eb6ff",
  inferred: "#e6b450",
  unknown: "#e06c75",
};

const NODE_WIDTH = 230;
// Wide enough that a dependency label can sit in the gutter between cards.
const COLUMN_PITCH = 360;
const ROW_PITCH = 150;
const LABEL_WIDTH = 118;
const LABEL_HEIGHT = 20;

type Point = { x: number; y: number };
type Box = { x: number; y: number; width: number; height: number };

const edgeTypes = { dependency: DependencyEdge };

function controlOffset(distance: number) {
  if (distance >= 0) return distance * 0.5;
  return 0.25 * 25 * Math.sqrt(-distance);
}

function bezierPoint(t: number, p0: Point, p1: Point, p2: Point, p3: Point): Point {
  const u = 1 - t;
  return {
    x: u ** 3 * p0.x + 3 * u ** 2 * t * p1.x + 3 * u * t ** 2 * p2.x + t ** 3 * p3.x,
    y: u ** 3 * p0.y + 3 * u ** 2 * t * p1.y + 3 * u * t ** 2 * p2.y + t ** 3 * p3.y,
  };
}

function curveOf(source: Point, target: Point) {
  const bend = controlOffset(target.y - source.y);
  return {
    source,
    target,
    c1: { x: source.x, y: source.y + bend },
    c2: { x: target.x, y: target.y - bend },
  };
}

function boxesOverlap(a: Box, b: Box, pad = 0) {
  return a.x < b.x + b.width + pad && a.x + a.width > b.x - pad && a.y < b.y + b.height + pad && a.y + a.height > b.y - pad;
}

function labelBox(point: Point): Box {
  return { x: point.x - LABEL_WIDTH / 2, y: point.y - LABEL_HEIGHT / 2, width: LABEL_WIDTH, height: LABEL_HEIGHT };
}

/** Keep the natural midpoint when it is clear, otherwise slide along the curve. */
function labelSpots(edges: { id: string; source: Point; target: Point }[], nodes: Box[]) {
  const spots = new Map<string, Point>();
  const placed: Box[] = [];
  const ranked = [...edges].sort((a, b) => {
    const span = (edge: { source: Point; target: Point }) => Math.hypot(edge.target.x - edge.source.x, edge.target.y - edge.source.y);
    return span(a) - span(b);
  });
  for (const edge of ranked) {
    const { source, c1, c2, target } = curveOf(edge.source, edge.target);
    let chosen = bezierPoint(0.5, source, c1, c2, target);
    let best = Number.POSITIVE_INFINITY;
    for (let step = 0; step <= 16; step += 1) {
      const t = 0.18 + (0.64 * step) / 16;
      const point = bezierPoint(t, source, c1, c2, target);
      const box = labelBox(point);
      const blocked = nodes.some((node) => boxesOverlap(box, node, 8)) || placed.some((other) => boxesOverlap(box, other, 6));
      const score = Math.abs(t - 0.5);
      if (!blocked && score < best) {
        best = score;
        chosen = point;
      }
    }
    spots.set(edge.id, chosen);
    placed.push(labelBox(chosen));
  }
  return spots;
}

function DependencyEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, label, style }: EdgeProps) {
  const [path] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const spot = useStore((state) => {
    const bounds = state.nodes.map((node) => ({
      id: node.id,
      x: node.position.x,
      y: node.position.y,
      width: node.measured?.width || NODE_WIDTH,
      height: node.measured?.height || 72,
    }));
    const geometry = state.edges.flatMap((edge) => {
      const source = bounds.find((node) => node.id === edge.source);
      const target = bounds.find((node) => node.id === edge.target);
      if (!source || !target) return [];
      return [{
        id: edge.id,
        source: { x: source.x + source.width / 2, y: source.y + source.height },
        target: { x: target.x + target.width / 2, y: target.y },
      }];
    });
    return labelSpots(geometry, bounds).get(id) ?? { x: sourceX, y: sourceY };
  }, (a, b) => a.x === b.x && a.y === b.y);
  return (
    <>
      <BaseEdge id={id} path={path} style={style} />
      {label ? (
        <EdgeLabelRenderer>
          <div
            className="workflow-edge-label"
            style={{ transform: `translate(-50%, -50%) translate(${spot.x}px, ${spot.y}px)` }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

function GraphShell({ children, hidden = false }: { children: ReactNode; hidden?: boolean }) {
  return (
    <div className="workflow-graph h-[520px] overflow-hidden rounded-lg border border-[var(--line)]" aria-hidden={hidden || undefined}>
      {children}
    </div>
  );
}

export function WorkflowGraph({ bundle }: { bundle: Bundle | null }) {
  const selectStep = useInspector((state) => state.selectStep);
  const setTime = useInspector((state) => state.setTime);
  if (!bundle) {
    return (
      <GraphShell hidden>
        <Bone className="bone-fill" />
      </GraphShell>
    );
  }
  const labels = objectLabels(bundle.objects);
  const nodes: Node[] = bundle.steps.map((step, index) => ({
    id: step.local_id,
    position: { x: (index % 3) * COLUMN_PITCH, y: Math.floor(index / 3) * ROW_PITCH },
    data: {
      label: (
        <div className="text-left text-xs">
          <div className="mono text-[10px] text-[var(--muted)]">{step.number}. {humanize(step.evidence_class)}</div>
          <div className="font-medium">{nameObjects(step.name, labels)}</div>
          <div className="text-[var(--muted)]">{nameObjects(step.completion_condition, labels)}</div>
        </div>
      ),
    },
    style: {
      background: "#141820",
      color: "#e7ecf3",
      border: "1px solid #2a3140",
      width: NODE_WIDTH,
    },
  }));
  const edges: Edge[] = bundle.dependencies.map((dep) => ({
    id: dep.id,
    source: dep.from_step,
    target: dep.to_step,
    type: "dependency",
    label: humanize(dep.category),
    style: { stroke: COLORS[dep.category] ?? "#8b95a8" },
    data: { rationale: dep.rationale },
  }));
  return (
    <GraphShell>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        onNodeClick={(_event, node) => {
          selectStep(node.id);
          const step = bundle.steps.find((item) => item.local_id === node.id);
          if (step?.time_start != null) setTime(step.time_start);
        }}
        onEdgeClick={(_event, edge) => {
          window.alert(String(edge.data?.rationale || edge.label || "Dependency"));
        }}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </GraphShell>
  );
}
