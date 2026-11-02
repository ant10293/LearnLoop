import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { Shell } from "./components/Shell";
import { queryClient } from "./lib/queryClient";
import { DashboardPage } from "./pages/Dashboard";
import { ExperimentPage } from "./pages/Experiment";
import { ComparePage, PromptsPage, ProvidersPage } from "./pages/Research";
import {
  EvaluatePage,
  InspectPage,
  LogsPage,
  ObjectsPage,
  PipelinePage,
  RunLayout,
  RunsPage,
  StatesPage,
  StepsPage,
  WorkflowPage,
} from "./pages/RunPages";
import { VideoDetailPage, VideosPage } from "./pages/Videos";
import { ProjectsPage } from "./pages/Projects";
import { AudioPage } from "./pages/Audio";
import { RecoveryPage } from "./pages/Recovery";

function TaskRedirect() {
  const { projectId = "" } = useParams();
  return <Navigate to={`/projects/${projectId}`} replace />;
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<DashboardPage />} />
            <Route path="videos" element={<VideosPage />} />
            <Route path="videos/:videoId" element={<VideoDetailPage />} />
            <Route path="projects" element={<ProjectsPage />} />
            <Route path="projects/:projectId" element={<ProjectsPage />} />
            <Route path="tasks" element={<Navigate to="/projects" replace />} />
            <Route path="tasks/:projectId" element={<TaskRedirect />} />
            <Route path="experiments/new" element={<ExperimentPage />} />
            <Route path="runs" element={<RunsPage />} />
            <Route path="providers" element={<ProvidersPage />} />
            <Route path="prompts" element={<PromptsPage />} />
            <Route path="compare" element={<ComparePage />} />
            <Route path="runs/:runId" element={<RunLayout />}>
              <Route index element={<InspectPage />} />
              <Route path="objects" element={<ObjectsPage />} />
              <Route path="states" element={<StatesPage />} />
              <Route path="workflow" element={<WorkflowPage />} />
              <Route path="steps" element={<StepsPage />} />
              <Route path="recovery" element={<RecoveryPage />} />
              <Route path="pipeline" element={<PipelinePage />} />
              <Route path="audio" element={<AudioPage />} />
              <Route path="evaluate" element={<EvaluatePage />} />
              <Route path="logs" element={<LogsPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
