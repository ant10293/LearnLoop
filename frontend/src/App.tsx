import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { queryClient } from "./lib/queryClient";
import { ProcedurePage } from "./pages/Procedure";
import { ProjectsPage } from "./pages/Projects";
import { TeachPage } from "./pages/Teach";
import { VerifyPage } from "./pages/Verify";

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/procedures/:id/teach" element={<TeachPage />} />
          <Route path="/procedures/:id/procedure" element={<ProcedurePage />} />
          <Route path="/procedures/:id/verify" element={<VerifyPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
