import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  webServer: [
    {
      command:
        "rm -rf ../data/e2e && ../backend/.venv/bin/uvicorn app.main:app --app-dir ../backend --port 8000",
      url: "http://127.0.0.1:8000/api/health",
      timeout: 180_000,
      reuseExistingServer: false,
      env: {
        LEARNLOOP_AUTO_SEED: "1",
        LEARNLOOP_WORKER_ENABLED: "1",
        LEARNLOOP_DATA_DIR: "../data/e2e",
        LEARNLOOP_DATABASE_URL: "sqlite:///../data/e2e/learnloop.db",
      },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
