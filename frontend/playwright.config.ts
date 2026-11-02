import { defineConfig, devices } from "@playwright/test";

// The API here runs with a stubbed Meta client and authored answers. See backend/tests/e2e_server.py.
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  use: {
    baseURL: "http://127.0.0.1:5199",
    screenshot: "on",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "rm -rf ../data/e2e && ../backend/.venv/bin/python ../backend/tests/e2e_server.py",
      url: "http://127.0.0.1:8199/api/health",
      timeout: 120_000,
      reuseExistingServer: false,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5199 --strictPort",
      url: "http://127.0.0.1:5199",
      reuseExistingServer: false,
      env: { LEARNLOOP_API_ORIGIN: "http://127.0.0.1:8199" },
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
