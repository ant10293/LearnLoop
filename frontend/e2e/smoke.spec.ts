import { expect, test } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Fixture clips and a stubbed Meta client. This proves the application flow, not model accuracy.
const dir = mkdtempSync(join(tmpdir(), "learnloop-e2e-"));
function clip(name: string, color: string) {
  const path = join(dir, `${name}.mp4`);
  execFileSync("ffmpeg", ["-v", "error", "-y", "-f", "lavfi", "-i", `color=c=${color}:s=640x360:d=10:r=15`, "-pix_fmt", "yuv420p", path]);
  return path;
}

test("teach, review, accept, verify, inspect evidence, confirm, export", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Learn a task. Verify the next attempt." })).toBeVisible();
  await page.getByRole("button", { name: "Create procedure" }).click();
  await page.getByLabel("Procedure name").fill("Cube assembly");
  await page.getByRole("button", { name: "Create", exact: true }).click();

  await expect(page.getByText("Supported: MP4, MOV, WebM")).toBeVisible();
  await page.screenshot({ path: "test-results/0-teach.png", fullPage: true });
  await page.locator('input[type="file"]').setInputFiles(clip("demo", "gray"));
  await page.getByRole("button", { name: "Learn the procedure" }).click({ timeout: 60_000 });
  await expect(page.getByRole("dialog", { name: "Approve Meta request" })).toContainText("No automatic retry");
  await page.screenshot({ path: "test-results/1-approve.png", fullPage: true });
  await page.getByRole("button", { name: "Send requests" }).click();

  await expect(page.getByRole("heading", { name: "Learned procedure" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("AI-generated draft")).toBeVisible();
  await page.getByLabel("Step 1 name").fill("Seat the red cube");
  await page.getByRole("button", { name: "Save edits" }).click();
  await expect(page.getByText("Human-reviewed")).toBeVisible();
  await expect(page.getByText("Human-edited: name")).toBeVisible();
  await page.screenshot({ path: "test-results/2-procedure.png", fullPage: true });
  await page.getByRole("button", { name: "Accept procedure" }).click();

  await expect(page.getByRole("heading", { name: "Verify an attempt" })).toBeVisible();
  await page.screenshot({ path: "test-results/0-verify.png", fullPage: true });
  await page.locator('input[type="file"]').setInputFiles(clip("omitted", "blue"));
  await page.getByRole("button", { name: "Analyze and verify" }).click({ timeout: 60_000 });
  await page.getByRole("button", { name: "Send requests" }).click();
  await expect(page.locator(".overall")).toContainText("FAIL", { timeout: 60_000 });
  const detail = page.getByRole("region", { name: /Details for/ });
  await expect(detail).toContainText("Missing component");
  await expect(detail).toContainText("Not seen anywhere in this recording");
  await page.screenshot({ path: "test-results/3-fail.png", fullPage: true });

  const seeks = page.locator("video");
  await page.getByRole("button", { name: /Seat the red cube/ }).click();
  await expect.poll(() => seeks.nth(0).evaluate((node) => (node as HTMLVideoElement).currentTime)).toBeCloseTo(2, 0);

  await page.locator('input[type="file"]').setInputFiles(clip("hidden", "green"));
  await page.getByRole("button", { name: "Analyze and verify" }).click({ timeout: 60_000 });
  await page.getByRole("button", { name: "Send requests" }).click();
  await expect(page.locator(".overall")).toContainText("INCONCLUSIVE", { timeout: 60_000 });
  await page.getByRole("button", { name: /Place green cube/ }).click();
  await page.screenshot({ path: "test-results/4-checkpoint.png", fullPage: true });
  await page.getByRole("button", { name: "Installed", exact: true }).click();
  await expect(page.locator(".overall")).toContainText("PASS");
  await expect(page.locator(".overall")).toContainText("Includes human-confirmed observations");
  await expect(page.getByRole("region", { name: /Details for/ })).toContainText("Human-confirmed");
  await page.screenshot({ path: "test-results/5-human.png", fullPage: true });

  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Export report" }).click();
  expect((await download).suggestedFilename()).toContain("learnloop-report");

  await page.locator(".brand").click();
  await expect(page.locator(".card")).toContainText("Cube assembly");
  await expect(page.locator(".card")).toContainText("3 expected steps");
  await page.screenshot({ path: "test-results/6-home.png", fullPage: true });
});
