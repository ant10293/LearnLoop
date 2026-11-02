import { expect, test } from "@playwright/test";

test("fixture laboratory can be inspected, corrected, compared, and exported", async ({ page, request }) => {
  await expect.poll(async () => {
    const response = await request.get("http://127.0.0.1:8000/api/runs");
    const runs = await response.json();
    return runs.filter((run: { status: string }) => run.status === "completed").length;
  }, { timeout: 90_000 }).toBeGreaterThan(0);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Research dashboard" })).toBeVisible();
  await page.locator('a[href^="/runs/"]').first().click();
  await expect(page.getByText("Fixture — authored, not model output")).toBeVisible();
  await page.getByRole("link", { name: "Objects" }).click();
  await page.locator("button.panel").first().click();
  const rename = page.locator("input[id^='label-']");
  await rename.fill("Corrected label");
  await page.getByRole("button", { name: "Save label" }).click();
  await expect(page.getByText("Corrected label").first()).toBeVisible();
  await page.getByRole("link", { name: "Workflow" }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Export archive" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toContain(".zip");
});
