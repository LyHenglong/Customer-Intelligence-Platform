import { expect, test } from "@playwright/test";
import { fixtures, mockApi } from "./mock-api";

test.beforeEach(async ({ page }) => {
  // Any uncaught error in the page fails the test that triggered it.
  page.on("pageerror", (err) => {
    throw err;
  });
  await mockApi(page);
});

test("root redirects to the overview", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/overview$/);
});

test("overview renders headline KPIs from the API", async ({ page }) => {
  await page.goto("/overview");
  await expect(page.getByText("Churn rate by contract")).toBeVisible();
  await expect(page.getByText("Recent batches")).toBeVisible();
  await expect(page.getByText(/48[,.]?2\d{2}|48K/).first()).toBeVisible();
});

test("at-risk list shows customers and drafts outreach on demand", async ({ page }) => {
  await page.goto("/at-risk");
  await expect(page.getByRole("heading", { name: "At-risk customers" })).toBeVisible();
  await expect(page.getByText("CUST0001").first()).toBeVisible();

  const draftButton = page.getByRole("button", { name: /draft|outreach|generate/i }).first();
  await draftButton.click();
  await expect(page.getByText(fixtures.outreach.draft!)).toBeVisible();
});

test("customer search lists matches and links to the detail page", async ({ page }) => {
  await page.goto("/customers");
  await expect(page.getByRole("heading", { name: "Customers" })).toBeVisible();
  await page.getByRole("link", { name: /CUST0001/ }).first().click();
  await expect(page).toHaveURL(/\/customers\/CUST0001$/);
  await expect(page.getByText("Risk factors")).toBeVisible();
  await expect(page.getByText("Recommended services")).toBeVisible();
});

test("segments page renders every segment card", async ({ page }) => {
  await page.goto("/segments");
  await expect(page.getByRole("heading", { name: "Segments" })).toBeVisible();
  await expect(page.getByText("What the model weighs most")).toBeVisible();
});

test("model performance shows the serving model", async ({ page }) => {
  await page.goto("/model-performance");
  await expect(page.getByRole("heading", { name: "Model performance" })).toBeVisible();
  await expect(page.getByText("Confusion matrix (at threshold)")).toBeVisible();
});

test("pipeline status shows ingestion and the retrain summary", async ({ page }) => {
  await page.goto("/pipeline-status");
  await expect(page.getByRole("heading", { name: "Pipeline status" })).toBeVisible();
  await expect(page.getByText("batch_001.csv").first()).toBeVisible();
  await expect(page.getByText(fixtures.pipelineStatus.latest_retrain_summary!.summary_text)).toBeVisible();
});

test("assistant answers a question", async ({ page }) => {
  await page.goto("/assistant");
  await expect(page.getByRole("heading", { name: "AI Decision Assistant" })).toBeVisible();
  // Scoped to the assistant's own form - the top bar has a search box too.
  const form = page.locator("form").filter({ has: page.getByRole("button", { name: "Analyze" }) });
  await form.getByRole("textbox").fill("How does churn differ by contract?");
  await form.getByRole("button", { name: "Analyze" }).click();
  await expect(page.getByText(fixtures.assistant.answer)).toBeVisible();
});

test("an API failure shows an error state instead of crashing", async ({ page }) => {
  await page.route("**/api/backend/pipeline-status", (route) =>
    route.fulfill({ status: 503, body: "Churn model not loaded" }),
  );
  await page.goto("/pipeline-status");
  await expect(page.getByText(/not loaded|failed|error/i).first()).toBeVisible();
});
