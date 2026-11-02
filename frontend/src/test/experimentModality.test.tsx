import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../lib/api";
import { ExperimentPage } from "../pages/Experiment";

const model = (model_id: string, native_video: boolean) => ({
  model_id,
  label: model_id,
  image: true,
  native_video,
  structured_output: true,
  supports_temperature: false,
  max_images: 16,
  reasoning_efforts: [],
  notes: "",
});

function renderForm() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/experiment?video=v1"]}>
        <ExperimentPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// A wrapping label's text includes its options, so the field is found from its caption.
const field = (caption: string) => screen.getByText(caption).closest("label")!.querySelector("select")!;
const modalityField = () => field("Input modality");

describe("experiment input modality", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "videos").mockResolvedValue([] as never);
    vi.spyOn(api, "prompts").mockResolvedValue([] as never);
    vi.spyOn(api, "audioModels").mockResolvedValue({ models: [] } as never);
    vi.spyOn(api, "providers").mockResolvedValue([
      { id: "meta", models: [model("muse-wonder-1.0", true), model("image-only", false)] },
    ] as never);
  });

  it("defaults a video model to native video and submits sampled frames when chosen", async () => {
    const createRun = vi.spyOn(api, "createRun").mockResolvedValue({ id: "r1" } as never);
    renderForm();
    await waitFor(() => expect(modalityField().disabled).toBe(false));
    expect(modalityField().value).toBe("native_video");

    fireEvent.change(modalityField(), { target: { value: "sampled_frames" } });
    expect(modalityField().value).toBe("sampled_frames");

    fireEvent.click(screen.getByRole("button", { name: "Start analysis" }));
    await waitFor(() => expect(createRun).toHaveBeenCalled());
    expect(createRun.mock.calls[0][0]).toMatchObject({ modality: "sampled_frames" });
  });

  it("locks an image-only model to sampled frames", async () => {
    renderForm();
    await waitFor(() => expect(modalityField().disabled).toBe(false));
    fireEvent.change(field("Primary model"), { target: { value: "image-only" } });
    expect(modalityField().value).toBe("sampled_frames");
    expect(modalityField().disabled).toBe(true);
  });
});
