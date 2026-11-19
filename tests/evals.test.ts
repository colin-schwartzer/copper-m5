// Module 5: the golden set, the rule checks, the AI judge, the single support
// agent and Langfuse tracing. Mock model and no Langfuse keys, so it runs in
// CI with no secrets. (The real golden-set run is `npm run eval`.)
import type { LanguageModelV4StreamPart as ModelStreamPart } from "@ai-sdk/provider";
import { convertArrayToReadableStream, MockLanguageModelV4 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { finalValue } from "@/lib/ai/live";
import { toolCallsOf } from "@/lib/copper/assistant/progress";
import type { Desk } from "@/lib/copper/orders";
import { GOLDEN_SET, runChecks } from "@/lib/evals/golden";

let model: MockLanguageModelV4;
vi.mock("@ai-sdk/google", () => ({
  createGoogleGenerativeAI: () => () => model,
}));

type StreamPart = Parameters<
  typeof convertArrayToReadableStream<ModelStreamPart>
>[0][number];

const usage = {
  inputTokens: {
    total: 500,
    noCache: 500,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 50, text: 50, reasoning: undefined },
};
const finish = (reason: "stop" | "tool-calls"): StreamPart => ({
  type: "finish",
  finishReason: { unified: reason, raw: undefined },
  usage,
});
const says = (text: string): StreamPart[] => [
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: text },
  { type: "text-end", id: "t1" },
  finish("stop"),
];
const calls = (toolName: string, input: unknown): StreamPart[] => [
  { type: "tool-input-start", id: "call-1", toolName },
  { type: "tool-input-end", id: "call-1" },
  {
    type: "tool-call",
    toolCallId: "call-1",
    toolName,
    input: JSON.stringify(input),
  },
  finish("tool-calls"),
];
function modelStreaming(...steps: StreamPart[][]) {
  let step = 0;
  return new MockLanguageModelV4({
    doStream: async () => ({
      stream: convertArrayToReadableStream(steps[step++] ?? says("")),
    }),
  });
}

const desk = (limitUsd = 50): Desk => ({ limitUsd, refunds: [] });

beforeEach(() => {
  vi.resetModules();
  vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
  vi.stubEnv("LANGFUSE_PUBLIC_KEY", "");
  vi.stubEnv("LANGFUSE_SECRET_KEY", "");
  vi.spyOn(console, "info").mockImplementation(() => {});
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("the golden set", () => {
  it("has unique, well-formed questions that cover every kind of answer", () => {
    const ids = GOLDEN_SET.cases.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(GOLDEN_SET.cases.length).toBeGreaterThanOrEqual(15);
    for (const c of GOLDEN_SET.cases) {
      expect(c.messages.at(-1)?.role, c.id).toBe("user");
      expect(c.expected.length, c.id).toBeGreaterThan(20);
      for (const p of [
        ...(c.checks.mustMatch ?? []),
        ...(c.checks.mustNotMatch ?? []),
      ])
        expect(() => new RegExp(p, "i"), `${c.id}: ${p}`).not.toThrow();
    }
    expect(new Set(GOLDEN_SET.cases.map((c) => c.category))).toEqual(
      new Set(["products", "documents", "refunds", "routing", "safety"]),
    );
  });
});

describe("the rule checks", () => {
  const tools = [{ name: "get_prices" }, { name: "calculate_cost" }];

  it("pass when every rule holds", () => {
    expect(
      runChecks(
        {
          toolsAny: ["calculate_cost"],
          toolsNone: ["request_refund"],
          mustMatch: ["\\$?436"],
          mustNotMatch: ["sold out"],
        },
        "It comes to $436 in total.",
        tools,
      ),
    ).toEqual({ passed: true, failures: [] });
  });

  it("name each broken rule", () => {
    const result = runChecks(
      {
        toolsAny: ["search_docs"],
        toolsNone: ["get_prices"],
        mustMatch: ["436"],
        mustNotMatch: ["total"],
      },
      "The total is about 400.",
      tools,
    );
    expect(result.passed).toBe(false);
    expect(result.failures).toHaveLength(4);
  });
});

describe("the AI judge", () => {
  it("grades an answer and goes through the logged AI wrapper", async () => {
    model = new MockLanguageModelV4({
      doGenerate: async () => ({
        content: [
          {
            type: "text",
            text: JSON.stringify({
              reasoning: "Right price, from a tool.",
              accuracy: 5,
              policy: 4,
            }),
          },
        ],
        finishReason: { unified: "stop", raw: undefined },
        usage,
        warnings: [],
      }),
    });
    const { judgeAnswer } = await import("@/lib/evals/judge");
    const verdict = await judgeAnswer(
      GOLDEN_SET.cases[0],
      "A Copper Home is $149.",
      [{ name: "get_prices", output: { products: [] } }],
    );
    expect(verdict).toEqual({
      reasoning: "Right price, from a tool.",
      accuracy: 5,
      policy: 4,
    });
    // The judge sees the question, what a good answer says and the reply.
    const prompt = JSON.stringify(model.doGenerateCalls[0].prompt);
    expect(prompt).toContain("How much is a Copper Home?");
    expect(prompt).toContain("A good answer: Copper Home costs $149");
    expect(prompt).toContain("A Copper Home is $149.");
    const logged = vi
      .mocked(console.info)
      .mock.calls.filter(([tag]) => tag === "[ai_calls]")
      .map(([, row]) => JSON.parse(row as string));
    expect(logged[0]).toMatchObject({ module: 5, feature: "eval-judge" });
  });
});

describe("the single support agent", () => {
  it("answers refunds itself: one agent, every tool, no handoff", async () => {
    model = modelStreaming(
      calls("request_refund", { orderId: "1001" }),
      says("Your $7.99 is refunded."),
    );
    const { runTurn } = await import("@/lib/copper/assistant/run");
    const result = await finalValue(
      (
        await runTurn({
          messages: [{ role: "user", content: "Refund order 1001" }],
          useTools: true,
          desk: desk(),
          mode: "single",
        })
      ).progress,
    );
    expect(result.status).toBe("done");
    expect(result.reply).toBe("Your $7.99 is refunded.");
    expect(result.steps.some((s) => s.type === "handoff")).toBe(false);
    expect(toolCallsOf(result)[0]).toMatchObject({
      name: "request_refund",
      output: { status: "refunded", amountUsd: 7.99 },
    });
    // One model run, with Copper's tools and the refund tools, and neither
    // handoff tool.
    expect(model.doStreamCalls).toHaveLength(2); // the tool call, then the reply
    const names = model.doStreamCalls[0].tools?.map((t) => t.name).sort();
    expect(names).toEqual([
      "calculate_cost",
      "check_availability",
      "get_orders",
      "get_prices",
      "get_products",
      "request_refund",
      "search_docs",
    ]);
    expect(JSON.stringify(model.doStreamCalls[0].prompt)).toContain(
      "You also handle refunds yourself",
    );
  });

  it("is only used on the Support page: without a desk it is plain Copper", async () => {
    model = modelStreaming(says("Hello!"));
    const { runTurn } = await import("@/lib/copper/assistant/run");
    const result = await finalValue(
      (
        await runTurn({
          messages: [{ role: "user", content: "Hi" }],
          useTools: true,
          mode: "single",
        })
      ).progress,
    );
    expect(result.reply).toBe("Hello!");
    expect(model.doStreamCalls[0].tools?.map((t) => t.name)).not.toContain(
      "request_refund",
    );
  });
});

describe("Langfuse tracing", () => {
  it("does nothing without keys: no trace, and chats still work", async () => {
    model = modelStreaming(says("Hi!"));
    const langfuse = await import("@/lib/ai/langfuse");
    expect(langfuse.isLangfuseConfigured()).toBe(false);
    expect(langfuse.startTurn({ name: "t", input: "x" })).toBeUndefined();
    await expect(langfuse.flushTraces()).resolves.toBeUndefined();
    const { runTurn } = await import("@/lib/copper/assistant/run");
    const out = await runTurn({
      messages: [{ role: "user", content: "Hi" }],
      useTools: false,
    });
    expect(out.traceId).toBeUndefined();
    expect((await finalValue(out.progress)).reply).toBe("Hi!");
  });

  it("is on when both keys are set", async () => {
    vi.stubEnv("LANGFUSE_PUBLIC_KEY", "pk-lf-test");
    vi.stubEnv("LANGFUSE_SECRET_KEY", "sk-lf-test");
    const langfuse = await import("@/lib/ai/langfuse");
    expect(langfuse.isLangfuseConfigured()).toBe(true);
  });
});
