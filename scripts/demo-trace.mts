// npm run demo-trace: runs one refund conversation through Ask Copper on the
// Support page (Copper hands off to the Refund agent, which asks the app) and
// shares its Langfuse trace, then prints the public link. Module 5, Part 2.
import { existsSync } from "node:fs";

if (existsSync(".env.local")) process.loadEnvFile(".env.local");

const { enableTracing, flushTraces, isLangfuseConfigured } =
  await import("@/lib/ai/langfuse");
const { finalValue } = await import("@/lib/ai/live");
const { runTurn } = await import("@/lib/copper/assistant/run");

if (!isLangfuseConfigured()) {
  console.error("Set the LANGFUSE_* keys in .env.local first.");
  process.exit(2);
}
enableTracing();

const { progress, traceId } = await runTurn({
  messages: [
    {
      role: "user",
      content: "Hi, I'd like a refund for my Copper Home, order #1004.",
    },
  ],
  useTools: true,
  desk: { limitUsd: 50, refunds: [] },
  traceName: "refund-conversation",
  publicTrace: true,
});
const final = await finalValue(progress);
await flushTraces();
console.log("Reply:", final.reply);
console.log(
  "Steps:",
  final.steps.map((s) => (s.type === "tool" ? s.name : s.type)).join(" -> "),
);
console.log("Trace id:", traceId);
process.exit(0);
