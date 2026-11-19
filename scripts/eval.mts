// npm run eval: runs the golden set through Ask Copper and grades every answer
// with the rule checks and the AI judge (Module 5).
//
//   npm run eval                       the two agents (what ships), gated
//   npm run eval -- --mode=single      the single support agent
//   npm run eval:compare               both, side by side, not gated
//
// Options: --mode=two-agents|single|both  --gate=<mode>|none  --min-pass-rate=0.8
//          --only=<case id>[,<case id>]
// With Langfuse keys set, the golden set is synced to a Langfuse dataset, each
// run becomes a dataset run (so runs can be compared in Langfuse) and the
// scores are saved on each answer's trace. Without keys it still runs and
// prints everything here. Exits 1 when a gated run's pass rate is too low.
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { LangfuseClient } from "@langfuse/client";

if (existsSync(".env.local")) process.loadEnvFile(".env.local");

const { enableTracing, flushTraces, isLangfuseConfigured } =
  await import("@/lib/ai/langfuse");
const { isAiConfigured } = await import("@/lib/ai/call");
const { GOLDEN_SET } = await import("@/lib/evals/golden");
const { runCase } = await import("@/lib/evals/run-case");
type CaseResult = Awaited<ReturnType<typeof runCase>>;
type Mode = "two-agents" | "single";

const args = new Map(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith("--"))
    .map((a) => {
      const [key, value = "true"] = a.slice(2).split("=");
      return [key, value] as const;
    }),
);

const modeArg = args.get("mode") ?? process.env.EVAL_MODE ?? "two-agents";
if (!["two-agents", "single", "both"].includes(modeArg)) {
  console.error(`--mode must be two-agents, single or both (got ${modeArg}).`);
  process.exit(2);
}
const modes: Mode[] =
  modeArg === "both" ? ["two-agents", "single"] : [modeArg as Mode];
const gateArg = args.get("gate") ?? (modeArg === "both" ? "none" : modeArg);
const minPassRate = Number(
  args.get("min-pass-rate") ?? process.env.EVAL_MIN_PASS_RATE ?? "0.8",
);
const only = args.get("only")?.split(",");

if (!isAiConfigured()) {
  console.error(
    "The eval needs GOOGLE_GENERATIVE_AI_API_KEY (in .env.local, or an Actions secret in CI).",
  );
  process.exit(2);
}

const cases = GOLDEN_SET.cases.filter((c) => !only || only.includes(c.id));
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
// With --only, run locally (traced, but not as a Langfuse dataset run), since
// a dataset run covers the whole dataset.
const langfuse =
  isLangfuseConfigured() && !only ? new LangfuseClient() : undefined;
enableTracing();
const DATASET = GOLDEN_SET.name;

type Totals = { cost: number; latencyMs: number; calls: number };
const summaries: {
  mode: Mode;
  results: CaseResult[];
  agent: Totals;
  judge: Totals;
  datasetRun?: string;
}[] = [];

// Every model call prints an [ai_calls] row. Tapping them gives each case's
// cost and time, with the judge's own calls counted apart from the agents'.
let tap: { agent: Totals; judge: Totals } | undefined;
const originalInfo = console.info.bind(console);
console.info = (...parts: unknown[]) => {
  if (parts[0] === "[ai_calls]" && tap) {
    try {
      const row = JSON.parse(String(parts[1]));
      const bucket = row.feature === "eval-judge" ? tap.judge : tap.agent;
      bucket.cost += row.cost_usd ?? 0;
      bucket.latencyMs += row.latency_ms ?? 0;
      bucket.calls += 1;
    } catch {
      // ignore a row we can't read
    }
    return; // keep the console quiet: the table below says it all
  }
  originalInfo(...parts);
};

// Google's free tier allows about 15 requests a minute, and one golden-set
// question takes 3 to 6 calls. So pace the questions, and when a question hits
// the limit anyway, wait and try that question again.
const PACE_MS = Number(process.env.EVAL_PACE_MS ?? "6000");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const rateLimited = (r: CaseResult) =>
  /quota|rate.?limit|429/i.test(r.judgeError ?? "") ||
  (r.reply.startsWith("Sorry, something went wrong") && !r.toolCalls.length);

const empty = (): Totals => ({ cost: 0, latencyMs: 0, calls: 0 });
const pct = (n: number, d: number) => (d ? Math.round((100 * n) / d) : 0);

async function syncDataset(client: LangfuseClient) {
  await client.api.datasets.create({
    name: DATASET,
    description: GOLDEN_SET.description,
  });
  for (const c of GOLDEN_SET.cases) {
    await client.api.datasetItems.create({
      datasetName: DATASET,
      id: `${DATASET}:${c.id}`,
      input: c.messages,
      expectedOutput: c.expected,
      metadata: { caseId: c.id, category: c.category },
    });
  }
}

async function runMode(mode: Mode) {
  const results: CaseResult[] = [];
  const perCase = new Map<string, { agent: Totals; judge: Totals }>();
  const runName = `${mode}-${stamp}`;
  const total = { agent: empty(), judge: empty() };

  const task = async (testCase: (typeof cases)[number]) => {
    let result!: CaseResult;
    let used = { agent: empty(), judge: empty() };
    for (let attempt = 1; attempt <= 4; attempt++) {
      await sleep(attempt === 1 ? PACE_MS : 35_000);
      tap = { agent: empty(), judge: empty() };
      result = await runCase(testCase, mode, `eval:${mode}`);
      used = tap;
      tap = undefined;
      if (!rateLimited(result)) break;
      originalInfo(`  ...${testCase.id} hit the rate limit, retrying`);
    }
    perCase.set(testCase.id, used);
    for (const key of ["agent", "judge"] as const) {
      total[key].cost += used[key].cost;
      total[key].latencyMs += used[key].latencyMs;
      total[key].calls += used[key].calls;
    }
    results.push(result);
    const mark = result.passed ? "PASS" : "FAIL";
    originalInfo(`  ${mark}  ${mode.padEnd(10)} ${testCase.id}`);
    return result;
  };

  if (langfuse) {
    const dataset = await langfuse.dataset.get(DATASET);
    await dataset.runExperiment({
      name: `Ask Copper: ${mode}`,
      runName,
      description: `Golden set on the ${mode} setup`,
      metadata: { mode },
      maxConcurrency: 1,
      task: async (item) => {
        const id = (item.metadata as { caseId?: string } | undefined)?.caseId;
        const testCase = cases.find((c) => c.id === id)!;
        const result = await task(testCase);
        return result.reply;
      },
      evaluators: [
        async ({ metadata }) => {
          const r = results.find(
            (x) => x.id === (metadata as { caseId?: string })?.caseId,
          );
          if (!r) return [];
          const usage = perCase.get(r.id);
          return [
            {
              name: "passed",
              value: r.passed ? 1 : 0,
              comment: r.checks.failures.join("; ") || r.judgeError,
            },
            {
              name: "rule_checks",
              value: r.checks.passed ? 1 : 0,
              comment: r.checks.failures.join("; ") || undefined,
            },
            ...(r.verdict
              ? [
                  {
                    name: "judge_accuracy",
                    value: r.verdict.accuracy,
                    comment: r.verdict.reasoning,
                  },
                  {
                    name: "judge_policy",
                    value: r.verdict.policy,
                    comment: r.verdict.reasoning,
                  },
                ]
              : []),
            { name: "agent_cost_usd", value: usage?.agent.cost ?? 0 },
            { name: "agent_latency_ms", value: usage?.agent.latencyMs ?? 0 },
          ];
        },
      ],
      runEvaluators: [
        async () => [
          {
            name: "pass_rate",
            value: results.length
              ? results.filter((r) => r.passed).length / results.length
              : 0,
          },
        ],
      ],
    });
  } else {
    for (const testCase of cases) await task(testCase);
  }

  return {
    mode,
    results,
    agent: total.agent,
    judge: total.judge,
    datasetRun: langfuse ? runName : undefined,
  };
}

console.log(
  `Golden set: ${cases.length} questions · modes: ${modes.join(", ")} · ${
    langfuse ? "Langfuse on" : "Langfuse off (no keys)"
  }`,
);
if (langfuse) await syncDataset(langfuse);

for (const mode of modes) {
  console.log(`\nRunning ${mode}…`);
  summaries.push(await runMode(mode));
}

if (langfuse) await langfuse.flush();
await flushTraces();

// ---- Report ----
const lines: string[] = [];
lines.push(
  "| Setup | Passed | Pass rate | Avg accuracy | Avg policy | Agent cost | Avg time per answer | Handoffs |",
);
lines.push("|---|---|---|---|---|---|---|---|");
let failedGate = false;
for (const s of summaries) {
  const n = s.results.length;
  const passed = s.results.filter((r) => r.passed).length;
  const judged = s.results.filter((r) => r.verdict);
  const avg = (pick: (r: CaseResult) => number) =>
    judged.length
      ? (judged.reduce((a, r) => a + pick(r), 0) / judged.length).toFixed(2)
      : "n/a";
  lines.push(
    `| ${s.mode} | ${passed}/${n} | ${pct(passed, n)}% | ${avg((r) => r.verdict!.accuracy)} | ${avg((r) => r.verdict!.policy)} | $${s.agent.cost.toFixed(4)} | ${n ? (s.agent.latencyMs / n / 1000).toFixed(1) : "0"}s | ${s.results.reduce((a, r) => a + r.handoffs, 0)} |`,
  );
  if (gateArg === s.mode && n > 0 && passed / n < minPassRate)
    failedGate = true;
}
console.log(`\n${lines.join("\n")}\n`);

for (const s of summaries) {
  const failures = s.results.filter((r) => !r.passed);
  if (!failures.length) continue;
  console.log(`Failures (${s.mode}):`);
  for (const r of failures) {
    const why = [
      ...r.checks.failures,
      r.verdict && (r.verdict.accuracy < 4 || r.verdict.policy < 4)
        ? `judge ${r.verdict.accuracy}/${r.verdict.policy}: ${r.verdict.reasoning}`
        : "",
      r.judgeError ? `judge error: ${r.judgeError}` : "",
    ].filter(Boolean);
    console.log(`  - ${r.id}: ${why.join(" | ")}`);
  }
}

mkdirSync("evals/results", { recursive: true });
writeFileSync(
  `evals/results/${stamp}.json`,
  JSON.stringify({ stamp, minPassRate, gate: gateArg, summaries }, null, 2),
);
if (process.env.GITHUB_STEP_SUMMARY) {
  appendFileSync(
    process.env.GITHUB_STEP_SUMMARY,
    `## Golden-set eval\n\n${lines.join("\n")}\n\nGate: ${gateArg} must pass at least ${Math.round(minPassRate * 100)}%.\n`,
  );
}

if (failedGate) {
  console.error(
    `\nEval failed: ${gateArg} passed fewer than ${Math.round(minPassRate * 100)}% of the golden set.`,
  );
  process.exit(1);
}
process.exit(0);
