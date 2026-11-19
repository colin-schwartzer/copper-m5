// Runs one golden-set case through Ask Copper (the same engine the chat uses),
// then grades the answer with the rule checks and the AI judge.
import { startAgent, startTurn } from "@/lib/ai/langfuse";
import { finalValue } from "@/lib/ai/live";
import { toolCallsOf } from "@/lib/copper/assistant/progress";
import { runTurn, type SupportMode } from "@/lib/copper/assistant/run";
import {
  type CheckResult,
  GOLDEN_SET,
  type GoldenCase,
  runChecks,
  type ToolCall,
} from "@/lib/evals/golden";
import { JUDGE_PASS_SCORE, judgeAnswer, type Verdict } from "@/lib/evals/judge";

export type CaseResult = {
  id: string;
  category: string;
  mode: SupportMode;
  reply: string;
  toolCalls: ToolCall[];
  /** Which agents wrote the answer ("copper", "refund"). */
  handoffs: number;
  checks: CheckResult;
  verdict: Verdict | null;
  judgeError?: string;
  passed: boolean;
  /** Langfuse trace of the chat turn, when Langfuse is set up. */
  traceId?: string;
};

export async function runCase(
  testCase: GoldenCase,
  mode: SupportMode,
  traceName: string,
): Promise<CaseResult> {
  const { progress, traceId } = await runTurn({
    messages: testCase.messages,
    useTools: true,
    desk: testCase.desk ?? GOLDEN_SET.desk,
    agent: testCase.startAgent,
    mode,
    traceName,
  });
  const final = await finalValue(progress);
  const toolCalls = toolCallsOf(final);
  const handoffs = final.steps.filter((s) => s.type === "handoff").length;
  const checks = runChecks(testCase.checks, final.reply, toolCalls);

  let verdict: Verdict | null = null;
  let judgeError: string | undefined;
  try {
    // The judge gets its own trace so it doesn't pollute the chat's trace.
    const judgeTrace = startAgent(
      startTurn({ name: "eval-judge", input: testCase.id }),
      "judge",
      testCase.id,
    );
    verdict = await judgeAnswer(testCase, final.reply, toolCalls, judgeTrace);
    judgeTrace?.update({ output: verdict }).end();
  } catch (error) {
    judgeError = error instanceof Error ? error.message : String(error);
  }

  const passed =
    final.status === "done" &&
    checks.passed &&
    verdict !== null &&
    verdict.accuracy >= JUDGE_PASS_SCORE &&
    verdict.policy >= JUDGE_PASS_SCORE;
  return {
    id: testCase.id,
    category: testCase.category,
    mode,
    reply: final.reply,
    toolCalls,
    handoffs,
    checks,
    verdict,
    judgeError,
    passed,
    traceId,
  };
}
