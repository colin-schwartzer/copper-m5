// The golden set (Module 5): questions Ask Copper must answer well, each with
// the facts a good answer contains. Kept in evals/golden-set.json so it can be
// read by a person, by the eval script and by Langfuse (as a dataset).
import data from "@/evals/golden-set.json";
import type { ChatMessage } from "@/lib/copper/assistant/run";
import type { Agent } from "@/lib/copper/assistant/progress";
import type { Desk } from "@/lib/copper/orders";

/** Rules a reply must follow, checked by code before the AI judge reads it. */
export type Checks = {
  /** At least one of these tools was called. */
  toolsAny?: string[];
  /** None of these tools was called. */
  toolsNone?: string[];
  /** Each regular expression (case-insensitive) matches the reply. */
  mustMatch?: string[];
  /** No regular expression (case-insensitive) matches the reply. */
  mustNotMatch?: string[];
};

export type GoldenCase = {
  id: string;
  category: string;
  messages: ChatMessage[];
  /** Who answers first (default Copper). */
  startAgent?: Agent;
  /** The Refund Console at the start (default: the golden set's desk). */
  desk?: Desk;
  /** What a good answer says and does, written for the AI judge. */
  expected: string;
  checks: Checks;
};

export type GoldenSet = {
  name: string;
  description: string;
  desk: Desk;
  cases: GoldenCase[];
};

export const GOLDEN_SET = data as GoldenSet;

export type ToolCall = { name: string; input?: unknown; output?: unknown };

export type CheckResult = { passed: boolean; failures: string[] };

/** Runs the rule checks against one answer. Pure, so it is easy to test. */
export function runChecks(
  checks: Checks,
  reply: string,
  toolCalls: ToolCall[],
): CheckResult {
  const failures: string[] = [];
  const called = new Set(toolCalls.map((call) => call.name));
  if (checks.toolsAny && !checks.toolsAny.some((name) => called.has(name)))
    failures.push(`expected a call to one of: ${checks.toolsAny.join(", ")}`);
  for (const name of checks.toolsNone ?? [])
    if (called.has(name)) failures.push(`must not call ${name}`);
  for (const pattern of checks.mustMatch ?? [])
    if (!new RegExp(pattern, "i").test(reply))
      failures.push(`reply should match /${pattern}/`);
  for (const pattern of checks.mustNotMatch ?? [])
    if (new RegExp(pattern, "i").test(reply))
      failures.push(`reply must not match /${pattern}/`);
  return { passed: failures.length === 0, failures };
}
