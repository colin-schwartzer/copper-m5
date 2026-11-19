"use server";

// The server action the Ask Copper chat calls. The work happens in run.ts, so
// the evals can run the same turns without the server.
import {
  type AskCopperRequest,
  type AskCopperResult,
  runTurn,
} from "@/lib/copper/assistant/run";

export type {
  AskCopperRequest,
  AskCopperResult,
  ChatMessage,
} from "@/lib/copper/assistant/run";

export async function askCopper(
  request: AskCopperRequest,
): Promise<AskCopperResult> {
  return runTurn(request);
}
