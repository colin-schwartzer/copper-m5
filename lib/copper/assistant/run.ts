// The engine behind Ask Copper's server action (ask.ts), kept apart so the
// evals can run the same turns without the server. It streams the conversation to
// the model through callStream (lib/ai/call.ts), with or without the tools,
// and sends back each step as it happens, so the panel can show Copper
// thinking and using its tools.
// Module 4: on the Support page the chat also sends the Refund Console's
// "desk" (the auto-refund limit and the refunds so far) and which agent is
// talking. Copper can hand the conversation to the Refund agent, a second
// agent with its own instructions and tools, and the Refund agent can hand it
// back. The agent that ends the turn answers the next message too.
// Module 5: every turn is traced in Langfuse (lib/ai/langfuse.ts), and
// mode "single" swaps the two agents for one that does everything.
import { hasToolCall, stepCountIs } from "ai";
import { callStream, THOUGHT_SUMMARIES } from "@/lib/ai/call";
import {
  flushTraces,
  startAgent,
  startTool,
  startTurn,
  type TraceParent,
} from "@/lib/ai/langfuse";
import { createLive, type Live } from "@/lib/ai/live";
import {
  COPPER_HANDOFF,
  COPPER_INSTRUCTIONS,
  REFUND_AGENT_INSTRUCTIONS,
  SINGLE_AGENT_INSTRUCTIONS,
} from "@/lib/copper/assistant/instructions";
import type {
  Agent,
  Progress,
  ToolStep,
} from "@/lib/copper/assistant/progress";
import {
  copperTools,
  refundAgentTools,
  singleAgentTools,
  transferToRefundAgent,
} from "@/lib/copper/assistant/tools";
import type { Desk } from "@/lib/copper/orders";
import type { LangfuseTool } from "@langfuse/tracing";

/** Copper and the Refund agent (two agents), or one support agent (single). */
export type SupportMode = "two-agents" | "single";

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type AskCopperRequest = {
  messages: ChatMessage[];
  useTools: boolean;
  /** The Refund Console's limit and refunds (Support page only). */
  desk?: Desk;
  /** Who answers first: whoever ended the last turn. Copper by default. */
  agent?: Agent;
  /** Module 5: the two agents (default) or one agent that does everything. */
  mode?: SupportMode;
  /** Groups the turns of one conversation in Langfuse. */
  sessionId?: string;
  /** Names this run in Langfuse, like "eval:golden-set". Chats leave it out. */
  traceName?: string;
};

export type AskCopperResult = {
  progress: Live<Progress>;
  /** The Langfuse trace of this turn, when Langfuse is set up. */
  traceId?: string;
};

/** At most this many model steps (tool calls plus the reply) per agent. */
const MAX_STEPS = 5;

/** Copper → Refund agent → Copper at most, so a turn can't bounce forever. */
const MAX_HANDOFFS = 2;

const HANDOFF_TOOLS: Record<string, Agent> = {
  transfer_to_refund_agent: "refund",
  transfer_to_copper: "copper",
};

// A handoff ends the agent's turn: the other agent answers.
const STOP = [
  stepCountIs(MAX_STEPS),
  hasToolCall("transfer_to_refund_agent", "transfer_to_copper"),
];

const SORRY = "Sorry, something went wrong on my end. Please try again.";

/** The single agent: Copper's tools and the refund tools, no handoff. */
function singleCall(messages: ChatMessage[], desk: Desk, trace?: TraceParent) {
  return callStream(
    { module: 5, feature: "support-agent", trace },
    {
      messages,
      providerOptions: THOUGHT_SUMMARIES,
      instructions: SINGLE_AGENT_INSTRUCTIONS,
      tools: singleAgentTools(desk),
      stopWhen: stepCountIs(MAX_STEPS),
    },
  );
}

/** One agent's call: its instructions and only its own tools. */
function agentCall(
  agent: Agent,
  messages: ChatMessage[],
  desk?: Desk,
  trace?: TraceParent,
) {
  if (agent === "refund" && desk)
    return callStream(
      { module: 4, feature: "refund-agent", trace },
      {
        messages,
        providerOptions: THOUGHT_SUMMARIES,
        instructions: REFUND_AGENT_INSTRUCTIONS,
        tools: refundAgentTools(desk),
        stopWhen: STOP,
      },
    );
  return callStream(
    { module: 2, feature: "ask-copper", trace },
    {
      messages,
      providerOptions: THOUGHT_SUMMARIES,
      instructions: desk
        ? `${COPPER_INSTRUCTIONS}\n\n${COPPER_HANDOFF}`
        : COPPER_INSTRUCTIONS,
      tools: desk
        ? { ...copperTools, transfer_to_refund_agent: transferToRefundAgent }
        : copperTools,
      stopWhen: STOP,
    },
  );
}

const SPAN_NAMES: Record<Agent, string> = {
  copper: "copper-agent",
  refund: "refund-agent",
};

/** Runs one chat turn and streams its progress. The server action calls this. */
export async function runTurn({
  messages,
  useTools,
  desk,
  agent = "copper",
  mode = "two-agents",
  sessionId,
  traceName = "ask-copper-turn",
}: AskCopperRequest): Promise<AskCopperResult> {
  const { live, update, done } = createLive<Progress>();
  const single = mode === "single" && Boolean(desk) && useTools;
  const first: Agent = desk && useTools && !single ? agent : "copper";
  // Langfuse (Module 5): one trace per turn, an agent observation for each
  // agent that answers, and a tool observation for each tool call.
  const lastUser = messages.findLast((m) => m.role === "user")?.content;
  const turn = startTurn({
    name: traceName,
    input: lastUser,
    sessionId,
    tags: [single ? "single-agent" : "two-agents"],
    metadata: { mode: single ? "single" : "two-agents", useTools },
  });
  const toolSpans = new Map<string, LangfuseTool | undefined>();
  const progress: Progress = {
    steps: [],
    reply: "",
    status: "working",
    agent: first,
  };
  // A fresh copy each time, so earlier updates never change underneath the
  // panel.
  const snapshot = (): Progress => ({
    ...progress,
    steps: progress.steps.map((step) => ({ ...step })),
  });
  // Tool call ids are per agent call, so each call's are kept apart.
  let call = 0;
  const toolStep = (id: string) =>
    progress.steps.find(
      (step): step is ToolStep =>
        step.type === "tool" && step.id === `${call}:${id}`,
    );

  /** Reads one agent's stream into progress. Returns who to hand off to. */
  async function follow(
    current: Agent,
    result: ReturnType<typeof callStream>,
    span?: TraceParent,
  ): Promise<Agent | undefined> {
    let handoff: Agent | undefined;
    for await (const part of result.fullStream) {
      switch (part.type) {
        case "reasoning-delta": {
          const last = progress.steps.at(-1);
          if (last?.type === "thinking" && last.agent === current)
            last.text += part.text;
          else
            progress.steps.push({
              type: "thinking",
              text: part.text,
              agent: current,
            });
          break;
        }
        case "tool-input-start":
          progress.steps.push({
            type: "tool",
            id: `${call}:${part.id}`,
            agent: current,
            name: part.toolName,
            state: "calling",
          });
          break;
        case "tool-call": {
          const step = toolStep(part.toolCallId);
          toolSpans.set(
            `${call}:${part.toolCallId}`,
            startTool(span, part.toolName, part.input),
          );
          if (step) step.input = part.input;
          else
            progress.steps.push({
              type: "tool",
              id: `${call}:${part.toolCallId}`,
              agent: current,
              name: part.toolName,
              state: "calling",
              input: part.input,
            });
          break;
        }
        case "tool-result": {
          const step = toolStep(part.toolCallId);
          toolSpans
            .get(`${call}:${part.toolCallId}`)
            ?.update({ output: part.output })
            .end();
          if (step) Object.assign(step, { output: part.output, state: "done" });
          if (part.toolName in HANDOFF_TOOLS)
            handoff = HANDOFF_TOOLS[part.toolName];
          break;
        }
        case "tool-error": {
          const step = toolStep(part.toolCallId);
          const message =
            part.error instanceof Error
              ? part.error.message
              : String(part.error);
          toolSpans
            .get(`${call}:${part.toolCallId}`)
            ?.update({
              output: message,
              level: "ERROR",
              statusMessage: message,
            })
            .end();
          if (step) Object.assign(step, { output: message, state: "error" });
          break;
        }
        case "text-delta":
          progress.reply += part.text;
          break;
        case "error":
          throw part.error;
        default:
          continue;
      }
      update(snapshot());
    }
    return handoff;
  }

  // Read the stream in the background, so the panel gets the first update now.
  void (async () => {
    try {
      if (!useTools) {
        // Without tools, send no instructions either: "always use your
        // tools" with no tools makes the model answer with nothing.
        const span = startAgent(turn, "copper-agent", lastUser);
        await follow(
          "copper",
          callStream(
            { module: 2, feature: "ask-copper", trace: span },
            { messages, providerOptions: THOUGHT_SUMMARIES },
          ),
          span,
        );
        span?.update({ output: progress.reply }).end();
      } else if (single && desk) {
        const span = startAgent(turn, "support-agent", lastUser);
        await follow("copper", singleCall(messages, desk, span), span);
        span?.update({ output: progress.reply }).end();
      } else {
        let current = first;
        for (let handoffs = 0; ; handoffs++) {
          const span = startAgent(turn, SPAN_NAMES[current], lastUser);
          const next = await follow(
            current,
            agentCall(current, messages, desk, span),
            span,
          );
          span
            ?.update({
              output: progress.reply,
              metadata: { handoffTo: next ?? "" },
            })
            .end();
          if (!next || next === current || handoffs >= MAX_HANDOFFS) break;
          progress.steps.push({ type: "handoff", from: current, to: next });
          progress.agent = current = next;
          progress.reply = "";
          call++;
          update(snapshot());
        }
      }
      progress.status = "done";
      if (!progress.reply.trim()) progress.reply = SORRY;
    } catch (error) {
      progress.status = "error";
      if (!progress.reply.trim()) progress.reply = SORRY;
      turn?.update({
        level: "ERROR",
        statusMessage: error instanceof Error ? error.message : String(error),
      });
    }
    // Send the traces before the turn ends, so a serverless function
    // doesn't stop with them still waiting.
    turn?.update({ output: progress.reply }).end();
    await flushTraces();
    done(snapshot());
  })();

  return { progress: live, traceId: turn?.traceId };
}
