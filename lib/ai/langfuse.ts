// Langfuse tracing (Module 5, observability). Every chat turn becomes one
// trace: a root "turn", one "agent" observation per agent that answered, and
// under each agent its model calls ("generation": model, tokens, cost, time)
// and its tool calls ("tool"). Open Langfuse > Tracing to see them.
//
// Tracing is optional, like every service in this app: with no Langfuse keys
// every function here does nothing, so the app and the tests run without them.
import { LangfuseSpanProcessor } from "@langfuse/otel";
import {
  type LangfuseAgent,
  type LangfuseGeneration,
  type LangfuseSpan,
  type LangfuseTool,
  propagateAttributes,
  setLangfuseTracerProvider,
  startObservation,
} from "@langfuse/tracing";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

/** Anything a model call or tool call can hang under in a trace. */
export type TraceParent = LangfuseSpan | LangfuseAgent;

export function isLangfuseConfigured(): boolean {
  return Boolean(
    process.env.LANGFUSE_PUBLIC_KEY?.trim() &&
    process.env.LANGFUSE_SECRET_KEY?.trim(),
  );
}

let processor: LangfuseSpanProcessor | undefined;

/** Starts the exporter once. Its own tracer provider, so it never fights Next.js's. */
function ensureTracing(): boolean {
  if (!isLangfuseConfigured()) return false;
  if (!processor) {
    processor = new LangfuseSpanProcessor({
      publicKey: process.env.LANGFUSE_PUBLIC_KEY?.trim(),
      secretKey: process.env.LANGFUSE_SECRET_KEY?.trim(),
      baseUrl: process.env.LANGFUSE_BASE_URL?.trim() || undefined,
      environment:
        process.env.LANGFUSE_TRACING_ENVIRONMENT?.trim() || undefined,
      // Export every observation we create, and nothing else.
      shouldExportSpan: () => true,
    });
    setLangfuseTracerProvider(
      new NodeTracerProvider({ spanProcessors: [processor] }),
    );
  }
  return true;
}

/** Turns tracing on now (scripts do this before they start their own traces). */
export function enableTracing(): boolean {
  return ensureTracing();
}

/** Sends any traces still waiting. Call at the end of a turn or a script. */
export async function flushTraces(): Promise<void> {
  if (!processor) return;
  try {
    await processor.forceFlush();
  } catch {
    // Tracing must never break a chat.
  }
}

export type TurnInfo = {
  /** Trace name, like "ask-copper-turn". */
  name: string;
  input: unknown;
  /** Groups the turns of one conversation. */
  sessionId?: string;
  tags?: string[];
  metadata?: Record<string, string | number | boolean>;
};

/**
 * Starts the trace for one turn. Returns undefined when Langfuse isn't set up,
 * so callers just pass `trace?.…` along.
 */
export function startTurn(info: TurnInfo): LangfuseSpan | undefined {
  if (!ensureTracing()) return undefined;
  try {
    return propagateAttributes(
      {
        traceName: info.name,
        sessionId: info.sessionId,
        tags: info.tags,
        metadata: info.metadata
          ? Object.fromEntries(
              Object.entries(info.metadata).map(([k, v]) => [k, String(v)]),
            )
          : undefined,
      },
      () =>
        startObservation(info.name, { input: info.input }, { asType: "span" }),
    );
  } catch {
    return undefined;
  }
}

/** One agent answering inside a turn. */
export function startAgent(
  parent: LangfuseSpan | undefined,
  name: string,
  input: unknown,
): LangfuseAgent | undefined {
  return parent?.startObservation(name, { input }, { asType: "agent" });
}

/** One tool call inside an agent. */
export function startTool(
  parent: TraceParent | undefined,
  name: string,
  input: unknown,
): LangfuseTool | undefined {
  return parent?.startObservation(name, { input }, { asType: "tool" });
}

export type GenerationInfo = {
  name: string;
  model: string;
  input: unknown;
  metadata?: Record<string, string | number | boolean>;
};

/**
 * One model call. Under `parent` when given, otherwise a trace of its own
 * (an eval run, the judge, the embedding of a search).
 */
export function startGeneration(
  parent: TraceParent | undefined,
  info: GenerationInfo,
): LangfuseGeneration | undefined {
  const attributes = {
    model: info.model,
    input: info.input,
    metadata: info.metadata,
  };
  if (parent) {
    return parent.startObservation(info.name, attributes, {
      asType: "generation",
    });
  }
  if (!ensureTracing()) return undefined;
  try {
    return startObservation(info.name, attributes, { asType: "generation" });
  } catch {
    return undefined;
  }
}
