import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type {
  ChatContentPart,
  ChatMessage,
  ChatRequest,
  ChatTool,
  ChatWebSearchTool,
  ChatToolChoice,
  ChatToolCall,
  ResponsesContentPart,
  ResponsesFunctionCallOutputItem,
  ResponsesInputItem,
  ResponsesMessageItem,
  ResponsesRequest,
  ResponsesTool,
  ResponsesToolChoice,
} from "./types.js";
import { log } from "../util/log.js";

// Material a stripped image so the agent can pass the path to ocr.py.
// - data: URL  → decode, write to <dropDir>/cache/images/<sha1>.<ext>, return path
// - http(s):   → return as-is (ocr.py accepts URLs directly)
// - other:     → null (can't materialize)
// `dropDir` is typically cfg.dataDir; falls back to os.tmpdir()/mimo2codex-images.
function materializeStrippedImage(imageUrl: string, dropDir?: string): string | null {
  try {
    if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
      return imageUrl;
    }
    if (!imageUrl.startsWith("data:")) return null;
    const m = /^data:([^;,]+)(?:;base64)?,(.*)$/s.exec(imageUrl);
    if (!m) return null;
    const mime = m[1] || "image/png";
    const isBase64 = imageUrl.startsWith(`data:${mime};base64,`) || /;base64,/.test(imageUrl);
    const payload = m[2];
    const bytes = isBase64
      ? Buffer.from(payload, "base64")
      : Buffer.from(decodeURIComponent(payload), "utf-8");
    const ext = mime.split("/")[1]?.split("+")[0] || "png";
    const hash = createHash("sha1").update(bytes).digest("hex").slice(0, 16);
    const base = dropDir && dropDir.length > 0 ? dropDir : join(tmpdir(), "mimo2codex");
    const dir = join(base, "cache", "images");
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    const filePath = join(dir, `${hash}.${ext}`);
    if (!existsSync(filePath)) writeFileSync(filePath, bytes);
    return filePath;
  } catch (e) {
    log.warn(`failed to materialize stripped image: ${(e as Error).message}`);
    return null;
  }
}

// Per MiMo docs (https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/multimodal-understanding/image-understanding),
// only `mimo-v2.5` and `mimo-v2-omni` (and *-omni* variants) accept image
// input. The other v2.5 variants (mimo-v2.5-pro, mimo-v2.5-pro[1m],
// mimo-v2-flash, …) return 404 "No endpoints found that support image input"
// when given image_url parts.
function modelSupportsImages(model: string): boolean {
  // Strip the optional context-window suffix like [1m]
  const base = model.replace(/\[[^\]]*\]$/, "").toLowerCase();
  if (base.includes("omni")) return true;
  if (base === "mimo-v2.5") return true;
  return false;
}

function partsToChatContent(
  parts: ResponsesContentPart[] | string,
  ctx: { model: string; supportsImages: boolean; imageDropDir?: string }
): string | ChatContentPart[] {
  if (typeof parts === "string") return parts;

  const out: ChatContentPart[] = [];
  const droppedRefs: string[] = [];
  let droppedCount = 0;
  for (const p of parts) {
    if (p.type === "input_text" || p.type === "output_text") {
      // Defensive: Codex/clients sometimes send malformed text parts where
      // `text` is missing or non-string. Coerce to "" and drop empties —
      // MiMo's parser rejects parts where `text` is missing/empty with
      // "Param Incorrect: `text` is not set".
      const text = typeof p.text === "string" ? p.text : "";
      if (text.length === 0) continue;
      out.push({ type: "text", text });
    } else if (p.type === "input_image") {
      if (ctx.supportsImages) {
        out.push({ type: "image_url", image_url: { url: p.image_url, detail: p.detail } });
      } else {
        droppedCount++;
        const ref = materializeStrippedImage(p.image_url, ctx.imageDropDir);
        if (ref) droppedRefs.push(ref);
      }
    } else if (p.type === "input_file") {
      // MiMo doesn't natively support file inputs in chat completions.
      // Drop the part but leave the message intact.
      log.warn("dropped input_file part — MiMo chat API does not accept file inputs");
    }
    // Unknown part types (e.g. summary_text in some Responses variants) are
    // silently skipped — they'd cause MiMo to 400 if forwarded as-is.
  }

  if (droppedCount > 0) {
    log.warn(
      `dropped ${droppedCount} image part(s) — model "${ctx.model}" does not support image input (use mimo-v2.5 or mimo-v2-omni for vision); materialized ${droppedRefs.length} to disk`
    );
    // Tell the agent BOTH that images were stripped AND where to find them
    // so it can OCR without asking the user for a path. Codex / DS / Qwen
    // etc. read this and route to mimoskill/scripts/ocr.py per AGENTS.md.
    const refList = droppedRefs.length > 0
      ? droppedRefs.map((r, i) => `  ${i + 1}. ${r}`).join("\n")
      : "  (could not materialize image — unknown URL form)";
    const plural = droppedCount > 1 ? "s" : "";
    out.push({
      type: "text",
      text:
        `[${droppedCount} image attachment${plural} omitted because the active model can't ingest images.\n` +
        `The proxy materialized them to disk so you can OCR / describe without asking the user for a path:\n` +
        refList +
        `\nTo extract text or describe, run:\n` +
        `  python3 mimoskill/scripts/ocr.py <path-from-above>\n` +
        `Engine auto-select (zero-key): mimo (if MIMO_API_KEY set) > tesseract (if installed, --mode text) > pollinations.\n` +
        `If pollinations is unreachable (e.g. mainland China), install tesseract once for offline OCR:\n` +
        `  brew install tesseract tesseract-lang  /  apt install tesseract-ocr tesseract-ocr-chi-sim\n` +
        `Or switch the chat model to mimo-v2.5 / mimo-v2-omni to see images directly.]`,
    });
  }

  // MiMo's image-understanding API REQUIRES at least one `text` part in the
  // content array whenever `image_url` parts are present. Otherwise it 400s
  // with "Param Incorrect: `text` is not set" (yes, even though OpenAI's
  // chat API doesn't require this). Codex sometimes sends image-only user
  // messages (paste-and-send), so we add an empty fallback to satisfy the
  // schema. The image alone is usually enough context for the model.
  const hasImage = out.some((p) => p.type === "image_url");
  const hasText = out.some((p) => p.type === "text");
  if (hasImage && !hasText) {
    out.push({ type: "text", text: " " });
  }

  // If the message is purely text, collapse to a string for cleaner upstream payloads.
  if (out.length > 0 && out.every((p) => p.type === "text")) {
    return out.map((p) => (p as { text: string }).text).join("");
  }
  if (out.length === 0) return "";
  return out;
}

// Flatten a function_call_output.output into a plain string suitable for the
// `tool` role's `content` field. Codex/Responses can send an array of content
// parts here when a tool returned images (image_gen, mimoskill image gen,
// etc.). Chat Completions tool messages only accept a string content across
// all upstreams (OpenAI's array extension isn't supported by MiMo / DeepSeek /
// most third-party providers, and DeepSeek explicitly 400s with
// "unknown variant `input_image`, expected `text`"). So we always textify and
// replace images with a one-line placeholder.
function toolOutputToString(output: ResponsesFunctionCallOutputItem["output"]): string {
  if (typeof output === "string") return output;
  if (!Array.isArray(output)) {
    try {
      return JSON.stringify(output);
    } catch {
      return String(output);
    }
  }
  const chunks: string[] = [];
  let droppedImages = 0;
  for (const p of output) {
    if (!p || typeof p !== "object") continue;
    if (p.type === "input_text" || p.type === "output_text") {
      const text = typeof p.text === "string" ? p.text : "";
      if (text.length > 0) chunks.push(text);
    } else if (p.type === "input_image") {
      droppedImages++;
    }
    // input_file and unknown parts are silently dropped.
  }
  if (droppedImages > 0) {
    log.warn(
      `dropped ${droppedImages} image part(s) from tool output — Chat Completions tool messages cannot carry images`
    );
    chunks.push(
      `[${droppedImages} image attachment${droppedImages > 1 ? "s" : ""} omitted from tool output: this chat backend cannot ingest images in tool results.]`
    );
  }
  return chunks.join("");
}

function messageItemToChat(
  item: ResponsesMessageItem,
  ctx: { model: string; supportsImages: boolean; imageDropDir?: string }
): ChatMessage {
  const role = item.role === "developer" ? "system" : item.role;
  const content = partsToChatContent(item.content, ctx);
  if (role === "assistant") {
    return { role: "assistant", content: typeof content === "string" ? content : "" };
  }
  return { role, content };
}

// Schema for Codex's `local_shell` builtin tool, mapped to a regular function
// tool that MiMo (and any chat-completions-only provider) can understand.
// Codex registers handlers for both `local_shell` (builtin) and `shell`
// (function), so emitting `shell` tool_calls back to it just works.
const LOCAL_SHELL_FN: ChatTool = {
  type: "function",
  function: {
    name: "shell",
    description:
      "Execute a shell command on the local machine. Returns stdout, stderr and exit code.",
    parameters: {
      type: "object",
      properties: {
        command: {
          type: "array",
          items: { type: "string" },
          description:
            "Argv array, e.g. [\"ls\", \"-la\"]. The first element is the program; remaining elements are arguments.",
        },
        workdir: {
          type: "string",
          description: "Working directory to run the command in (optional).",
        },
        timeout_ms: {
          type: "number",
          description: "Timeout in milliseconds (optional, default 30000).",
        },
      },
      required: ["command"],
    },
    strict: null,
  },
};

// Tools that exist server-side at OpenAI but have no equivalent at MiMo.
// We drop them silently (only log at debug level) — there's nothing a chat
// completions provider can do with them. NOTE: web_search and
// web_search_preview are NOT in this list — MiMo has its own native
// web_search builtin (requires the Web Search Plugin to be activated in
// the MiMo console: https://platform.xiaomimimo.com/#/console/plugin).
const SERVER_SIDE_TOOLS = new Set([
  "code_interpreter",
  "file_search",
  "image_generation",
  "computer_use_preview",
  "computer_use",
]);

// Track tool types we've already warned about so we don't spam the log on
// every request (Codex re-sends the full tool list each turn).
const warnedTypes = new Set<string>();
function warnOnce(toolType: string, msg: string): void {
  if (warnedTypes.has(toolType)) return;
  warnedTypes.add(toolType);
  log.warn(msg);
}

export interface ReqToChatOpts {
  // Add `thinking: {type: "disabled"}` to the chat body — MiMo skips its
  // reasoning mode, which makes the model less likely to "narrate" without
  // calling tools. Useful for agentic / coding workflows.
  disableThinking?: boolean;
  // Override Codex's `parallel_tool_calls: false` to true. Codex defaults to
  // serial tool calls (one per round-trip); for MiMo this can cause the model
  // to give up after 3-4 explore rounds without ever calling apply_patch.
  forceParallelToolCalls?: boolean;
  // Translate Codex's web_search/web_search_preview to MiMo's web_search
  // builtin. Default OFF — MiMo's Web Search Plugin is separately billed and
  // returns 400 "webSearchEnabled is false" if not activated.
  enableWebSearch?: boolean;
  // Directory where stripped images are materialized for non-vision models.
  // Typically cfg.dataDir; falls back to os.tmpdir()/mimo2codex when unset.
  // Lets the agent OCR pasted images without asking the user for a path.
  imageDropDir?: string;
}

// Returns one or more ChatTools (a `namespace` wrapper can expand to many),
// or null if the tool is unrecognized / unsupported.
function toolToChat(t: ResponsesTool, opts: ReqToChatOpts): ChatTool | ChatTool[] | null {
  // 1. Standard OpenAI function tool — pass through.
  if (t.type === "function") {
    const ft = t as { type: "function"; name?: string; description?: string; parameters?: Record<string, unknown>; strict?: boolean | null };
    if (!ft.name) {
      log.debug("dropping function tool with no name");
      return null;
    }
    return {
      type: "function",
      function: {
        name: ft.name,
        description: ft.description,
        parameters: ft.parameters,
        strict: ft.strict ?? null,
      },
    };
  }

  // 2. Codex's `local_shell` builtin → emit as a regular `shell` function tool
  //    with the canonical schema (see LOCAL_SHELL_FN above). Codex's tool router
  //    accepts both names.
  if (t.type === "local_shell") {
    return LOCAL_SHELL_FN;
  }

  // 2.5. OpenAI's `web_search` / `web_search_preview` → MiMo's native `web_search`.
  //      Forwarded by default; the server passes enableWebSearch: true. The
  //      option is kept for tests and for any future "off switch". If MiMo
  //      rejects with `webSearchEnabled is false` (plugin not activated for
  //      that account), mimoClient surfaces a clear error to the user — we do
  //      NOT silently retry, so users see the real configuration issue.
  if (t.type === "web_search" || t.type === "web_search_preview") {
    if (!opts.enableWebSearch) {
      return null;
    }
    const w = t as {
      user_location?: ChatWebSearchTool["user_location"];
      max_keyword?: number;
      force_search?: boolean;
      limit?: number;
    };
    const tool: ChatWebSearchTool = { type: "web_search" };
    if (w.user_location) tool.user_location = w.user_location;
    if (typeof w.max_keyword === "number") tool.max_keyword = w.max_keyword;
    if (typeof w.force_search === "boolean") tool.force_search = w.force_search;
    if (typeof w.limit === "number") tool.limit = w.limit;
    return tool;
  }

  // 3. Codex / OpenAI `custom` tool — freeform tool, often used for grammar-
  //    constrained outputs. We can't enforce the grammar at MiMo, but we can
  //    forward the name + description as a parameter-less function so the
  //    model can still call it. Format here:
  //      { type: "custom", name, description?, format?: { type: "grammar" | "text", ... } }
  if (t.type === "custom") {
    const ct = t as { name?: string; description?: string; format?: { type?: string } };
    if (!ct.name) {
      log.debug("dropping custom tool with no name");
      return null;
    }
    const formatType = ct.format?.type;
    const desc =
      (ct.description ?? "") +
      (formatType
        ? ` (originally a "${formatType}"-format custom tool; output should follow that format).`
        : "");
    return {
      type: "function",
      function: {
        name: ct.name,
        description: desc.trim() || undefined,
        // Permissive schema since we don't know the original input shape.
        parameters: {
          type: "object",
          properties: {
            input: {
              type: "string",
              description: "Input text for the tool.",
            },
          },
          additionalProperties: true,
        },
        strict: null,
      },
    };
  }

  // 4. `namespace` wrapper — Codex bundles MCP / grouped tools under this. Shape
  //    we've seen in the wild:
  //       { type: "namespace", name?: string, tools?: Tool[] }
  //    Recurse into nested tools and flatten. If there's no nested array, drop.
  if (t.type === "namespace") {
    const ns = t as { name?: string; tools?: ResponsesTool[] };
    if (!Array.isArray(ns.tools) || ns.tools.length === 0) {
      log.debug(
        `dropping "namespace" tool ${ns.name ? `"${ns.name}"` : ""} with no nested tools`
      );
      return null;
    }
    const nested: ChatTool[] = [];
    for (const inner of ns.tools) {
      const r = toolToChat(inner, opts);
      if (Array.isArray(r)) nested.push(...r);
      else if (r) nested.push(r);
    }
    return nested.length > 0 ? nested : null;
  }

  // 5. Server-side tools that only OpenAI/Azure can fulfill — silently drop.
  if (SERVER_SIDE_TOOLS.has(t.type)) {
    log.debug(
      `dropping server-side tool "${t.type}" — no MiMo equivalent (only OpenAI/Azure can fulfill)`
    );
    return null;
  }

  // 6. Truly unknown — warn once per type so we get a heads-up but don't spam.
  warnOnce(
    t.type,
    `dropping unsupported tool type "${t.type}" — please open an issue if this should be translated`
  );
  return null;
}

function toolChoiceToChat(tc: ResponsesToolChoice | undefined): ChatToolChoice | undefined {
  if (tc === undefined) return undefined;
  if (typeof tc === "string") return tc;
  if (tc.type === "function") {
    const name = tc.function?.name ?? tc.name;
    if (!name) return undefined;
    return { type: "function", function: { name } };
  }
  return undefined;
}

interface AssemblyState {
  pendingReasoning: string | null;
  pendingToolCalls: ChatToolCall[];
  pendingAssistantText: string | null;
  // Messages deferred during tool execution. Codex can inject developer/user
  // messages (e.g. permission approvals) between function_call and
  // function_call_output. Emitting them immediately would break the OpenAI
  // API requirement that tool messages must directly follow the assistant
  // message with tool_calls. We hold them here and flush after all tool
  // outputs arrive.
  deferredMessages: ChatMessage[];
  // Number of function_calls whose output hasn't arrived yet. Used to
  // decide when it's safe to flush deferredMessages without breaking the
  // assistant(tool_calls)→tool(s) adjacency requirement.
  outstandingToolCalls: number;
}

function flushDeferred(messages: ChatMessage[], state: AssemblyState): void {
  if (state.deferredMessages.length === 0) return;
  messages.push(...state.deferredMessages);
  state.deferredMessages = [];
}

function flushAssistant(messages: ChatMessage[], state: AssemblyState): void {
  const hasReasoning = state.pendingReasoning !== null;
  const hasTools = state.pendingToolCalls.length > 0;
  const hasText = state.pendingAssistantText !== null;
  if (!hasReasoning && !hasTools && !hasText) return;

  const msg: ChatMessage = { role: "assistant", content: hasText ? state.pendingAssistantText : null };
  if (hasTools) msg.tool_calls = state.pendingToolCalls;
  if (hasReasoning) msg.reasoning_content = state.pendingReasoning;
  messages.push(msg);

  state.pendingReasoning = null;
  state.pendingToolCalls = [];
  state.pendingAssistantText = null;
}

function inputItemsToMessages(
  items: ResponsesInputItem[],
  ctx: { model: string; supportsImages: boolean; imageDropDir?: string }
): ChatMessage[] {
  const out: ChatMessage[] = [];
  const state: AssemblyState = {
    pendingReasoning: null,
    pendingToolCalls: [],
    pendingAssistantText: null,
    deferredMessages: [],
    outstandingToolCalls: 0,
  };

  for (const rawItem of items) {
    // Compatibility with Chat-Completions-shaped probes (cc-switch test
    // connection, raw OpenAI SDK requests sent at the wrong endpoint, etc.):
    // `{role, content: string}` with NO `type` field. Real Codex always sends
    // `{type: "message", role, content: [...]}`. Promote the simple shape to
    // a message item so we don't silently drop it and produce empty messages.
    let item: ResponsesInputItem = rawItem;
    if (item && typeof item === "object" && !("type" in item)) {
      const legacy = item as unknown as { role?: string; content?: unknown };
      if (typeof legacy.role === "string") {
        const text =
          typeof legacy.content === "string"
            ? legacy.content
            : Array.isArray(legacy.content)
              ? legacy.content
                  .map((p: unknown) => {
                    if (typeof p === "string") return p;
                    if (p && typeof p === "object" && "text" in p)
                      return String((p as { text: unknown }).text ?? "");
                    return "";
                  })
                  .join("")
              : "";
        item = {
          type: "message",
          role: legacy.role as "user" | "system" | "developer" | "assistant",
          content: [
            { type: legacy.role === "assistant" ? "output_text" : "input_text", text },
          ],
        } as ResponsesInputItem;
      }
    }

    switch (item.type) {
      case "message": {
        if (item.role === "assistant") {
          // Buffer assistant text into pending, but DO NOT flush yet — any
          // following function_call items (same agent turn) need to merge into
          // ONE assistant message alongside the text and reasoning_content.
          // Splitting them would leave the tool-call message without reasoning,
          // which DeepSeek V4 thinking mode rejects with
          // "The reasoning_content in the thinking mode must be passed back".
          //
          // Edge case: two assistant messages back-to-back (rare). Flush the
          // previous one first so we don't drop its text.
          if (state.pendingAssistantText !== null) {
            flushAssistant(out, state);
          }
          const content = partsToChatContent(item.content, ctx);
          state.pendingAssistantText =
            typeof content === "string" ? content : "";
        } else {
          // If tool calls are in-flight (Codex injected this message
          // between function_call and function_call_output), defer it so
          // the OpenAI-required assistant(tool_calls)→tool ordering is
          // preserved. Otherwise emit normally.
          if (state.outstandingToolCalls > 0) {
            state.deferredMessages.push(messageItemToChat(item, ctx));
          } else {
            flushAssistant(out, state);
            out.push(messageItemToChat(item, ctx));
          }
        }
        break;
      }
      case "reasoning": {
        flushAssistant(out, state);
        // Prefer `encrypted_content` — that's where respToResponses /
        // streamToSse pin the FULL reasoning trace (Codex echoes it
        // back verbatim across turns, summary may be empty under
        // --no-reasoning). Fall back to summary text for compatibility
        // with reasoning items emitted by other clients / older
        // mimo2codex versions.
        let text = "";
        if (typeof item.encrypted_content === "string" && item.encrypted_content.length > 0) {
          text = item.encrypted_content;
        } else {
          text = item.summary
            .filter((s) => s.type === "summary_text")
            .map((s) => s.text)
            .join("");
        }
        state.pendingReasoning = text;
        break;
      }
      case "function_call": {
        state.pendingToolCalls.push({
          id: item.call_id,
          type: "function",
          function: { name: item.name, arguments: item.arguments },
        });
        state.outstandingToolCalls++;
        break;
      }
      case "function_call_output": {
        flushAssistant(out, state);
        out.push({
          role: "tool",
          tool_call_id: item.call_id,
          content: toolOutputToString(item.output),
        });
        state.outstandingToolCalls--;
        // Only release deferred messages after the LAST tool output —
        // all tool messages must be consecutive after the assistant
        // message with tool_calls.
        if (state.outstandingToolCalls === 0) {
          flushDeferred(out, state);
        }
        break;
      }
    }
  }
  flushAssistant(out, state);
  flushDeferred(out, state);
  return out;
}

export function reqToChat(req: ResponsesRequest, opts: ReqToChatOpts = {}): ChatRequest {
  const messages: ChatMessage[] = [];
  const ctx = {
    model: req.model,
    supportsImages: modelSupportsImages(req.model),
    imageDropDir: opts.imageDropDir,
  };

  if (req.instructions) {
    messages.push({ role: "system", content: req.instructions });
  }

  if (typeof req.input === "string") {
    messages.push({ role: "user", content: req.input });
  } else if (Array.isArray(req.input)) {
    for (const m of inputItemsToMessages(req.input, ctx)) {
      messages.push(m);
    }
  }

  const chat: ChatRequest = {
    model: req.model,
    messages,
    stream: req.stream ?? false,
  };
  if (chat.stream) {
    chat.stream_options = { include_usage: true };
  }

  if (req.tools && req.tools.length > 0) {
    const mapped: ChatTool[] = [];
    for (const t of req.tools) {
      const r = toolToChat(t, opts);
      if (Array.isArray(r)) mapped.push(...r);
      else if (r) mapped.push(r);
    }
    if (mapped.length > 0) chat.tools = mapped;
  }
  const tc = toolChoiceToChat(req.tool_choice);
  if (tc !== undefined) chat.tool_choice = tc;

  // parallel_tool_calls: --force-parallel-tool-calls overrides Codex's value.
  if (opts.forceParallelToolCalls) {
    chat.parallel_tool_calls = true;
  } else if (req.parallel_tool_calls !== undefined) {
    chat.parallel_tool_calls = req.parallel_tool_calls;
  }
  if (req.temperature !== undefined && req.temperature !== null) {
    chat.temperature = req.temperature;
  }
  if (req.top_p !== undefined && req.top_p !== null) {
    chat.top_p = req.top_p;
  }
  if (req.max_output_tokens !== undefined && req.max_output_tokens !== null) {
    chat.max_completion_tokens = req.max_output_tokens;
  }

  // --disable-thinking: tells MiMo to skip its reasoning mode. Helps when the
  // model would otherwise spend tokens narrating ("I'll do X") and end the
  // turn without ever calling a tool.
  if (opts.disableThinking) {
    chat.thinking = { type: "disabled" };
  }

  return chat;
}
