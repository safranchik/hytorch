#!/usr/bin/env node
// hytorch-pi.mjs runs one hytorch node's agent instance to completion via Pi
// (@earendil-works/pi-coding-agent) and prints only the agent's final
// assistant response and persisted session identity as JSON. hytorch.PiHarness
// spawns this script for both the forward turn and its resumed optimizer turn.
//
// Pi is configured here to call OpenAI models exclusively: by default
// through the operator's ChatGPT Plus/Pro (Codex) subscription login
// (Pi's "openai-codex" provider, authenticated once via `pi` + `/login` ->
// "ChatGPT Plus/Pro (Codex)", stored in ~/.pi/agent/auth.json and
// auto-refreshed), or via a plain OPENAI_API_KEY against Pi's "openai"
// provider. No Anthropic model or key is ever selected here.
//
// Usage:
//   node hytorch-pi.mjs --cwd <dir> --prompt-file <path> [--provider <id>] [--model <id>]
//       --session-dir <dir> [--session-file <path>]
//       [--temperature <number>] [--max-tokens <positive integer>]
import { readFile } from "node:fs/promises";
import { relative } from "node:path";
import {
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";

function parseArgs(argv) {
	const out = {};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg.startsWith("--")) {
			out[arg.slice(2)] = argv[i + 1];
			i++;
		}
	}
	return out;
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const cwd = args.cwd;
	const promptFile = args["prompt-file"];
	const provider = args.provider || "openai-codex";
	const modelId = args.model || "";
	const sessionDir = args["session-dir"];
	const sessionFile = args["session-file"];
	const temperature = parseNonNegativeNumber(args.temperature, "temperature");
	const maxTokens = parsePositiveInteger(args["max-tokens"], "max-tokens");

	if (!cwd || !promptFile || !sessionDir) {
		console.error("usage: hytorch-pi.mjs --cwd <dir> --prompt-file <path> --session-dir <dir> [--session-file <path>] [--provider <id>] [--model <id>]");
		process.exit(2);
	}

	const prompt = await readFile(promptFile, "utf8");

	// Permit Pi's model runtime to refresh its catalog so newly released
	// OpenAI/Codex models do not require a HyTorch release merely to be selected.
	const modelRuntime = await ModelRuntime.create({ allowModelNetwork: true });

	let model;
	if (modelId) {
		model = modelRuntime.getModel(provider, modelId);
		if (!model) {
			console.error(`hytorch-pi: no model "${modelId}" registered for provider "${provider}"`);
			process.exit(1);
		}
		const available = await modelRuntime.getAvailable();
		if (!available.some((m) => m.provider === provider && m.id === modelId)) {
			console.error(
				`hytorch-pi: model "${provider}/${modelId}" has no configured auth. ` +
					`Run 'pi' once and '/login' -> "ChatGPT Plus/Pro (Codex)" (provider ` +
					`"openai-codex"), or set OPENAI_API_KEY (provider "openai").`,
			);
			process.exit(1);
		}
	} else {
		const available = await modelRuntime.getAvailable();
		model = available.find((m) => m.provider === provider);
		if (!model) {
			console.error(
				`hytorch-pi: provider "${provider}" has no configured auth. ` +
					`Run 'pi' once and '/login' -> "ChatGPT Plus/Pro (Codex)" (provider ` +
					`"openai-codex"), or set OPENAI_API_KEY (provider "openai").`,
			);
			process.exit(1);
		}
	}

	const agentDir = getAgentDir();
	const settingsManager = SettingsManager.create(cwd, agentDir);
	const resourceLoader = new DefaultResourceLoader({
		cwd,
		agentDir,
		settingsManager,
		// This is Pi's documented context extension hook. The note is
		// ephemeral (not persisted in the transcript) and is regenerated
		// before every LLM request, including after each tool action.
		extensionFactories: maxTokens === undefined ? [] : [budgetExtension(maxTokens)],
	});
	await resourceLoader.reload();

	const sessionManager = sessionFile
		? SessionManager.open(sessionFile, sessionDir)
		: SessionManager.create(cwd, sessionDir);
	const { session } = await createAgentSession({
		cwd,
		model,
		modelRuntime,
		settingsManager,
		resourceLoader,
		sessionManager,
		tools: ["read", "bash", "edit", "write"],
	});
	const transportTemperature = model.api === "openai-codex-responses" ? undefined : temperature;

	// Pi's SDK exposes a per-request transport hook. Compatible model APIs
	// receive the sampling temperature. The Codex Responses API rejects that
	// field, so DFM conveys its semantic mutation scale in the backward prompt.
	// The remaining total budget is enforced across tool-use turns.
	if (transportTemperature !== undefined || maxTokens !== undefined) {
		const stream = session.agent.streamFunction;
		let outputTokensUsed = 0;
		session.subscribe((event) => {
			if (event.type === "message_end" && event.message?.role === "assistant") {
				outputTokensUsed += event.message.usage?.output || 0;
			}
		});
			session.agent.streamFunction = async (requestModel, context, options) => {
			const remaining = maxTokens === undefined ? undefined : maxTokens - outputTokensUsed;
			if (remaining !== undefined && remaining <= 0) {
				throw new Error("DFM token budget exhausted before the workspace mutation completed");
			}
			return stream(requestModel, context, {
				...options,
				...(transportTemperature === undefined ? {} : { temperature: transportTemperature }),
				...(remaining === undefined ? {} : { maxTokens: remaining }),
			});
		};
	}

	// Each assistant message_end carries that message's full
	// text content; the last one observed before the session settles is
	// the node's handoff summary (earlier assistant text, if any,
	// preceded further tool calls and isn't the final response).
	let finalText = "";
	const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
	const unsubscribe = session.subscribe((event) => {
		if (event.type === "message_end" && event.message?.role === "assistant") {
			usage.input += event.message.usage?.input || 0;
			usage.output += event.message.usage?.output || 0;
			usage.cacheRead += event.message.usage?.cacheRead || 0;
			usage.cacheWrite += event.message.usage?.cacheWrite || 0;
			const text = (event.message.content || [])
				.filter((block) => block.type === "text")
				.map((block) => block.text)
				.join("");
			if (text) {
				finalText = text;
			}
		}
	});

	try {
		await session.prompt(prompt);
	} finally {
		unsubscribe();
		session.dispose();
	}

	if (!finalText) {
		console.error("hytorch-pi: agent produced no final text response");
		process.exit(1);
	}
	if (!session.sessionFile) {
		console.error("hytorch-pi: persisted session has no session file");
		process.exit(1);
	}
	process.stdout.write(JSON.stringify({
		text: finalText.trim(),
		session_id: session.sessionId,
		session_file: relative(sessionDir, session.sessionFile),
		usage,
	}));
}

function budgetExtension(maxTokens) {
	let outputTokensUsed = 0;
	return {
		name: "hytorch-budget",
		hidden: true,
		factory(pi) {
			pi.on("message_end", (event) => {
				if (event.message.role === "assistant") {
					outputTokensUsed += event.message.usage?.output || 0;
				}
			});
			pi.on("context", (event) => {
				const remaining = Math.max(0, maxTokens - outputTokensUsed);
				return {
					messages: [
						...event.messages,
						{
							role: "user",
							content: [{
								type: "text",
								text: `HyTorch mutation budget: ${remaining} output tokens remain. ` +
									"Complete the smallest useful workspace edit, validate it, and finish with a concise handoff before the budget is exhausted.",
							}],
							timestamp: Date.now(),
						},
					],
				};
			});
		},
	};
}

main().catch((err) => {
	console.error("hytorch-pi:", err?.stack || String(err));
	process.exit(1);
});

function parseNonNegativeNumber(value, name) {
	if (value === undefined) return undefined;
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || parsed < 0) {
		throw new Error(`--${name} must be a non-negative number`);
	}
	return parsed;
}

function parsePositiveInteger(value, name) {
	if (value === undefined) return undefined;
	const parsed = Number(value);
	if (!Number.isInteger(parsed) || parsed <= 0) {
		throw new Error(`--${name} must be a positive integer`);
	}
	return parsed;
}
