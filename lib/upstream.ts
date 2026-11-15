import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

export function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function object(value: unknown): Json {
  return isRecord(value) ? value : {};
}

function args(value: unknown): Json {
  if (typeof value !== 'string') return object(value);
  try {
    return object(JSON.parse(value));
  } catch {
    return {};
  }
}

function blocks(content: unknown): Json[] {
  return typeof content === 'string' ? [{ type: 'text', text: content }] : records(content);
}

function imageSource(url: string): Json {
  const inline = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  return inline ? { type: 'base64', media_type: inline[1], data: inline[2] } : { type: 'url', url };
}

export function chatToMessages(body: Json, model: string): Json {
  const system: Json[] = [];
  const messages: Json[] = [];
  for (const raw of records(body.messages)) {
    if (raw.role === 'system' || raw.role === 'developer') {
      system.push(...blocks(raw.content).filter((block) => block.type === 'text'));
      continue;
    }
    if (raw.role === 'tool') {
      const result = { type: 'tool_result', tool_use_id: raw.tool_call_id, content: raw.content };
      const previous = messages.at(-1);
      if (previous?.role === 'user' && Array.isArray(previous.content)) previous.content.push(result);
      else messages.push({ role: 'user', content: [result] });
      continue;
    }
    const content = blocks(raw.content).flatMap((block): Json[] => {
      if (block.type === 'text') return [{ type: 'text', text: block.text }];
      const url = object(block.image_url).url;
      return block.type === 'image_url' && typeof url === 'string' ? [{ type: 'image', source: imageSource(url) }] : [];
    });
    for (const call of records(raw.tool_calls)) {
      const fn = object(call.function);
      content.push({ type: 'tool_use', id: call.id, name: fn.name, input: args(fn.arguments) });
    }
    messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
  }
  const tools = records(body.tools).filter((tool) => tool.type === 'function').map((tool) => {
    const fn = object(tool.function);
    return { name: fn.name, description: fn.description, input_schema: fn.parameters ?? { type: 'object', properties: {} } };
  });
  const tool = object(object(body.tool_choice).function).name;
  const toolChoice = typeof tool === 'string' ? { type: 'tool', name: tool }
    : body.tool_choice === 'required' ? { type: 'any' }
      : body.tool_choice === 'none' ? { type: 'none' } : undefined;
  return {
    model, messages, stream: body.stream === true, max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 1024,
    system: system.length ? system : undefined, tools: tools.length ? tools : undefined,
    stop_sequences: typeof body.stop === 'string' ? [body.stop] : body.stop,
    temperature: body.temperature, top_p: body.top_p, tool_choice: toolChoice,
  };
}

export function messagesToGemini(body: Json, format: unknown): Json {
  const names = new Map<unknown, unknown>();
  const contents = records(body.messages).map((message) => ({
    role: message.role === 'assistant' ? 'model' : 'user',
    parts: blocks(message.content).flatMap((block): Json[] => {
      if (block.type === 'text') return [{ text: block.text }];
      if (block.type === 'image') {
        const source = object(block.source);
        return source.type === 'base64' ? [{ inlineData: { mimeType: source.media_type, data: source.data } }]
          : [{ fileData: { fileUri: source.url } }];
      }
      if (block.type === 'tool_use') {
        names.set(block.id, block.name);
        return [{ functionCall: { name: block.name, args: block.input } }];
      }
      if (block.type === 'tool_result') {
        const result = blocks(block.content).map((part) => part.text).join('\n');
        return [{ functionResponse: { name: names.get(block.tool_use_id), response: { result } } }];
      }
      return [];
    }),
  }));
  const system = blocks(body.system).map((block) => ({ text: block.text }));
  const declarations = records(body.tools).map((tool) => ({
    name: tool.name, description: tool.description, parametersJsonSchema: tool.input_schema,
  }));
  const choice = object(body.tool_choice);
  const choiceMode = choice.type === 'none' ? 'NONE' : choice.type === 'tool' || choice.type === 'any' ? 'ANY' : 'AUTO';
  const responseFormat = object(format);
  const jsonMode = responseFormat.type === 'json_object' || responseFormat.type === 'json_schema';
  return {
    contents,
    systemInstruction: system.length ? { parts: system } : undefined,
    tools: declarations.length ? [{ functionDeclarations: declarations }] : undefined,
    toolConfig: declarations.length ? { functionCallingConfig: {
      mode: choiceMode, allowedFunctionNames: choice.type === 'tool' ? [choice.name] : undefined,
    } } : undefined,
    generationConfig: {
      maxOutputTokens: body.max_tokens, stopSequences: body.stop_sequences, temperature: body.temperature,
      topP: body.top_p, responseMimeType: jsonMode ? 'application/json' : undefined,
      responseJsonSchema: object(responseFormat.json_schema).schema,
    },
  };
}

function resultReason(reason: unknown, hasTools = false): string {
  if (hasTools || reason === 'tool_use') return 'tool_calls';
  if (reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (['refusal', 'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'RECITATION'].includes(String(reason))) return 'content_filter';
  return 'stop';
}

function tokenUsage(input: unknown, output: unknown): Json {
  const prompt = typeof input === 'number' ? input : 0;
  const completion = typeof output === 'number' ? output : 0;
  return { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion };
}

function openAITool(id: unknown, name: unknown, parameters: unknown): Json {
  return { id, type: 'function', function: { name, arguments: JSON.stringify(parameters ?? {}) } };
}

export function toChatAnswer(body: Json, wire: 'anthropic' | 'gemini', model: string): Json {
  const gemini = wire === 'gemini';
  const candidate = records(body.candidates)[0] ?? {};
  const content = gemini ? records(object(candidate.content).parts) : records(body.content);
  const text = content.flatMap((part) => typeof part.text === 'string' ? [part.text] : []).join('');
  const calls = content.flatMap((part, index) => {
    if (gemini && isRecord(part.functionCall)) {
      const call = part.functionCall;
      return [openAITool(call.id ?? `call_${index}`, call.name, call.args)];
    }
    return part.type === 'tool_use' ? [openAITool(part.id, part.name, part.input)] : [];
  });
  const usage = object(gemini ? body.usageMetadata : body.usage);
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate', object: 'chat.completion', created: 0, model,
    choices: [{ index: 0, message: { role: 'assistant', content: text || (calls.length ? null : ''),
      tool_calls: calls.length ? calls : undefined }, finish_reason: resultReason(gemini ? candidate.finishReason : body.stop_reason, calls.length > 0) }],
    usage: gemini ? tokenUsage(usage.promptTokenCount, usage.candidatesTokenCount) : tokenUsage(usage.input_tokens, usage.output_tokens),
  };
}

async function* events(stream: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      if (done && pending.trim()) pending += '\n\n';
      for (;;) {
        const boundary = /\r?\n\r?\n/.exec(pending);
        if (!boundary) break;
        const frame = pending.slice(0, boundary.index);
        pending = pending.slice(boundary.index + boundary[0].length);
        const data = frame.split(/\r?\n/).filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).replace(/^ /, '')).join('\n');
        if (!data || data === '[DONE]') continue;
        const value: unknown = JSON.parse(data);
        if (isRecord(value)) yield value;
      }
      if (done) return;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export function chatEventStream(stream: ReadableStream<Uint8Array>, wire: 'anthropic' | 'gemini', model: string): ReadableStream<Uint8Array> {
  const iterator = chatEvents(stream, wire, model);
  const encoder = new TextEncoder();
  return new ReadableStream({
    async pull(controller) {
      const next = await iterator.next();
      if (next.done) controller.close();
      else controller.enqueue(encoder.encode(next.value));
    },
    async cancel() { await iterator.return(undefined); },
  });
}

async function* chatEvents(stream: ReadableStream<Uint8Array>, wire: 'anthropic' | 'gemini', model: string): AsyncGenerator<string> {
  const base = { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: 0, model };
  const delta = (value: Json, finish: string | null = null) => sseEvent(null, { ...base, choices: [{ index: 0, delta: value, finish_reason: finish }] });
  let input: unknown;
  let output: unknown;
  let reason: unknown;
  let nextTool = 0;
  const toolIndices = new Map<unknown, number>();
  yield delta({ role: 'assistant', content: '' });
  for await (const event of events(stream)) {
    if (event.type === 'error' || isRecord(event.error)) throw new Error('Provider returned an error event');
    if (wire === 'gemini') {
      const candidate = records(event.candidates)[0] ?? {};
      for (const part of records(object(candidate.content).parts)) {
        if (typeof part.text === 'string' && part.text) yield delta({ content: part.text });
        if (isRecord(part.functionCall)) {
          const index = nextTool++;
          const { id, name, args } = part.functionCall;
          yield delta({ tool_calls: [{ index, ...openAITool(id ?? `call_${index}`, name, args) }] });
        }
      }
      const usage = object(event.usageMetadata);
      input = usage.promptTokenCount ?? input;
      output = usage.candidatesTokenCount ?? output;
      reason = candidate.finishReason ?? reason;
      continue;
    }
    if (event.type === 'message_start') {
      const usage = object(object(event.message).usage);
      input = usage.input_tokens;
      output = usage.output_tokens;
    } else if (event.type === 'content_block_start') {
      const block = object(event.content_block);
      if (block.type === 'text' && typeof block.text === 'string' && block.text) yield delta({ content: block.text });
      if (block.type === 'tool_use') {
        const index = nextTool++;
        toolIndices.set(event.index, index);
        yield delta({ tool_calls: [{ index, id: block.id, type: 'function', function: { name: block.name, arguments: '' } }] });
      }
    } else if (event.type === 'content_block_delta') {
      const part = object(event.delta);
      if (part.type === 'text_delta') yield delta({ content: part.text });
      const index = toolIndices.get(event.index);
      if (part.type === 'input_json_delta' && index !== undefined) {
        yield delta({ tool_calls: [{ index, function: { arguments: part.partial_json } }] });
      }
    } else if (event.type === 'message_delta') {
      reason = object(event.delta).stop_reason ?? reason;
      output = object(event.usage).output_tokens ?? output;
    }
  }
  if (reason === undefined) throw new Error('Provider stream ended before the finish event');
  yield delta({}, resultReason(reason, nextTool > 0));
  yield sseEvent(null, { ...base, choices: [], usage: tokenUsage(input, output) });
  yield 'data: [DONE]\n\n';
}
