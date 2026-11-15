import { Readable } from 'node:stream';
import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';
import { anthropicEventsFromOpenAI, anthropicToOpenAI, openAIToAnthropic, sseEvent } from './wire.ts';

export type ClientWire = 'openai' | 'anthropic';
type ProviderWire = ProviderDirectoryEntry['wire'];

const object = (value: unknown): Json => isRecord(value) ? value : {};
const objects = (value: unknown): Json[] => Array.isArray(value) ? value.filter(isRecord) : [];
const text = (value: unknown): string => typeof value === 'string' ? value : '';
const tokens = (value: unknown): number => typeof value === 'number' && Number.isFinite(value) ? value : 0;

function argumentObject(value: unknown): Json {
  if (isRecord(value)) return value;
  try { return object(JSON.parse(text(value))); } catch { return {}; }
}

function textParts(content: unknown): Json[] {
  return typeof content === 'string' ? [{ type: 'text', text: content }] : objects(content);
}

function imageData(part: Json): { mime: string; data: string } | null {
  const url = text(object(part.image_url).url);
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(url);
  return match ? { mime: match[1], data: match[2] } : null;
}

function chatToAnthropic(body: Json): Json {
  const messages = objects(body.messages);
  const system = messages.filter(message => ['system', 'developer'].includes(text(message.role)))
    .flatMap(message => textParts(message.content)).filter(part => part.type === 'text');
  const conversation = messages.flatMap<Json>(message => {
    if (['system', 'developer'].includes(text(message.role))) return [];
    if (message.role === 'tool') return [{
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: message.tool_call_id, content: message.content }],
    }];
    const content = textParts(message.content).flatMap<Json>(part => {
      if (part.type === 'text') return [part];
      if (part.type !== 'image_url') return [];
      const image = imageData(part);
      const source = image ? { type: 'base64', media_type: image.mime, data: image.data }
        : { type: 'url', url: object(part.image_url).url };
      return [{ type: 'image', source }];
    });
    for (const call of objects(message.tool_calls)) {
      const fn = object(call.function);
      content.push({ type: 'tool_use', id: call.id, name: fn.name, input: argumentObject(fn.arguments) });
    }
    return [{ role: message.role === 'assistant' ? 'assistant' : 'user', content }];
  });
  const result: Json = {
    model: body.model,
    messages: conversation,
    max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 1024,
    stream: body.stream === true,
  };
  if (system.length) result.system = system;
  if (body.stop !== undefined) result.stop_sequences = typeof body.stop === 'string' ? [body.stop] : body.stop;
  for (const key of ['temperature', 'top_p']) if (body[key] !== undefined) result[key] = body[key];
  const tools = objects(body.tools).map(tool => object(tool.function)).filter(fn => typeof fn.name === 'string');
  if (tools.length) result.tools = tools.map(fn => ({
    name: fn.name, description: fn.description, input_schema: fn.parameters ?? { type: 'object' },
  }));
  const tool = object(object(body.tool_choice).function).name;
  if (typeof tool === 'string') result.tool_choice = { type: 'tool', name: tool };
  else if (body.tool_choice === 'required') result.tool_choice = { type: 'any' };
  else if (body.tool_choice === 'auto' || body.tool_choice === 'none') result.tool_choice = { type: body.tool_choice };
  return result;
}

function chatToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system: Json[] = [];
  const names = new Map<unknown, unknown>();
  for (const message of objects(body.messages)) {
    const parts = textParts(message.content).flatMap<Json>(part => {
      if (part.type === 'text') return [{ text: part.text }];
      if (part.type !== 'image_url') return [];
      const image = imageData(part);
      return image ? [{ inlineData: { mimeType: image.mime, data: image.data } }]
        : [{ fileData: { fileUri: object(part.image_url).url } }];
    });
    if (['system', 'developer'].includes(text(message.role))) {
      system.push(...parts);
      continue;
    }
    if (message.role === 'tool') {
      const name = names.get(message.tool_call_id) ?? message.name;
      contents.push({ role: 'user', parts: [{
        functionResponse: { name, response: { result: message.content } },
      }] });
      continue;
    }
    for (const call of objects(message.tool_calls)) {
      const fn = object(call.function);
      names.set(call.id, fn.name);
      parts.push({ functionCall: { name: fn.name, args: argumentObject(fn.arguments) } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const config: Json = {
    maxOutputTokens: body.max_tokens ?? body.max_completion_tokens,
    stopSequences: typeof body.stop === 'string' ? [body.stop] : body.stop,
    temperature: body.temperature,
    topP: body.top_p,
  };
  const format = object(body.response_format);
  if (format.type === 'json_schema' || format.type === 'json_object') {
    config.responseMimeType = 'application/json';
    const schema = object(format.json_schema).schema;
    if (isRecord(schema)) config.responseJsonSchema = schema;
  }
  const payload: Json = { contents, generationConfig: config };
  if (system.length) payload.systemInstruction = { parts: system };
  const tools = objects(body.tools).map(tool => object(tool.function)).filter(fn => typeof fn.name === 'string');
  if (tools.length) {
    payload.tools = [{ functionDeclarations: tools.map(fn => ({
      name: fn.name, description: fn.description, parametersJsonSchema: fn.parameters,
    })) }];
    const choiceName = object(object(body.tool_choice).function).name;
    const mode = body.tool_choice === 'none' ? 'NONE'
      : body.tool_choice === 'required' || typeof choiceName === 'string' ? 'ANY' : 'AUTO';
    payload.toolConfig = { functionCallingConfig: {
      mode, ...(typeof choiceName === 'string' ? { allowedFunctionNames: [choiceName] } : {}),
    } };
  }
  return payload;
}

export async function requestUpstream(
  envUrl: string, provider: ProviderDirectoryEntry, model: string, api: ClientWire, body: Json,
): Promise<Response> {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: body.stream === true ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const chat = api === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
  const format = object(object(body.output_config).format);
  if (api === 'anthropic' && format.type === 'json_schema') {
    chat.response_format = { type: 'json_schema', json_schema: { name: 'output', schema: format.schema } };
  }
  let path: string;
  let payload: Json;
  switch (provider.wire) {
    case 'openai':
      path = '/v1/chat/completions';
      payload = chat;
      if (body.stream === true) payload.stream_options = { ...object(chat.stream_options), include_usage: true };
      break;
    case 'anthropic':
      path = '/v1/messages';
      payload = api === 'anthropic' ? { ...body, model } : chatToAnthropic(chat);
      headers['anthropic-version'] = '2023-06-01';
      break;
    case 'gemini':
      path = '/v1beta/models/' + encodeURIComponent(model)
        + (body.stream === true ? ':streamGenerateContent?alt=sse' : ':generateContent');
      payload = chatToGemini(chat);
      break;
  }
  return fetch(base + path, {
    method: 'POST', headers, body: JSON.stringify(payload), redirect: 'error',
    signal: AbortSignal.timeout(12_000),
  });
}

function usage(input: unknown, output: unknown): Json {
  return { prompt_tokens: tokens(input), completion_tokens: tokens(output), total_tokens: tokens(input) + tokens(output) };
}

function finish(reason: unknown): string {
  if (['MAX_TOKENS', 'max_tokens'].includes(text(reason))) return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (['SAFETY', 'RECITATION', 'refusal'].includes(text(reason))) return 'content_filter';
  return 'stop';
}

function call(id: unknown, name: unknown, args: unknown): Json {
  return { id, type: 'function', function: { name, arguments: JSON.stringify(object(args)) } };
}

function transcript(wire: ProviderWire, answer: Json, model: string): Json {
  const candidate = objects(answer.candidates)[0] ?? {};
  const parts = wire === 'gemini' ? objects(object(candidate.content).parts) : objects(answer.content);
  const content = parts.map(part => text(part.text)).join('');
  const calls = parts.flatMap((part, index) => {
    if (wire === 'anthropic' && part.type === 'tool_use') return [call(part.id, part.name, part.input)];
    const fn = object(part.functionCall);
    return wire === 'gemini' && typeof fn.name === 'string' ? [call(fn.id ?? 'call_' + index, fn.name, fn.args)] : [];
  });
  const count = object(wire === 'gemini' ? answer.usageMetadata : answer.usage);
  return {
    id: answer.id ?? answer.responseId ?? 'chatcmpl-bridge',
    model, object: 'chat.completion', created: 0,
    choices: [{
      index: 0, finish_reason: calls.length ? 'tool_calls' : finish(answer.stop_reason ?? candidate.finishReason),
      message: { role: 'assistant', content: content || (calls.length ? null : ''), ...(calls.length ? { tool_calls: calls } : {}) },
    }],
    usage: wire === 'gemini' ? usage(count.promptTokenCount, count.candidatesTokenCount)
      : usage(count.input_tokens, count.output_tokens),
  };
}

async function* frames(body: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() + '\n\n' : decoder.decode(value, { stream: true });
      let boundary = /\r?\n\r?\n/.exec(buffer);
      while (boundary) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
          .map(line => line.substring(5).trimStart()).join('\n');
        if (data === '[DONE]') return;
        if (data) yield object(JSON.parse(data));
        boundary = /\r?\n\r?\n/.exec(buffer);
      }
      if (done) return;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function* chatEvents(body: ReadableStream<Uint8Array>, wire: ProviderWire, model: string): AsyncGenerator<string> {
  const base = { id: 'chatcmpl-bridge', object: 'chat.completion.chunk', created: 0, model };
  const chunk = (delta: Json, reason: string | null = null) =>
    sseEvent(null, { ...base, choices: [{ index: 0, delta, finish_reason: reason }] });
  let reason = 'stop';
  let input = 0;
  let output = 0;
  let nextTool = 0;
  const toolIndexes = new Map<unknown, number>();
  yield chunk({ role: 'assistant', content: '' });
  for await (const event of frames(body)) {
    if (wire === 'gemini') {
      const candidate = objects(event.candidates)[0] ?? {};
      if (candidate.finishReason) reason = finish(candidate.finishReason);
      for (const part of objects(object(candidate.content).parts)) {
        if (typeof part.text === 'string' && part.text) yield chunk({ content: part.text });
        const fn = object(part.functionCall);
        if (typeof fn.name === 'string') {
          const index = nextTool++;
          yield chunk({ tool_calls: [{ index, ...call(fn.id ?? 'call_' + index, fn.name, fn.args) }] });
        }
      }
      const count = object(event.usageMetadata);
      input = tokens(count.promptTokenCount ?? input);
      output = tokens(count.candidatesTokenCount ?? output);
      continue;
    }
    if (event.type === 'message_start') {
      input = tokens(object(object(event.message).usage).input_tokens);
    } else if (event.type === 'content_block_start') {
      const block = object(event.content_block);
      if (block.type === 'text' && block.text) yield chunk({ content: block.text });
      if (block.type === 'tool_use') {
        const index = nextTool++;
        toolIndexes.set(event.index, index);
        yield chunk({ tool_calls: [{ index, id: block.id, type: 'function', function: { name: block.name, arguments: '' } }] });
      }
    } else if (event.type === 'content_block_delta') {
      const delta = object(event.delta);
      if (delta.type === 'text_delta') yield chunk({ content: delta.text });
      if (delta.type === 'input_json_delta') {
        yield chunk({ tool_calls: [{ index: toolIndexes.get(event.index), function: { arguments: delta.partial_json } }] });
      }
    } else if (event.type === 'message_delta') {
      reason = finish(object(event.delta).stop_reason);
      output = tokens(object(event.usage).output_tokens);
    }
  }
  yield chunk({}, nextTool ? 'tool_calls' : reason);
  yield sseEvent(null, { ...base, choices: [], usage: usage(input, output) });
  yield 'data: [DONE]\n\n';
}

function eventBody(events: AsyncIterable<string>): ReadableStream<Uint8Array> {
  async function* encode() {
    const encoder = new TextEncoder();
    for await (const event of events) yield encoder.encode(event);
  }
  return Readable.toWeb(Readable.from(encode())) as ReadableStream<Uint8Array>;
}

export async function responseFor(
  response: Response, wire: ProviderWire, api: ClientWire, model: string, stream: boolean,
): Promise<string | ReadableStream<Uint8Array>> {
  if (!stream) {
    if (wire === api) return response.text();
    const body = object(await response.json());
    const chat = wire === 'openai' ? body : transcript(wire, body, model);
    return JSON.stringify(api === 'openai' ? chat : openAIToAnthropic(chat, model));
  }
  if (!response.body) throw new Error('Upstream returned no stream');
  if (wire === api) return response.body;
  const chat = wire === 'openai' ? response.body : eventBody(chatEvents(response.body, wire, model));
  return api === 'openai' ? chat : eventBody(anthropicEventsFromOpenAI(chat, model));
}
