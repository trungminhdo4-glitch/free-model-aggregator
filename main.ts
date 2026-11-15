import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import {
  buildCatalog, capabilitiesOf, isRecord, joinBase, modelFree, probeProvider, ratingOfModel,
  readBrief, meetsRequirements, type Json, type Requirement,
} from './lib/providers.ts';
import { openAIModelList, anthropicModelList, openAIError, anthropicError } from './lib/wire.ts';
import { requestUpstream, responseFor, type ClientWire } from './lib/upstream.ts';

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const envOrigin = envUrl ? new URL(envUrl).origin : '';
const inTrial = brief.directory.providers.filter(provider => {
  if (!envUrl) return false;
  try { return new URL(joinBase(envUrl, provider.base)).origin === envOrigin; }
  catch { return false; }
});
const probes = await Promise.all(inTrial.map(provider => probeProvider(envUrl, provider)));
const catalog = buildCatalog(envUrl, probes, brief.ratings);
const routes = probes.flatMap(({ provider, models }) => models.flatMap(model => {
  const rating = ratingOfModel(brief.ratings, provider.id, model.id);
  if (!modelFree(model, rating)) return [];
  return [{ provider, model, rating: rating?.rating_milli ?? 0, caps: capabilitiesOf(model, rating) }];
}));
routes.sort((a, b) => b.rating - a.rating
  || a.provider.id.localeCompare(b.provider.id, 'en') || a.model.id.localeCompare(b.model.id, 'en'));

const servedModels = [...new Map(routes.map(({ provider, model }) => [
  model.id, { id: model.id, provider: provider.id, display_name: model.display_name },
])).values()];

const manifest = brief.workload?.manifest ?? [];
const unavailable = new Set<string>();
const cooldown = new Map<string, number>();
let nextRequest = 0;

function requirements(body: Json, input: ClientWire, index: number): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const vision = messages.some(message => Array.isArray(message.content)
    && message.content.some(block => isRecord(block) && ['image', 'image_url'].includes(String(block.type))));
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const outputFormat = isRecord(body.output_config) ? body.output_config.format : undefined;
  const declared = manifest[index]?.requires ?? {};
  return {
    ...declared,
    tools: declared.tools === true || (Array.isArray(body.tools) && body.tools.length > 0),
    vision: declared.vision === true || vision,
    json: declared.json === true || format === 'json_object' || format === 'json_schema'
      || (input === 'anthropic' && isRecord(outputFormat)),
  };
}

async function readBody(req: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const raw of req) {
    const chunk = Buffer.from(raw);
    bytes += chunk.length;
    if (bytes > 32 * 1024 * 1024) throw new Error('Request exceeds 32 MiB');
    chunks.push(chunk);
  }
  const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!isRecord(value)) throw new Error('Expected a JSON object');
  return value;
}

function json(res: ServerResponse, status: number, body: Json): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

function error(res: ServerResponse, api: ClientWire, status: number, message: string): void {
  const type = status === 400 ? 'invalid_request_error'
    : status === 404 ? 'not_found_error' : status === 503 ? 'overloaded_error' : 'api_error';
  json(res, status, api === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://local').pathname;
  const api: ClientWire = path === '/v1/messages' ? 'anthropic' : 'openai';
  if (req.method === 'GET' && path === '/v1/models') {
    json(res, 200, req.headers['x-api-key'] ? anthropicModelList(servedModels) : openAIModelList(servedModels));
    return;
  }
  if (req.method !== 'POST' || !['/v1/chat/completions', '/v1/messages'].includes(path)) {
    error(res, api, 404, 'Unknown route');
    return;
  }
  let body: Json;
  try {
    body = await readBody(req);
  } catch {
    error(res, api, 400, 'Expected a JSON object within the request size limit');
    return;
  }

  const index = nextRequest++;
  const need = requirements(body, api, index);
  const attemptedProviders = new Set<string>();
  const candidates = routes.filter(({ provider, caps }) =>
    !unavailable.has(provider.id) && meetsRequirements(need, caps)
    && !(api === 'openai' && provider.wire === 'anthropic' && need.json));
  candidates.sort((a, b) =>
    Number((cooldown.get(a.provider.id) ?? -1) >= index)
    - Number((cooldown.get(b.provider.id) ?? -1) >= index));

  for (const { provider, model } of candidates) {
    if (attemptedProviders.has(provider.id)) continue;
    attemptedProviders.add(provider.id);
    let upstream: Response;
    try {
      upstream = await requestUpstream(envUrl, provider, model.id, api, body);
    } catch {
      continue;
    }
    if (!upstream.ok) {
      const reason = await upstream.text().catch(() => '');
      if (upstream.status === 402 || upstream.status === 429) {
        if (/budget is spent|\bis out\b|insufficient_quota/i.test(reason)) unavailable.add(provider.id);
        else if (upstream.headers.has('retry-after')) cooldown.set(provider.id, index + 1);
      }
      continue;
    }

    // Only fail over before a successful upstream response. Streaming a second answer after
    // sending the first can duplicate tool calls and leave clients with an invalid event sequence.
    const response = await responseFor(upstream, provider.wire, api, model.id, body.stream === true);
    res.writeHead(200, {
      'content-type': body.stream === true ? 'text/event-stream' : 'application/json',
      'cache-control': 'no-cache',
    });
    if (typeof response === 'string') res.end(response);
    else await pipeline(Readable.fromWeb(response), res);
    return;
  }
  error(res, api, 503, 'No free provider could serve the request');
}

if (process.argv.includes('--list')) {
  process.stdout.write(JSON.stringify(catalog) + '\n');
} else {
  if (!envUrl) throw new Error('MAKEFUN_ENV_URL is required');
  const server = createServer((req, res) => {
    void handle(req, res).catch(() => {
      if (res.headersSent) res.destroy();
      else error(res, req.url === '/v1/messages' ? 'anthropic' : 'openai', 502, 'Upstream response failed');
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', resolve);
  });
  try {
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('No port assigned');
    const registered = await fetch(envUrl + '/register', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
      signal: AbortSignal.timeout(5_000),
    });
    if (!registered.ok) throw new Error('Trial registration failed (' + registered.status + ')');
    await registered.arrayBuffer();
    for (;;) {
      await delay(100);
      const work = await fetch(envUrl + '/work', { signal: AbortSignal.timeout(2_000) })
        .then(response => response.ok ? response.json() : null).catch(() => null);
      if (work?.done === true) break;
    }
  } finally {
    server.close();
    server.closeAllConnections();
  }
}
