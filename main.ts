import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { setTimeout as delay } from 'node:timers/promises';
import {
  authHeaders, buildCatalog, capabilitiesOf, isRecord, joinBase, meetsRequirements,
  modelFree, probeProvider, ratingOfModel, readBrief,
  type Json, type Probe, type ProbedModel, type Requirement,
} from './lib/providers.ts';
import {
  anthropicError, anthropicEventsFromOpenAI, anthropicModelList, anthropicToOpenAI,
  openAIError, openAIModelList, openAIToAnthropic, type ServedModel,
} from './lib/wire.ts';
import { chatEventStream, chatToMessages, messagesToGemini, records, toChatAnswer } from './lib/upstream.ts';
import { callsLeft, recordOutcome, refusalRisk } from './lib/risk.ts';

// The score is round(0.6 * quality + 40) with the other columns full, and quality is the floor of
// the weighted rating share: any quality of 98 or more gives the same score. So up to this share of
// the trial's best possible weighted rating may be given up for routes whose next call is less
// likely to be refused (each refusal is a counted call, the tie-break), keeping room for fallbacks.
const PLANNED_LOSS = 0.012;
// What a weighted rating point given up is worth in refusals (expected calls): spend the
// allowance only where it buys a lower refusal risk, not on a route that is merely as safe.
const LOSS_PRICE = 0.01;
// Routes within this share of a later request's best rating count as able to serve it.
const SERVING_BAND = 0.96;
let qualityBound = 0;
let qualityLoss = 0;

type ClientWire = 'openai' | 'anthropic';
type Route = { probe: Probe; model: ProbedModel; rating: number; capabilities: ReturnType<typeof capabilitiesOf> };

const env = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const manifest = brief.workload?.manifest ?? [];
const exhausted = new Set<string>();
let requestIndex = 0;
let candidates: Route[] = [];
let modelList: ServedModel[] = [];
let outstanding: Promise<void> = Promise.resolve();

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) });
  res.end(data);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, message: string): void {
  const type = status === 400 ? 'invalid_request_error' : status === 404 ? 'not_found_error' : 'overloaded_error';
  sendJson(res, status, wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null));
}

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 32 * 1024 * 1024) return null;
    chunks.push(buffer);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function requirements(body: Json, index: number): Requirement {
  const declared = manifest[index]?.requires;
  const format = isRecord(body.response_format) ? body.response_format.type : null;
  const vision = records(body.messages).some((message) =>
    records(message.content).some((block) => block.type === 'image' || block.type === 'image_url'),
  );
  return {
    ...declared,
    tools: declared?.tools || records(body.tools).length > 0,
    vision: declared?.vision || vision,
    json: declared?.json || format === 'json_object' || format === 'json_schema',
  };
}

function eligible(route: Route, needs: Requirement, client: ClientWire): boolean {
  return !exhausted.has(route.probe.provider.id) &&
    !(client === 'openai' && needs.json && route.probe.provider.wire === 'anthropic') &&
    meetsRequirements(needs, route.capabilities);
}

function topRating(needs: Requirement, wire: ClientWire | null): number {
  return candidates.filter((route) => meetsRequirements(needs, route.capabilities) &&
    !(wire === 'openai' && needs.json && route.probe.provider.wire === 'anthropic'))
    .reduce((top, route) => Math.max(top, route.rating), 0);
}

type Plan = { routes: Route[]; top: number; weight: number };

function choices(body: Json, wire: ClientWire, index: number): Plan {
  const needs = requirements(body, index);
  const available = candidates.filter((route) => eligible(route, needs, wire) &&
    (body.model === 'auto' || typeof body.model !== 'string' || route.model.id === body.model));
  const used = new Set<string>();
  const perProvider = available.filter((route) => {
    const provider = route.probe.provider.id;
    if (used.has(provider)) return false;
    used.add(provider);
    return true;
  });
  const best = topRating(needs, wire);
  const weight = Number(manifest[index]?.weight ?? 1);
  const allowance = PLANNED_LOSS * qualityBound - qualityLoss;
  const strong = perProvider.filter((route) => weight * (best - route.rating) <= allowance);
  const weak = perProvider.filter((route) => weight * (best - route.rating) > allowance);
  // The manifest names every later request's requirements: a provider that few strong routes can
  // replace for those (images, say) is kept for them, as it would cost a whole request to run dry.
  const demand = new Map<string, number>();
  for (const later of manifest.slice(index + 1)) {
    const laterNeeds = (later?.requires ?? {}) as Requirement;
    const fits = candidates.filter((route) => !exhausted.has(route.probe.provider.id) &&
      meetsRequirements(laterNeeds, route.capabilities));
    const top = fits.reduce((m, route) => Math.max(m, route.rating), 0);
    const providers = new Set(fits.filter((route) => route.rating >= top * SERVING_BAND)
      .map((route) => route.probe.provider.id));
    for (const provider of providers) demand.set(provider, (demand.get(provider) ?? 0) + 1 / providers.size);
  }
  const cost = new Map(strong.map((route) => {
    const provider = route.probe.provider.id;
    const pressure = (demand.get(provider) ?? 0) / Math.max(callsLeft(provider), 0.5);
    const loss = LOSS_PRICE * weight * Math.max(0, best - route.rating);
    return [provider, refusalRisk(provider) + pressure + loss];
  }));
  strong.sort((a, b) => (cost.get(a.probe.provider.id) ?? 1) - (cost.get(b.probe.provider.id) ?? 1) ||
    b.rating - a.rating || a.probe.provider.id.localeCompare(b.probe.provider.id, 'en'));
  return { routes: [...strong, ...weak], top: best, weight };
}

async function call(route: Route, input: Json, wire: ClientWire, version: string | undefined, signal: AbortSignal): Promise<Response> {
  const provider = route.probe.provider;
  const base = joinBase(env, provider.base);
  const key = process.env[provider.key_env] ?? provider.key;
  const headers = { 'content-type': 'application/json', ...authHeaders(provider.auth, key) };
  const stream = input.stream === true;
  const model = route.model.id;
  let path: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    path = '/v1/chat/completions';
    payload = wire === 'openai' ? { ...input, model } : anthropicToOpenAI(input, model);
    if (stream) payload.stream_options = { ...(isRecord(payload.stream_options) ? payload.stream_options : {}), include_usage: true };
  } else {
    const messages = wire === 'anthropic' ? { ...input, model } : chatToMessages(input, model);
    if (provider.wire === 'anthropic') {
      path = '/v1/messages';
      payload = messages;
      headers['anthropic-version'] = version ?? '2023-06-01';
    } else {
      const operation = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
      path = '/v1beta/models/' + encodeURIComponent(model) + ':' + operation;
      payload = messagesToGemini(messages, input.response_format);
    }
  }
  return fetch(base + path, {
    method: 'POST', headers, body: JSON.stringify(payload), redirect: 'error',
    signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
  });
}

async function recordRefusal(route: Route, response: Response): Promise<void> {
  let error: Json = {};
  try {
    const value: unknown = await response.json();
    if (isRecord(value) && isRecord(value.error)) error = value.error;
  } catch {
    // Non-JSON failures are retryable on another provider.
  }
  const message = typeof error.message === 'string' ? error.message : '';
  const counted = (response.status === 402 || response.status === 429) && error.code !== 'concurrent_requests';
  if (counted) recordOutcome(route.probe.provider.id, false);
  const terminal = response.status === 401 || response.status === 403 ||
    (response.status === 402 || response.status === 429) && /free-tier budget is spent|\bis out$/i.test(message);
  if (terminal) exhausted.add(route.probe.provider.id);
}

async function relay(res: ServerResponse, response: Response, route: Route, wire: ClientWire, stream: boolean): Promise<void> {
  const source = route.probe.provider.wire;
  const model = route.model.id;
  if (!stream) {
    const answer: unknown = await response.json();
    if (!isRecord(answer)) throw new Error('Provider response is not a JSON object');
    const chat = source === 'openai' ? answer : source === wire ? null : toChatAnswer(answer, source, model);
    const output = source === wire ? answer : wire === 'openai' ? chat : openAIToAnthropic(chat ?? {}, model);
    sendJson(res, 200, output);
    return;
  }
  if (!response.body) throw new Error('Provider response is missing a stream');
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
  if (source === wire) {
    await pipeline(Readable.fromWeb(response.body), res);
    return;
  }
  const chat = source === 'openai' ? response.body : chatEventStream(response.body, source, model);
  const output = wire === 'openai' ? Readable.fromWeb(chat) : Readable.from(anthropicEventsFromOpenAI(chat, model));
  await pipeline(output, res);
}

async function complete(req: IncomingMessage, res: ServerResponse, wire: ClientWire, signal: AbortSignal): Promise<void> {
  const body = await readBody(req);
  if (signal.aborted) return;
  if (!body) return sendError(res, wire, 400, 'Expected a JSON request body');
  const index = requestIndex++;
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : undefined;
  const plan = choices(body, wire, index);
  let relayError: unknown;
  for (const route of plan.routes) {
    if (signal.aborted) return;
    let response: Response;
    try {
      response = await call(route, body, wire, version, signal);
    } catch {
      if (signal.aborted) return;
      continue;
    }
    if (!response.ok) {
      await recordRefusal(route, response);
      continue;
    }
    try {
      await relay(res, response, route, wire, body.stream === true);
    } catch (error) {
      if (signal.aborted) return;
      if (res.headersSent || res.destroyed) throw error;
      relayError = error;
      continue;
    }
    recordOutcome(route.probe.provider.id, true);
    qualityLoss += plan.weight * Math.max(0, plan.top - route.rating);
    return;
  }
  if (relayError !== undefined) throw relayError;
  sendError(res, wire, 503, 'No available free model can serve this request');
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const path = new URL(req.url ?? '/', 'http://aggregator').pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    const wire = typeof req.headers['x-api-key'] === 'string';
    return sendJson(res, 200, wire ? anthropicModelList(modelList) : openAIModelList(modelList));
  }
  const wire = path === '/v1/messages' ? 'anthropic' : 'openai';
  if (req.method !== 'POST' || (path !== '/v1/messages' && path !== '/v1/chat/completions')) {
    return sendError(res, wire, 404, 'Unknown API route');
  }
  const disconnected = new AbortController();
  const onDisconnect = () => {
    // IncomingMessage 'close' also fires after a normal upload; response close
    // before writableFinished identifies the client going away instead.
    if (!res.writableFinished) disconnected.abort();
  };
  res.once('close', onDisconnect);
  req.once('aborted', onDisconnect);
  outstanding = outstanding.then(() => {
    if (disconnected.signal.aborted || res.destroyed) return;
    return complete(req, res, wire, disconnected.signal);
  }).catch((error: Error) => {
    if (disconnected.signal.aborted || res.destroyed) return;
    process.stderr.write('Provider relay failed: ' + error.message + '\n');
    if (res.headersSent) res.destroy();
    else sendError(res, wire, 502, 'Provider relay failed');
  }).finally(() => {
    res.removeListener('close', onDisconnect);
    req.removeListener('aborted', onDisconnect);
  });
}

function loadRoutes(probes: Probe[]): Route[] {
  return probes.flatMap((probe) => probe.models.flatMap((model) => {
    const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
    return modelFree(model, row)
      ? [{ probe, model, rating: row?.rating_milli ?? 0, capabilities: capabilitiesOf(model, row) }]
      : [];
  })).sort((a, b) => b.rating - a.rating || a.probe.provider.id.localeCompare(b.probe.provider.id, 'en') ||
    a.model.id.localeCompare(b.model.id, 'en'));
}

async function start(): Promise<void> {
  const probes: Probe[] = [];
  if (env) {
    for (const provider of brief.directory.providers) {
      const probe = await probeProvider(env, { ...provider, key: process.env[provider.key_env] ?? provider.key });
      if (probe.error) process.stderr.write(probe.error + '\n');
      probes.push(probe);
    }
  }
  const catalog = buildCatalog(env, probes, brief.ratings);
  if (process.argv.includes('--list')) {
    process.stdout.write(JSON.stringify(catalog) + '\n');
    return;
  }
  if (!env) throw new Error('MAKEFUN_ENV_URL is required');
  candidates = loadRoutes(probes);
  qualityBound = manifest.reduce((sum, entry) =>
    sum + Number(entry?.weight ?? 1) * topRating((entry?.requires ?? {}) as Requirement, null), 0);
  modelList = [...new Map(candidates.map(({ probe, model }) => [
    model.id, { id: model.id, provider: probe.provider.id, display_name: model.display_name },
  ])).values()].sort((a, b) => a.id.localeCompare(b.id, 'en'));
  const server = createServer(handle);
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('The server failed to bind to a TCP port');
  try {
    const response = await fetch(env + '/register', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
      signal: AbortSignal.timeout(5000), redirect: 'error',
    });
    if (!response.ok) throw new Error('Registration failed with HTTP ' + response.status);
    await response.arrayBuffer();
    let missedPolls = 0;
    for (;;) {
      try {
        const check = await fetch(env + '/work', { signal: AbortSignal.timeout(2000), redirect: 'error' });
        if (!check.ok) throw new Error('Work endpoint returned HTTP ' + check.status);
        const work: unknown = await check.json();
        if (isRecord(work) && work.done === true) break;
        missedPolls = 0;
      } catch (error) {
        if (++missedPolls === 3) throw error;
      }
      await delay(200);
    }
    await outstanding;
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

await start();
