import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer, request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { PROVIDERS_SCHEMA } from '../lib/providers.ts';

for (const mode of ['headers', 'json-body', 'stream']) {
test(`disconnect during ${mode} cancels the provider, skips queued work and leaves the queue usable`, { timeout: 8000 }, async (t) => {
  const temp = mkdtempSync(join(tmpdir(), 'aggregator-cancel-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const brief = join(temp, 'brief.json');
  writeFileSync(brief, JSON.stringify({
    schema: PROVIDERS_SCHEMA,
    providers: ['a', 'b'].map((id) => ({ id, wire: 'openai', auth: 'none', path: `/p/${id}` })),
  }));
  const requests: string[] = [];
  const answer = { choices: [{ message: { role: 'assistant', content: 'recovered' } }] };
  let registered!: (port: number) => void;
  const registration = new Promise<number>((resolve) => { registered = resolve; });
  let activeStarted!: () => void;
  const started = new Promise<void>((resolve) => { activeStarted = resolve; });
  let activeClosed!: () => void;
  const closed = new Promise<void>((resolve) => { activeClosed = resolve; });
  let done = false;
  const fixture = createServer(async (req, res) => {
    const path = req.url ?? '';
    res.setHeader('content-type', 'application/json');
    if (path === '/register') {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      registered(JSON.parse(Buffer.concat(chunks).toString()).port);
      res.end('{}');
    } else if (path === '/work') {
      res.end(JSON.stringify({ done }));
    } else if (path.endsWith('/v1/models')) {
      res.end(JSON.stringify({ data: [{ id: path.split('/')[2], free: true }] }));
    } else if (path.endsWith('/v1/chat/completions')) {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const marker = JSON.parse(Buffer.concat(chunks).toString()).messages[0].content;
      requests.push(`${path.split('/')[2]}:${marker}`);
      if (marker === 'active') {
        res.once('close', activeClosed);
        if (mode === 'json-body') res.write('{"choices":');
        if (mode === 'stream') {
          res.setHeader('content-type', 'text/event-stream');
          res.write('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n');
        }
        activeStarted();
        // Deliberately leave the upstream response pending until its fetch aborts.
      } else {
        res.end(JSON.stringify(answer));
      }
    } else {
      res.writeHead(404).end();
    }
  });
  fixture.listen(0, '127.0.0.1');
  await once(fixture, 'listening');
  t.after(() => { fixture.closeAllConnections(); fixture.close(); });
  const address = fixture.address();
  assert.ok(address && typeof address !== 'string');
  const child = spawn(process.execPath, [fileURLToPath(new URL('../main.ts', import.meta.url))], {
    env: { ...process.env, MAKEFUN_ENV_URL: `http://127.0.0.1:${address.port}`, MAKEFUN_PROMPT_FILE: brief },
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let stderr = '';
  child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
  const exited = once(child, 'exit');
  t.after(async () => { child.kill(); await exited; });
  const port = await Promise.race([registration, exited.then(() => { throw new Error(`Aggregator exited before registration: ${stderr}`); })]);
  const payload = (marker: string) => JSON.stringify({
    model: 'auto', messages: [{ role: 'user', content: marker }], stream: marker === 'active' && mode === 'stream',
  });
  const send = (marker: string) => {
    const client = request(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST', headers: {
        'content-type': 'application/json',
        ...(marker === 'queued' ? { expect: '100-continue' } : {}),
      },
    });
    client.on('error', () => { /* An intentional disconnect reports ECONNRESET. */ });
    if (marker === 'queued') client.flushHeaders();
    else client.end(payload(marker));
    return client;
  };
  const active = send('active');
  const streamed = mode === 'stream' ? new Promise<void>((resolve) => {
    active.once('response', (response) => response.once('data', () => resolve()));
  }) : Promise.resolve();
  t.after(() => active.destroy());
  await started;
  await streamed;
  const queued = send('queued');
  // Node sends 100 Continue and dispatches the request handler synchronously,
  // proving the queued request reached the server before we disconnect it.
  await once(queued, 'continue');
  queued.end(payload('queued'));
  await once(queued, 'finish');
  const queuedClosed = new Promise<void>((resolve) => queued.once('close', resolve));
  queued.destroy();
  await queuedClosed;
  active.destroy();
  let deadline: ReturnType<typeof setTimeout>;
  try {
    await Promise.race([closed, new Promise<never>((_resolve, reject) => {
      deadline = setTimeout(() => reject(new Error('Active provider was not canceled after client disconnect')), 1500);
    })]);
  } finally {
    clearTimeout(deadline!);
  }

  const recovered = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: payload('recovery'),
    signal: AbortSignal.timeout(2000),
  });
  assert.equal(recovered.status, 200);
  assert.deepEqual(await recovered.json(), answer);
  assert.deepEqual(requests, ['a:active', 'a:recovery']);
  assert.equal(stderr, '');
  done = true;
  assert.deepEqual(await exited, [0, null]);
});
}
