import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { PROVIDERS_SCHEMA } from '../lib/providers.ts';

test('retries a malformed success using the next eligible provider', { timeout: 10000 }, async (t) => {
  const temp = mkdtempSync(join(tmpdir(), 'aggregator-fallback-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const brief = join(temp, 'brief.json');
  writeFileSync(brief, JSON.stringify({
    schema: PROVIDERS_SCHEMA,
    providers: ['a', 'b'].map((id) => ({ id, wire: 'openai', auth: 'none', path: `/p/${id}` })),
  }));
  const requests: string[] = [];
  const answer = { choices: [{ message: { role: 'assistant', content: 'fallback' } }] };
  let register: (port: number) => void;
  const registration = new Promise<number>((resolve) => { register = resolve; });
  let done = false;
  const fixture = createServer(async (req, res) => {
    const path = req.url ?? '';
    res.setHeader('content-type', 'application/json');
    if (path === '/register') {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      register(JSON.parse(Buffer.concat(chunks).toString()).port);
      res.end('{}');
    } else if (path === '/work') {
      res.end(JSON.stringify({ done }));
    } else if (path.endsWith('/v1/models')) {
      res.end(JSON.stringify({ data: [{ id: path.split('/')[2], free: true }] }));
    } else if (path.endsWith('/v1/chat/completions')) {
      const provider = path.split('/')[2];
      requests.push(provider);
      res.end(provider === 'a' ? '<html>invalid upstream response</html>' : JSON.stringify(answer));
    } else {
      res.writeHead(404).end();
    }
  });
  fixture.listen(0, '127.0.0.1');
  await once(fixture, 'listening');
  t.after(() => { fixture.closeAllConnections(); fixture.close(); });
  const address = fixture.address();
  assert.ok(address && typeof address !== 'string');
  const entry = fileURLToPath(new URL('../main.ts', import.meta.url));
  const child = spawn(process.execPath, [entry], {
    env: { ...process.env, MAKEFUN_ENV_URL: `http://127.0.0.1:${address.port}`, MAKEFUN_PROMPT_FILE: brief },
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let stderr = '';
  child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
  const exited = once(child, 'exit');
  t.after(async () => { child.kill(); await exited; });
  const port = await Promise.race([registration, exited.then(() => { throw new Error(`Aggregator exited before registration: ${stderr}`); })]);
  const send = (model: string) => fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: 'Hello' }] }),
    signal: AbortSignal.timeout(3000),
  });

  const recovered = await send('auto');
  assert.equal(recovered.status, 200);
  assert.deepEqual(await recovered.json(), answer);
  assert.deepEqual(requests, ['a', 'b']);

  const unavailable = await send('a');
  assert.equal(unavailable.status, 502);
  assert.equal((await unavailable.json()).error.message, 'Provider relay failed');
  assert.deepEqual(requests, ['a', 'b', 'a']);
  done = true;
  assert.deepEqual(await exited, [0, null]);
});
