# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/trungminhdo4-glitch/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/trungminhdo4-glitch/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server exposes OpenAI Chat Completions and Anthropic Messages over OpenAI, Anthropic, and Gemini providers. It probes provider catalogs, ranks eligible free models by the trial's quality table, and falls back when a provider refuses a request. You can use either client API for JSON responses or server-sent events.

## Run it

Use Node.js 22.18 or later. Set `MAKEFUN_ENV_URL` to the trial environment's URL and `MAKEFUN_PROMPT_FILE` to the brief provided by the kit.

```sh
node main.ts
node main.ts --list
```

The trial client discovers the server's port and its catalog through registration. `--list` prints the same catalog without starting a listener.

This entry point implements the trial lifecycle: provider discovery, registration,
and polling `/work` until the trial completes. It is not a standalone production
gateway. With no trial environment configured, `node main.ts --list` prints an
empty provider catalog and makes no network requests.

## Client APIs and request lifecycle

- `GET /v1/models` lists eligible discovered models.
- `POST /v1/chat/completions` accepts the OpenAI chat wire format.
- `POST /v1/messages` accepts the Anthropic messages wire format.
- `model: "auto"` selects an eligible free route; an explicit model limits
  fallback to providers offering that model.

Requests are processed serially. A provider call retains its existing 10-second
deadline; refusals, connection failures, and malformed buffered responses can
fall back to the next eligible route before response headers are sent. Once a
stream has started, a failure terminates that stream rather than mixing answers
from different providers.

If a client disconnects, the server aborts its active provider fetch, stops
fallback for that request, and skips any disconnected request still in the
queue. Cancellation covers pending headers, buffered response bodies, and
streams. This releases local work and lets the next request run; it cannot undo
provider-side computation or guarantee that an already accepted call consumes
no quota. Normal completed responses do not cancel subsequent requests.

## Offline development and tests

No package installation, provider key, or trial access is needed for the tests:

```sh
npm test
```

The test command runs `node --test "test/*.test.ts"`. Local HTTP fixtures provide
synthetic catalogs and responses. Tests exercise malformed-response fallback,
client cancellation before headers, during a JSON body, and during an SSE stream,
skipped canceled queue entries, and recovery of the next request. The fixture
processes set their own loopback environment URL; they do not call real model
providers. Node must be allowed to open local loopback sockets.

## Architecture and limitations

`lib/providers.ts` reads the brief and probes catalog contracts, `lib/risk.ts`
tracks trial-specific refusal estimates, and `lib/wire.ts`/`lib/upstream.ts`
translate supported messages and streams. `main.ts` selects routes and owns the
HTTP lifecycle. The implementation has no runtime package dependencies.

Ranking is tailored to the trial's supplied ratings and workload manifest.
There is no durable queue, authentication layer, or persistent provider cache.
The tests establish behavior for their synthetic contracts, not compatibility
with every real provider API or equivalent model output quality.

## License

MIT; see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
