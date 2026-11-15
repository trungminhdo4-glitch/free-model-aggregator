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

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
