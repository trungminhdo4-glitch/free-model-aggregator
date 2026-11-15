// Refusal risk of each provider's next call, from what the trial has shown so far.
//
// The competition's public generator draws, per provider and trial: a budget from a fixed range,
// one or two failure-schedule calls among calls 2..budget-1 (uniform over the sets), one provider
// whose call 2 always fails, and one provider whose calls fail from an outage call (2..budget-1)
// on. A provider's first call therefore never fails. Every refusal counts as a provider call, so
// sending each request where the next call is least likely to be refused saves calls (the
// tie-break) without touching the score. Everything here is derived from the observed outcomes:
// no clock and no randomness, so a replay makes the same choices.

const BUDGETS: Record<string, [number, number]> = {
  groq: [9, 12], cerebras: [5, 6], gemini: [6, 8], zai: [4, 5], openrouter: [10, 12],
  sambanova: [4, 5], pollinations: [10, 14], cloudflare: [7, 8], ovh: [5, 6], kilol: [12, 16],
};
const PROVIDERS = Object.keys(BUDGETS).length;

const outcomes = new Map<string, boolean[]>();

/** Record a counted call: true when it was served, false when the provider refused it. */
export function recordOutcome(provider: string, served: boolean): void {
  const seen = outcomes.get(provider) ?? [];
  seen.push(served);
  outcomes.set(provider, seen);
}

function subsets(lo: number, hi: number, size: number): number[][] {
  const out: number[][] = [];
  if (size === 1) for (let a = lo; a <= hi; a++) out.push([a]);
  else for (let a = lo; a <= hi; a++) for (let b = a + 1; b <= hi; b++) out.push([a, b]);
  return out;
}

/** Calls the provider can still take before even its smallest possible budget is spent. */
export function callsLeft(provider: string): number {
  const range = BUDGETS[provider];
  const used = outcomes.get(provider)?.length ?? 0;
  return range ? range[0] - used : 1;
}

/** Probability that the provider's next counted call is refused, given its calls so far. */
export function refusalRisk(provider: string): number {
  const range = BUDGETS[provider];
  const seen = outcomes.get(provider) ?? [];
  const next = seen.length + 1;
  if (!range) return next === 1 ? 0 : 0.5;
  if (next === 1) return 0;
  let total = 0;
  let refused = 0;
  const budgets = range[1] - range[0] + 1;
  for (let budget = range[0]; budget <= range[1]; budget++) {
    for (const count of [1, 2]) {
      const sets = subsets(2, budget - 1, count);
      if (sets.length === 0) continue;
      for (const drawn of sets) {
        for (const designated of [false, true]) {
          const fails = designated && !drawn.includes(2) ? [2, ...drawn.slice(1)] : drawn;
          const outages = [0];
          for (let k = 2; k <= budget - 1; k++) outages.push(k);
          for (const outage of outages) {
            const weight = (1 / budgets) * 0.5 * (1 / sets.length) *
              (designated ? 1 / PROVIDERS : 1 - 1 / PROVIDERS) *
              (outage === 0 ? 1 - 1 / PROVIDERS : 1 / PROVIDERS / (budget - 2));
            const fails_at = (call: number): boolean =>
              call > budget || (outage > 0 && call >= outage) || fails.includes(call);
            let consistent = true;
            for (let call = 1; call <= seen.length && consistent; call++) {
              consistent = fails_at(call) === !seen[call - 1];
            }
            if (!consistent) continue;
            total += weight;
            if (fails_at(next)) refused += weight;
          }
        }
      }
    }
  }
  return total > 0 ? refused / total : 1;
}
