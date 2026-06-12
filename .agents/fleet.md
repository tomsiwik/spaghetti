# fleet — scaling the conductor to 100% of this machine (and beyond)

## One machine, N conductors
Open N tabs, run `experiment start` in each — that's the whole fleet story. Widen GPU concurrency
with `experiment run --parallel 2` (or `parallel: 2` in `.experimentrc`).

What makes N instances safe:
- **Channel isolation:** each session's channel server binds an OS-assigned port and registers at
  `.ralph/agent/channels/<session>.json`; the per-run watcher addresses `exp_done` to the
  submitting session only. No cross-talk.
- **Atomic claims:** `experiment claim <worker>` is a Turso transaction — two conductors can never
  hold the same experiment. Each instance claims as `$CONDUCTOR_NAME-<session8>`.
- **GPU serialization:** all runs share the pueue `experiments` group. Default `parallel: 1`
  (.experimentrc). On 48GB, `--parallel 2` is safe for inference-only evals (~8–12GB each);
  keep 1 while a training rung is queued.

What saturates the machine: token-work (spark/author/review) is CPU/network and overlaps freely
across instances; the GPU stays busy because some instance always has a run queued. 3 conductors ×
parallel 2 ≈ full utilization without Metal thrash.

## Other machines
The DB (Turso) and the bet ladders (git) are the only shared state — pueue + channels are local.
On any Mac: clone, `bun install`, creds, `experiment start --fleet N`. Claims stay atomic across
machines; league scores land in the same DB. Don't share `.ralph/` between machines.

## Maxing GPT & Gemini
They are *divergence engines inside the hats*, not extra conductors:
- **sparker** fans out per spark: `gemini` (widest divergence + newest-arxiv scouting via web) and
  `codex exec -m gpt-5.5` (systematize into a falsifiable rung mechanism). Opus grounds and files.
- **reviewer** MAY request one GPT second opinion on a borderline PROCEED (cheap adversarial diversity).
- Quota-bound, not machine-bound: more conductors ⇒ proportionally more gemini/codex calls. If a
  CLI errors, the hat does the step itself — never blocks the loop.
