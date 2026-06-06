# AGENTS.md — start here

MLX-native, Apple-Silicon research repo for composable LoRA-adapter experiments on Gemma-4. Read these three, in order:

1. **`STATUS.md`** — what is actually true now (verified against the experiment DB; the source of truth). Read before repeating any claim.
2. **`experiments/GUIDE.md`** — **how to run experiments** (the loop, proof-first discipline, kill/verdict rules, CLI, platform rules). The canonical process doc.
3. **`PLAN.md`** — current roadmap + platform (base model, adapter recipe, next gate).

## Repo map (5 folders)

| Folder | What |
|---|---|
| `pierre/` | live MLX code — composition core (`pierre.py`) + `merge/` (merge libraries) |
| `experiments/` | the research log — `models/` (local MLX), `macro/` (GPU), `_runs/`, shared lib, `GUIDE.md` |
| `tooling/` | framework — `packages/` (experiment CLI + Turso DB), `scripts/`, `tools/` |
| `data/` | `adapters/` (weights + registry) + `corpora/` (datasets) |
| `docs/` | guides, reference, research notes, archive |

## Non-negotiables

- Track every experiment with the **`experiment` CLI** (load the `experiment` skill for flags). Run experiments via `experiment run <id>` — never bare `uv run python`.
- **Invoke `/mlx-dev` and `/fast-mlx` before writing MLX code.**
- Proof-first: `MATH.md` (theorem + pre-registered kill criteria) before code. Every claim needs a behavioral target-metric KC (`experiments/GUIDE.md §3`).
