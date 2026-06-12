# How agents run experiments — the canonical guide

> **Single source of truth for the experiment process.** If any other doc describes how to run experiments, it should defer to this file. Companion docs: `STATUS.md` (what's true now), `PLAN.md` (roadmap + platform), `README.md` (repo map).

Every experiment lives in its own folder under `experiments/models/<name>/` (micro, local MLX) or `experiments/macro/<name>/` (GPU/RunPod). One experiment = one folder = the quartet below. The experiment DB (Turso, via the `experiment` CLI) is the authoritative index.

---

## 1. The loop

```
claim ─▶ MATH.md ─▶ run_experiment.py ─▶ results.json ─▶ PAPER.md ─▶ REVIEW-adversarial.md ─▶ complete ─▶ finding-add ─▶ (loop)
```

1. **Load the `experiment` skill** to get exact CLI flags (do this before any `experiment` command).
2. `experiment list -s open` → pick work; `experiment claim <worker>` to grab the next (returns full YAML: kill criteria, deps, tags).
3. **Invoke `/mlx-dev` and `/fast-mlx` before writing any MLX code** — skipping them is the #1 cause of broken experiments.
4. Write **`MATH.md`**: theorem + quantitative predictions + pre-registered kill criteria (K1/K2/K3). Cite an arxiv id or a prior finding. No code before the proof.
5. Write **`run_experiment.py`** and run it with **`experiment run <id>`** (uses `pueue` — NEVER bare `uv run python`; pueue guarantees Metal-buffer cleanup).
6. Read **`results.json`**; write **`PAPER.md`** with a prediction-vs-measurement table.
7. Write **`REVIEW-adversarial.md`** — self-review, max 3 blocking issues.
8. `experiment complete <id>` with status + kill results + evidence + dir.
9. `experiment finding-add` with title/status/result/caveat/failure-mode/impossibility-structure.
10. Loop.

**Backlog-drain done (`RESEARCH_BACKLOG_DRAINED`):** `experiment list --status open` has nothing at priority ≤ 2; `--status active` is empty (nothing stuck claimed); every completed experiment has the full file set.

---

## 2. Proof-first discipline (constructive mathematics)

Every experiment needs a formal proof **before** code:

1. **Identify the failure mode** — what specific degenerate behavior is prevented?
2. **Cite prior math** — JL-lemma, Welch bound, contractions, etc. No analogies.
3. **Derive a guarantee** — theorem/lemma that makes the failure impossible or bounds it.
4. **Predict specific numbers** — the proof makes quantitative predictions.
5. **Pre-register kill criteria** — thresholds come from the proof, not arbitrary choice.

Three experiment types: **Verification** (proof complete, confirm predictions) · **Guided exploration** (proven framework, unknown parameter) · **Frontier extension** (extend a proven result; mark the gap).

---

## 3. Kill criteria & verdict rules

> The lean, de-grounded rules now live in **`.agents/method.md`** — that's the source the hats follow.
> This section is the longer reference; it does not override `method.md`, and no rule here is dogma.

**Prefer behavioral outcomes over proxies** (use judgment — a proxy moving without a behavioral change is
weak evidence, but there is no single mandatory rule, and the research stays open to whatever signal the
*question* needs). Don't straitjacket a hypothesis to a past finding.

**Kill-criteria discipline.** Pre-register your refutation threshold in `MATH.md` before the first run; if
the data crosses it, the verdict is `killed` — don't move the goalposts. Need a different criterion? Design a v2.

**Verdict consistency** (before `complete --status supported`, all must hold): `results.json["verdict"] != "KILLED"`; `all_pass` is True if present; `PAPER.md` verdict has no PROVISIONAL/PARTIAL/NOT SUPPORTED/INCONCLUSIVE/DEGENERATE; `is_smoke:true` runs complete as `provisional`, never supported/killed; no KC modified since MATH.md (git history); no `type: fix` antipattern applies.

**Finding status:** `conclusive` (proof verified, all predictions match) · `supported` (proof mostly verified / unknown narrowed) · `provisional` (empirical, awaiting proof) · `killed` (predictions refuted).

---

## 4. CLI reference

```bash
experiment list -s open                            # find work
experiment claim <worker>                          # pick next, get full YAML
experiment run <id>                                # run via pueue (MANDATORY — never bare uv run python)
experiment complete <id> --status supported \
  --dir experiments/models/<name>/ --k <kill-id>:pass --evidence "K1 PASS: val"
experiment finding-add --title "..." --status supported --result "..." \
  --caveat "..." --failure-mode "..." --impossibility-structure "..."
experiment query "search"                          # FTS across experiments + evidence + findings
experiment ref-add --arxiv <id> --title "..." --relevance "..."
```
Status values: `open | active | supported | killed | proven | provisional`. Full flag reference: load the `experiment` skill.

---

## 5. Platform rules (MLX / Apple Silicon)

- **Invoke `/mlx-dev` and `/fast-mlx` before writing platform code** — no exceptions, even smoke tests.
- **Memory safety:** phased execution — each compute phase in its own function with explicit cleanup between phases. `mx.clear_cache()` after loading large weights; `del` + `gc.collect()` between phases.
- **No PyTorch/CUDA on GPU.** If a library needs CUDA it's not usable here — find an MLX-native alternative.
- Cite the `mlx-lm` version in `MATH.md` (API changed across 0.21→0.31 and broke prior experiments).
- Base model, adapter recipe, and current focus: `PLAN.md` Part 2.

---

## 6. Antipattern catalog (`type: fix`)

Kept in `.ralph/agent/memories.md` (auto-injected in the loop). Current set: composition math bugs (`Σ Bᵢ@Aᵢ`, never `(ΣB)(ΣA)`), tautological routing, unsafe adapter scales (LORA_SCALE ≤ 8), KC-swap-after-failure, verdict-DB mismatch, smoke-as-full, tautological KCs, thinking-mode truncation (`enable_thinking=True` in train+eval), wrong-model proxy, synthetic padding, `shutil.copy` as new adapter, hardcoded `"pass": True`, file-existence cache, copy-paste scaffolding, dispatch-kill mislabel.

---

## 7. SIGREG reasoning chain (apply to every hypothesis)

- Are you treating symptoms or the disease?
- What structure makes the failure geometrically impossible?
- Derive from existing math, not analogy.
- Each eliminated hyperparameter is one understood degree of freedom.

Anchors: LeJEPA (`arxiv:2511.08544`), LeWorldModel (`arxiv:2603.19312`).

**Forbidden experiment classes:** information-theory analogies without LLM evidence; data-structure routing analogies (skip-lists, hash rings, cuckoo, bloom) unless paper-grounded for LLM/LoRA; mechanisms with no prior paper for LLM/LoRA use.

---

## 8. The 3-hat loop & where state lives

The conductor (`experiment start`) runs a **channel-driven** loop over this process — 🌶️ **Sparker** (novel mechanism for the active bet rung, or wildcat), 🔬 **Researcher** (author + async run → MATH/run/PAPER), 🔴 **Reviewer** (adversarial check → REVIEW, verdict routing), 🧠 **Analyst** (LEARNINGS + finding + PIERRE-IMPACT), 🚢 **Shipper** (supported bet findings → code on `../pierre`'s `bet/<name>` branch). Orchestration is the open-standard **`.agents/`** home: hats in `.agents/hats/*.md`, wiring in `.agents/conductor.md`, rules in `.agents/method.md`, **strategy in `.agents/bets/*.md`** (the rung ladders + gates), scaling in `.agents/fleet.md`. (`.claude/agents/*` symlink to the hats for Claude Code's loader.) Multiple conductors don't conflict — just run `experiment start` again in another tab: per-session channel ports, session-addressed `exp_done`, atomic claims.

| State | Lives in |
|---|---|
| Per-experiment record (authoritative) | experiment DB (Turso) via `experiment` CLI |
| Experiment files | `experiments/models/<name>/` (MATH/run/results/PAPER/REVIEW/LEARNINGS) |
| Bet ladders + gates (strategy) | `.agents/bets/*.md` |
| League table (branch competition) | `../pierre/LEAGUE.md` (scores also DB evidence, tag `league`) |
| Roadmap + platform | `PLAN.md` Part 2 |
| Verified status / source of truth | `STATUS.md` |
| Antipatterns | `.ralph/agent/memories.md` (`type: fix`) |
