# STATUS — verified research checkpoint

> **2026-06-03 checkpoint.** Single source of truth, reconciled against the experiment DB frontier (highest finding IDs win). When this disagrees with any archived doc, this is correct. Companion to `PLAN.md` (Part 1 = stable framework discipline; Part 2 = live roadmap). The marketing docs that drifted a thesis ahead of the data are in `docs/archive/2026-06-03-superseded/`.

---

## 1. One-paragraph ground truth

This repo is an MLX-native, Apple-Silicon research program (937 experiments, 845 findings, 54% kill rate) testing whether composing LoRA adapters on a frozen Gemma-4-E4B base can specialize a small model per-task without interference. The **infrastructure is real and good** (a 265-LOC MLX composition core, <1ms adapter hot-swap, an honest experiment-tracking DB). The **grand product thesis is not** — "strategies transfer cross-domain, knowledge doesn't, and orthogonal composition beats frontier" was **refuted by the team's own May-2026 experiments**. What actually survives is modest and real: a **static K=7 Fisher-Rao / Pico+TIES composition ≈ 64–68% avg** on GSM8K/HumanEval/MedQA, roughly **+2–4pp over the 4B base**. Solo single-domain adapters give large on-domain lifts (+22–62pp) but **interfere destructively off-domain** (−12 to −14pp). The productization repo (`../pierre`) is a solid framework that currently shows **Pierre == base** in every measured comparison.

---

## 2. Finding-spine reconciliation

The load-bearing findings the product docs cite, each reconciled against the DB frontier. Verdict = SURVIVES / WEAKENED / REFUTED / NEVER-RAN.

### ✅ SURVIVES — build on these
| F# | Claim | Why it holds |
|---|---|---|
| **275 / 677** | NRE (norm-rescaled) = Fisher-Rao Karcher mean; norm preservation IS the composition mechanism | Confirmed at production scale (N=25, Gemma-4-E4B); FR costs 68× for no benefit. The one composition principle that scaled. |
| **627** | Solo single-domain LoRA: GSM8K +22pp, HumanEval +48pp, MedQA +62pp | Robust substrate; F#827 re-uses these adapters and reconfirms on-domain lift. Read it as solo-only — it does **not** license free composition. |
| **766** | <1ms adapter hot-swap on Gemma-4-E4B (0.97ms median, bitwise-identical) | Actually executed on the target model, tight variance, self-corrected parent errors. Pure latency claim, orthogonal to the interference kills. |
| **455** | Multi-tenant KV sharing = 8× memory reduction, bit-exact | Algebraic identity (no k_proj adapter ⇒ K is adapter-independent). Cannot fail unless a k_proj adapter is added. |
| **345** | M2P centroid-trap impossibility proof (conclusive) | A *kill* that survives: homogenized B-matrix forces parameter collision; Grassmannian A cannot protect B. Repeatedly invoked, never refuted. |
| **836 / 841** | Static K=7 Fisher-Rao = 64.7%; Pico+TIES = 68.0% avg | The surviving composition result. Relative comparison valid; absolute lift-over-base is soft (baseline eval unreliable). |

### ⚠️ WEAKENED — real measurement, over-extrapolated in the docs
F#248/250 (scale "phase transition" is math-only — F#251/252 killed the general framing) · F#262 (NTP>SFT reasoning gap real but n=50, no formal test, single condition) · F#292 (86.8 tok/s real, but attn-only craters code −67%) · F#304 (per-domain module recipe thin; undercut by F#827) · F#458 (98.8% routing real but orthogonal to the actual blocker — F#847 shows routing can't rescue composition) · F#466 (domain-conditional retrain works but costs −capability: "Math MCQ at 10%") · F#510 (the *negative* half — pre-merge=0% — stands; the "Grassmannian orthogonality is the fix" *spin* is dead per F#822/823) · F#536 (MMLU-Pro 62.1% baseline holds; adapter-suppresses-thinking is single-adapter) · F#747 (layer-skip rule from one domain/one ε) · F#752 (τ≈0.48 cross-term real; "bounded/benign" framing dead).

### ❌ REFUTED — the docs' headline pillars, killed at scale
| F# | Cited as | Killed by |
|---|---|---|
| **203** | "Wrong adapter keeps 87% benefit" (cheap routing OK) | F#827 — cross-domain interference is real and behavioral (PPL-only caveat materialized) |
| **204** | "Code adapter helps math 10%→70%" (strategies transfer) | F#844/827/837 — single-seed n=10, scale=20 unablated; strategy adapters are domain-entangled |
| **250** | Sharp LoRA scale phase transition | F#251/252 — math/format-specific artifact of binary eval |
| **362 / 364** | "M2P 99.6% one-shot personalization" | F#345 (centroid-trap proof) + F#820 + F#486 (cos=0.9986, all docs → identical LoRA, QA F1=0.6%); toy d=1024, sort+reverse only |
| **428 / 440** | "N=25/100 orthogonal composition, cos=2e-8" | F#822/823 — orthogonality benefit is noise (0.0018 @ 22 layers); 20/25 adapters were B=0 synthetic + exclusive routing (never tested real composition) |
| **508** | "E2E system +19–56pp" | Solo-only by its own caveat; sister F#510 shows composed = 0%/0%/20% |
| **830** | "Uniform 1/N is optimal composition" | F#836/841 — Fisher-Rao/Pico+TIES beat it by +3–4pp |

### ⛔ NEVER-RAN — cited as "design proven", zero measurement
F#683 / F#684 (**Hedgehog** strategy/procedural distillation) and F#685 (**MEMENTO** compression) are design-only `provisional` records — `NotImplementedError`, "all untested", training loops never executed (exceeded the loop's 30-min cap). `ARCHITECTURE.md`/`GAMEPLAN.md` cite them as validated. They are not.

---

## 3. The survivor architecture (what is actually true)

```
Frozen Gemma-4-E4B-4bit  (the knowledge; never modified)
        │
        ├─ Solo domain adapter (r=6, q_proj → migrate to v_proj+o_proj per F#627)
        │     → large ON-domain lift (+22..+62pp)   ✅ works
        │     → destructive OFF-domain interference (−12..−14pp)   ⚠️ real (F#827/837)
        │
        └─ Composition = static K=7 Fisher-Rao / Pico+TIES merge   ✅ ~64–68%, +2–4pp
              ├─ NOT per-prompt routing (F#847 kills it)
              ├─ NOT uniform-1/N optimal (F#830 superseded)
              └─ NOT Grassmannian orthogonality (F#822/823 kill the benefit)

Serving primitives that work: <1ms hot-swap (F#766), 8× KV share (F#455), NRE=FR merge (F#275).
Dead/never-built: M2P personalization (F#345 proof), MEMENTO (F#685), Hedgehog (F#683/684).
```

**Core invariants** (do not re-litigate): composition math is `Σ B_i @ A_i`, never `(ΣB)(ΣA)`; LORA_SCALE ≤ 8; route per-sample not per-domain; `enable_thinking=True` in train + eval; PPL does **not** predict task quality (r≈0.08) — every claim needs a behavioral target-metric KC (Target-Gated Kill Rule, F#666, `PLAN.md §1`).

**Base-model identity must be fixed** (productization blocker): `../pierre` references three different bases — declared `gemma-4-e4b-it-4bit`, but every trained adapter is on `gemma-4-e2b-it-4bit`, while tool-use scripts target `gemma-4-26b-a4b-it-4bit`. Pick one canonical base; make manifests + training scripts + parity test agree.

---

## 4. Productization status (`../pierre`)

- **Real & good:** a hand-written MLX Gemma-4 reimplementation (1,472 LOC) with a 5-tier hook taxonomy, parity-tested vs mlx-lm; **271 tests pass**; an OpenAI-compatible server; a unit-tested NRE/TIES compose library; an honest layered eval framework (`docs/EVAL.md`, `docs/FEATURES.md`).
- **The gap:** every measured comparison shows **Pierre == base** (`comparison_results.json` delta 0.0 on all 11 prompts). Both terminal-bench runs scored 0.0 and **never completed a single trial** (3/89, all errored). Only **toy q_proj/layer-0 adapters** exist. The composition engine is **not in the serving path**. The router is a keyword placeholder. The "71.3% avg" number was measured in *this* repo, not the product.
- **One thing must exist before anything ships:** a single adapter that, served, beats raw Gemma on one real task, with a working eval harness. It does not exist yet.

---

## 5. Candidate roads (grounded in survivors, not the dead thesis)

Each is a hypothesis to verify, not a commitment. Pick one per cycle; pre-register a behavioral target-metric KC.

1. **Single-domain hot-swap router** (lowest risk, ship-shaped). The robust facts — big solo lift (F#627) + <1ms hot-swap (F#766) + 8× KV share (F#455) — already make a *fast per-query single-adapter specialization* product, **no composition required**. This is the honest MVP and sidesteps every kill.
2. **Make K=7 static composition reproducible & real.** The surviving +2–4pp (F#836/841) was never multi-seed or behaviorally clean. Lock it down: multi-seed, behavioral eval, honest base. If it holds, it's a modest but defensible composition story.
3. **Characterize interference instead of denying it.** F#827 found *both* destructive (−14pp) and surprising *positive* transfers (python→MedQA +50pp). A predictive model of which adapter pairs help vs hurt is a genuine open research question the data supports.
4. **Migrate adapters q_proj → v_proj+o_proj (F#627 targets)** and re-test whether the better target set reduces the off-domain interference that q_proj adapters cause.

Explicitly **closed** (do not reopen): M2P/MEMENTO/Hedgehog; Grassmannian-orthogonality-as-the-fix; spectral/energy/Gini surgery (NRE is the ceiling); per-prompt routing as a composition rescue; "strategies are domain-agnostic."

---

## 6. Forward process — replacing the unattended loop

The root cause of the doc/DB drift was `ralph.yml`: an autonomous 3-hat Claude loop (max 1000 iterations / 24h, *"NEVER wait for user input, always pick and proceed"*) that generated experiments faster than any human reconciled them. Replacement is **human-in-the-loop checkpoints**, not more autonomy.

**The cycle (one road at a time):**
```
1. VERIFY   reproduce the source arxiv result for ONE load-bearing finding (multi-seed, behavioral)
2. INTEGRATE wire the verified result into the product as ONE experiment with a behavioral target KC
3. MEASURE  run it; does it beat base on a real task? (no PPL-only claims)
4. CHECKPOINT update this STATUS.md + the changelog below; THEN choose the next road (human gate)
```

**Rules of the new process:**
- **DB frontier is authoritative.** Before repeating any claim, check whether a higher-numbered finding killed it. Never cite a finding without its caveats.
- **One central truth doc.** Status lives here + `PLAN.md`. No new `VISION_*.md` forks. When direction shifts, edit `PLAN.md` Part 2 and append a checkpoint entry below.
- **Reproduce *our own* spine, not the arxiv corpus.** The arxiv papers are sound; the failure was over-extrapolation from ~15 thin internal findings. Verify those.
- **Behavioral target-metric KC required** on every experiment (`PLAN.md §1`, F#666). Proxy-only (PPL/cosine/routing-accuracy) cannot support a claim.
- **A human gates each checkpoint.** The loop may *propose*; it does not get to declare Phase-N "COMPLETE."

---

## 7. Checkpoint changelog

> Append one entry per checkpoint. Newest first. This is the project paper-trail.

### 2026-06-06 — Checkpoint 2: doc consolidation + knowledge-graph audit
- **Ran `graphify`** (local, AST-only, $0) over the repo → 83,374-node graph. Diagnosis: the file sprawl is the experiment layer's copy-paste, not the science — `log()`/`cleanup()`/`log_memory()` are redefined in 350+ experiments each (~1,475 duplicate helper defs, none in the shared lib); `generate_grassmannian_A` reimplemented 24×, `compose_adapters` 43×; `.agents/skills/notebooklm` is a 30 MB vendored package. (Code dedup deferred by choice — records left frozen.)
- **Centralized the experiment process into one canonical doc**: created **`experiments/GUIDE.md`** (the loop, proof-first discipline, target-gated kill rule, verdict consistency, CLI, platform rules, antipatterns, 3-hat model) by consolidating `AGENTS.md` + `docs/EXPERIMENT.md` + `docs/PROMPT.md` + `PLAN.md` Part 1. Deleted `docs/EXPERIMENT.md` + `docs/PROMPT.md`; slimmed `AGENTS.md` to an entry-pointer and `PLAN.md` Part 1 to a pointer.
- **Organized `docs/`**: `guides/` (MLX, adapters) · `research/` (notes) · `references/` (papers) · `archive/` · `assets/`. README now has a "where to look" table.

### 2026-06-04 — Checkpoint 1: repo restructure to 5 root folders
- **Collapsed ~20 root folders → 5**: `pierre/` (core + `merge/` = the 5 merge libs), `experiments/` (was `micro/`→`models/` + `macro/` + `_runs/`), `tooling/` (was `packages/` + `scripts/` + `tools/`), `data/` (`adapters/` + `corpora/`), `docs/` (+ `references/`, `archive/`, `assets/`). Root `.md` kept minimal: README, STATUS, PLAN, AGENTS.
- **Depth-preserving rename** `micro/`→`experiments/` kept the 735 experiments at the same depth, so the 145 files that compute `REPO_ROOT` by counting parents stayed valid. `macro/` nested under `experiments/macro/` (depth +1; its handful of repo-root refs fixed).
- **Coordinated disk move + DB rewrite**: bulk-updated 887 `experiment_dir` rows in Turso (`micro/…`→`experiments/…`, `macro/…`→`experiments/macro/…`) so `experiment run` still resolves. Rewrote 188 `from micro.`→`from experiments.` imports, 25 `from scripts.`→`from tooling.scripts.`, ~80 `REPO_ROOT/"adapters"`→`/"data"/"adapters"`, registry.json paths, and pnpm/global-bin links. Named the framework folder `tooling/` (not `platform/`) to avoid shadowing Python's stdlib `platform`.
- **Verified**: `experiment` CLI resolves new paths; all shared-lib + core imports load; 427 edited `.py` files compile clean; 0 residual broken refs.
- Full move map + rationale: this changelog + `experiments/README.md`.

### 2026-06-03 — Checkpoint 0: reconciliation & cleanup
- **Audited** 937 experiments / 845 findings against the product docs. Found the docs drifted a full thesis ahead of the DB.
- **Reconciled** the finding spine (§2): 8 SURVIVE, ~10 WEAKENED, 8 REFUTED, 3 NEVER-RAN. The strategy-transfer / orthogonal-composition thesis is **refuted** by F#827/837/844/822/823.
- **Established ground truth:** real ceiling is K=7 static Fisher-Rao ≈64–68% (+2–4pp over base); solo adapters lift on-domain but interfere off-domain; M2P/MEMENTO/Hedgehog are dead or never-run.
- **Cleaned the tracked surface:** archived 4 contradictory marketing docs + orphaned mindmap → `docs/archive/2026-06-03-superseded/`; archived dead torch/vLLM `composer/` → `docs/archive/composer/` (dropped from `pyproject.toml`); removed stale `compose_registry.json` (wrong base model) and fake `macro/api.py` stub. The honest research log (the experiment dirs + the DB) is **frozen, not touched**.
- **Confirmed productization gap:** `../pierre` is a solid framework with Pierre == base in every measurement; base-model identity is inconsistent and must be fixed.
- **Next gate:** choose one Road from §5 (recommended: Road 1, the single-domain hot-swap MVP) and run one VERIFY→INTEGRATE→MEASURE cycle.
