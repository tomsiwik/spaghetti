# PAPER.md — exp_hedgehog_adapter_rust_domain

**Verdict: PROVISIONAL (design-only; KCs K1866/K1867 untested — implementation deferred to `exp_hedgehog_adapter_rust_domain_impl`)**

## Claim

Per-layer cos-sim distillation from a Gemma 4 26B teacher with Rust Book + rustonomicon +
selected RFC excerpts in context produces a rank-8 LoRA adapter on `(v_proj, o_proj)` of
Gemma 4 E4B that encodes Rust domain knowledge (ownership/move semantics, borrow-checker/
lifetimes, traits+trait-objects, iterators+closures, unsafe/FFI, error-handling via
`Result`/`Option`+`?`, macros, zero-cost abstractions) as an attention-routing
perturbation. The adapter is predicted to (a) achieve PPL ≤ base + generic token-space
LoRA on a held-out Rust eval set (K1866) and (b) uplift Rust idiomaticity auto-judge by
≥ +5 pp vs base (K1867). Third Hedgehog-axis domain experiment after JS (F#696) and
Python (F#697), selected to test whether cos-sim distillation transfers beyond scripting
languages to a systems language whose compositional hardness lies in static
borrow-graph reasoning (aliasing-XOR-mutation) rather than dynamic-typing nuance.

## Scope (this iteration)

This iteration executes **design-only** per sibling precedent
(`exp_hedgehog_adapter_python_domain` F#697 + `exp_hedgehog_domain_adapter_js` F#696 +
`exp_hedgehog_behavior_adapter_politeness` F#683 +
`exp_hedgehog_procedural_adapter_refactor` F#684). The scaffold in
`run_experiment.py` loads `mlx.core`, logs memory, writes `results.json`, and raises
`NotImplementedError` in the five phases that require the 4–6 h custom MLX training
loop (Phase 0 Rust corpus curation; Phase A/B teacher capture + per-layer cos-sim
distillation; Phase Baseline generic token-space LoRA matched-params; Phase C K1866
PPL head-to-head; Phase D K1867 idiomaticity judge including `cargo check`
compile-check hard-floor).

A dedicated `_impl` follow-up experiment (`exp_hedgehog_adapter_rust_domain_impl`,
P=3) is filed inline this iteration per
`mem-antipattern-impl-follow-up-delegation` remedy. K-IDs K1866/K1867 inherit verbatim
into the `_impl` row (no renumbering; DB issues new parallel KC-IDs that point to the
same canonical text).

## Prediction vs measurement

| KC | Prediction | Kill condition (KILL if TRUE) | Measurement (this iter) |
|---|---|---|---|
| K1866 proxy PPL | `PPL(Hedgehog) / PPL(base + generic LoRA) ∈ [0.95, 1.02]` — PASS expected | `PPL(Hedgehog) > PPL(base + generic LoRA)` strictly | not measured (Phase B + Baseline not implemented) |
| K1867 target idiomaticity | `Δ ∈ [+4, +9] pp` vs base; mean +6 pp — PASS expected | `Δ < +5 pp` vs base | not measured (Phase B not implemented) |

Both KCs locked pre-run; no post-hoc relaxation. Verdict is PROVISIONAL because
nothing was measured — design fidelity only.

## Why K1867 sets a slightly lower predicted mean than Python sibling

Python sibling (F#697) predicted idiomaticity Δ ∈ [+5, +10] pp with mean +7 pp. Rust
prediction is Δ ∈ [+4, +9] pp with mean +6 pp. The 1 pp lower prediction is load-bearing
reasoning, not analytical noise:

- Rust idiomaticity subsumes *borrow-checker-correctness* (aliasing-XOR-mutation), which
  is a structural rather than surface-lexical property. Per-layer cos-sim on attention
  outputs carries surface-routing signal well (Zhang 2402.04347 99 %) but may fail to
  fully transfer graph-reasoning over borrow-lifetimes at rank 8.
- Python idiomaticity (list-vs-generator, context-managers, decorators) is more nearly
  surface-choice; the teacher-in-context ceiling and distilled floor are both higher.
- If the predicted 1 pp gap holds and K1867 measures +6 pp for Rust vs +7 pp for
  Python, this is an *axis-specificity* finding about which cos-sim captures
  efficiently — not a kill.

## Scope-preservation explicit rejections (antipattern-t)

The following "silent downscales" are explicitly out of scope and would be treated as
REVISE-blocking antipatterns if attempted in `_impl`:

- **Teacher proxy.** Substituting Gemma 4 E4B for the 26B teacher would erase the
  teacher-with-docs gap that K1866/K1867 measure. Forbidden.
- **CE swap.** Swapping per-layer cos-sim for cross-entropy next-token SFT would test a
  different hypothesis (surface imitation of A tokens). Not a valid fallback — file
  PROVISIONAL instead.
- **Baseline skip.** K1866 is *head-to-head*: `PPL(Hedgehog) vs PPL(base + generic
  LoRA)`. Skipping the baseline would leave K1866 with no comparator.
- **N_STEPS reduction without SMOKE_TEST flag.** Reducing from 800 to 200 without
  setting `IS_SMOKE=True` would produce a silently-underconverged result reported as
  "full N."
- **Dropping `cargo check` hard-floor from K1867 judge rubric.** Compile-fail code
  scoring non-zero on the correctness axis would decouple idiomaticity from Rust
  semantic validity — the judge would reward "stylish but wrong" code. Load-bearing.

## Measurement blockers (to resolve in `_impl`)

1. **Phase 0 dataset curation** — Rust Book + nomicon + RFC corpus, 200 train + 50
   held-out (Q, A) pairs stratified by 8 focus topics, external crates.io PPL eval
   corpus (tokio/serde/clap/rayon function bodies, disjoint from training).
2. **Phase A teacher capture** — 26B Gemma 4 + π_Rs in context, capture `{layer_idx:
   attn_output}` for all 42 layers. Peak-memory load-bearing on 48 GB (F#673);
   sequential-phase eviction or offline precompute-to-disk.
3. **Phase B student training** — custom MLX training loop with per-layer attention-
   output hooks, `nn.value_and_grad + AdamW`, `mx.eval + mx.clear_cache` between
   batches. Not available via `mlx_lm.lora` CLI.
4. **Phase Baseline** — generic token-space LoRA via `mlx_lm.lora` CLI at matched
   rank/targets/scale/steps. Runs but deferred to keep K1866 arms paired.
5. **Phase C K1866 PPL** — three-configuration PPL eval (base, base+gen-LoRA, base+Hedgehog).
6. **Phase D K1867 judge** — blind-paired 50-prompt judge including `cargo check`
   compile-success hard-floor. Requires a Rust toolchain on the eval machine.

Shared blocker: **26B Gemma 4 teacher model not yet cached (~14 GB)** — common to all
four Hedgehog-axis `_impl` follow-ups + `exp_model_knowledge_gap_26b_base`.

## Assumptions (from MATH.md §8, restated for paper context)

A1 teacher-with-docs > 4B-alone gap exists (spot-check validation required).
A2 Rust Book + nomicon + RFC per-topic excerpts fit in 128 k teacher context.
A3 PPL on external crates.io slice is non-contaminated by Rust Book training text.
A4 50 blind-paired judge pairs detect +5 pp at α=0.05 (MDE ~ +3 pp).
A5 single-iteration cap (30 min / 40 tool calls) — full pipeline out of scope here.
A6 `LORA_SCALE ≤ 8` per F#328/F#330; using 6.0.
A7 only 2 KCs pre-registered (same as Python sibling; JS sibling had 4).
A8 `cargo check` compile-success is a hard behavioral signal for Rust — used as judge
  correctness-axis hard-floor. This is strictly stricter than JS/Python siblings.
A9 F#702 hygiene-patch is APPLICABLE: experiment row shipped with 3 hygiene defects
  (success_criteria=[], platform=~, references=[]) but K1867 is a target KC, so
  `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does NOT fire.
  Hygiene patch applied via DB update (platform set, success_criteria + references
  added before `experiment complete`).

## Sibling-axis position

This is the **5th Hedgehog-axis PROVISIONAL**:

| # | Finding | Axis | Topics | KC count |
|---|---|---|---|---|
| 1 | F#683 | politeness | behavioral (formal↔informal register) | 4 |
| 2 | F#684 | procedural refactor | procedural (refactor-trace reasoning) | 4 |
| 3 | F#696 | JS domain | domain (JavaScript language nuance) | 4 |
| 4 | F#697 | Python domain | domain (Python language nuance) | 2 |
| 5 | **this** | **Rust domain** | **domain (Rust systems language)** | **2** |

Sibling pattern at 5 instances is candidate for "novel-mechanism PROVISIONAL at
Hedgehog-axis" classification — same mechanism, different axis. The 5th instance is
the one that pushes us from "confirmed-recurrent" (≥ 3) toward bucket-saturation
territory. Whether that triggers a taxonomy refactor (per F#711 convention for
F#666-pure bucket saturation) is the analyst's call; the researcher view is that 4
domain-axis instances across JS/Python/Rust/SQL-pending (the 5th domain-axis sibling
`exp_hedgehog_adapter_sql_domain` remains open at P=2) is within "confirmed-recurrent"
range for novel-mechanism replication, and a refactor is not yet motivated.

## References

- Moudgil et al., Hedgehog attention distillation, arxiv:2604.14191 §3.1 eq. 6.
- Zhang et al., cosine-loss attention recovery, arxiv:2402.04347.
- Cassano et al., MultiPL-E (Rust supported), arxiv:2208.08227.
- Pierre F#627 (v_proj+o_proj LoRA sufficiency); F#614/F#536 (thinking-mode load-
  bearing); F#328/F#330 (LORA_SCALE ≤ 8); F#673 (mx.clear_cache between phases,
  MLX audit 2026-04-17).
- F#666 target-gating convention; F#702 hygiene-patch PROVISIONAL; F#683/F#684/
  F#696/F#697 Hedgehog-axis PROVISIONAL precedents.
- `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — inapplicable here
  (target KC present).

## Handoff

- Status: PROVISIONAL.
- `_impl` follow-up: `exp_hedgehog_adapter_rust_domain_impl` filed inline at P=3, KCs
  inherited verbatim.
- Hygiene-patch applied: platform set to "M5 Pro 48GB MLX", success_criteria +
  references populated before `experiment complete`.
- No reviewer-side `_impl` filing required (researcher-filed per
  `mem-antipattern-impl-follow-up-delegation`).
