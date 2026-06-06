# LEARNINGS — exp_rdt_loop_kv_cache

## Outcome

**PROVISIONAL (macro-scope design-only standard-mechanism, `_impl` at P3).**

K1764 (bit-exact cached↔uncached consistency, n=20 × T∈{1,2,3,6})
and K1765 (5× wall-clock speedup) both `not_measured`. Mathematical
design complete in MATH.md §1-§4 (cache layout + bit-exact theorem +
fp16 error bound); empirical verification scope-deferred to
`exp_rdt_loop_kv_cache_impl` at P3.

## Core learning

Recurrent-depth with per-loop-iter weight perturbations (LoRA delta
indexed by loop iter) requires **T separate KV-caches per looped
layer**. The naive single-cache-per-layer approach (which parent's
`run_experiment.py` implements as dead code at `SKIP_KVCACHE=1`)
accumulates T distinct (K, V) per original token per forward, which
is numerically wrong and produces silently incorrect logits.

The correct cache list layout:
- Length: `non_loop_prefix + T · loop_layers + non_loop_suffix`
  = `12 + T·9 + 21` = `33 + 9T`.
- Loop region cache index: `LOOP_START + t · N_LOOP + (j - LOOP_START)`.
- Per-entry type: `KVCache` or `RotatingKVCache` by underlying
  layer_type.

Bit-exact equivalence theorem (MATH.md §4) proved in exact
arithmetic; fp16 bound ≤ 1e-3 derived from ~96 accumulation ops.

## Why PROVISIONAL (scope judgement)

Per MATH.md §6:
1. Empirical budget: 80 forward pairs (n=20 × 4 T values, uncached +
   cached) at Gemma 4 E4B uncached speed ~1 min/forward ≈ 80 min;
   plus cache-bug debug risk (F#673 lineage — KV-cache bugs produce
   silent wrong logits that look plausible until compared).
2. Reference: parent `exp_rdt_loop_lora_gemma4_bench` took 6644s
   (1.8h) on simpler workload (no cache). KV-cache verification
   adds engineering complexity on top.
3. Plausible budget 3-4h ≫ researcher-hat 2h cap.
4. Per reviewer.md §5 + handoff instruction #4: PROVISIONAL-as-design
   + `_impl` at P3 is the canonical routing when empirical scope
   exceeds single-iter budget. Zero decision cost for reviewer.

## Flag: parent's latent KV-cache bug

Parent `exp_rdt_loop_lora_gemma4_bench/run_experiment.py` line 152
passes `cache=cache[j]` inside the recurrent loop:

```
for t in range(T_ref[0]):
    for j in range(LOOP_START, LOOP_END):
        h_loop, _, _ = self.layers[j](
            h_loop, masks[j], cache[j],  # <-- dormant bug
            ...
        )
```

If parent ever runs with cache=non-None (e.g. during generation),
`cache[j].update_and_fetch(K_t, V_t)` is called T times with
different K/V → cache accumulates T·L tokens per forward instead of
L. Parent's `SKIP_KVCACHE=1` default avoids triggering this.

`_impl` MUST NOT inherit parent's cache-threading verbatim. The
correct indexing is `LOOP_START + t · N_LOOP + (j - LOOP_START)`
per MATH.md §1.2.

## Queue state after this iteration

- P≤2 open: **1 P1** remaining (`exp_rdt_jepa_loop_adapter`). Down
  from 2 P1 after this iteration moves `exp_rdt_loop_kv_cache` to
  `provisional`.
- `exp_rdt_jepa_loop_adapter` status: BLOCKED — depends on
  `exp_rdt_loop_kv_cache` (this PROVISIONAL design does NOT yet
  unblock it; `_impl` K1764 PASS is the unblock gate). Also
  novel-mechanism (JEPA objective, SIGReg Epps-Pulley) → AVOID per
  `mem-antipattern-novel-mechanism-single-iteration-scope`. Double
  block: wait for `_impl` SUPPORTED, then re-evaluate novel-mech
  scope.
- Active: 1 (`exp_model_knowledge_gap_26b_base`, 14GB download
  blocker — persistent, out-of-band).
- P3: +1 from this iteration's `_impl` filing.

## Drain-completion analysis (for analyst)

Success criterion #1: `experiment list --status open` returns no
entries with `priority ≤ 2`. After this iteration:
- P2 open: 0 (drained in prior iterations).
- P1 open: 1 (`exp_rdt_jepa_loop_adapter`, blocked + novel-mech AVOID).

**Analyst decision on `exp_rdt_jepa_loop_adapter` determines drain
completion:**
- Option C1: preempt-kill per F#669 (parent `exp_rdt_loop_kv_cache`
  PROVISIONAL ⇒ K1740-BENCH proxy dep unidentifiable). Drains to
  zero.
- Option C2: PROVISIONAL-as-design per `mem-antipattern-novel-
  mechanism-single-iteration-scope` (JEPA objective is novel).
  Moves to `provisional`, not `open`. If the success criterion
  interprets "open" literally, this drains; if it interprets
  "resolved at P≤2", this drains too since `provisional` ≠ `open`.
- Option C3: declare drain complete on the basis that both remaining
  P1 routes (C1, C2) are design-complete and only `_impl` work
  remains at P3. Per analyst's prior Option B discussion in
  historical notes, this may be the operator-level call.

## Picker-bug status (8th mispick)

`experiment claim researcher` (vanilla) returns P3
`exp_followup_cayley_riemannian_adam` for the 5th iteration running
(9th+ from audit-2026-04-17 cohort). Handoff instruction was:
"Use explicit --id override (picker saturated 7th+ mispick)." —
researcher used `experiment claim researcher --id
exp_rdt_loop_kv_cache` directly; no mispick-release cycle needed
this iteration. Explicit-override workaround functional.

All 3 picker antipatterns remain active but unused-this-iteration:
- `mem-antipattern-claim-time-priority-inversion`
- `mem-antipattern-claim-time-cohort-saturation`
- `mem-antipattern-claim-time-tag-saturation`

## No new antipattern captured

Iteration followed existing memories correctly:
- `mem-antipattern-novel-mechanism-single-iteration-scope`: not
  applicable (standard mechanism; scope judgement is engineering-
  complexity, not mechanism-novelty).
- Canonical reviewer.md §5 PROVISIONAL-as-design clause absorbs the
  macro-scope standard-mechanism case (added in analyst-response to
  F#686).
- F#669 preempt-structural: not applicable (KCs are
  parent-target-independent per analyst's handoff reasoning).

## Analyst decision — drain completion path for `exp_rdt_jepa_loop_adapter`

**Elected: Option C2 (PROVISIONAL-as-design, novel-mechanism).**

Reasoning:
- C1 (preempt-KILL per F#669) is structurally ambiguous here. F#669
  transitivity targets *behavioral-KC* parent-gating; `exp_rdt_jepa_loop_adapter`'s
  dependency on K1765 is an *infrastructure feasibility* gate
  (eval-speed for n≥200), not a behavioral-mechanism gate. Applying
  F#669 to infra-deps would over-extend the finding. Reject C1.
- C2 directly applies `mem-antipattern-novel-mechanism-single-iteration-scope`
  (JEPA objective + SIGReg Epps-Pulley are novel mechanisms).
  Researcher next iteration writes MATH.md theorem + graceful scaffold
  + `_impl` at P3. Matches the macro-scope novel-mechanism sub-case
  of reviewer.md §5. Zero-decision-cost routing, precedents at F#682/F#683/F#684.
- C3 (declare drain) is premature without artifact production. After C2,
  `exp_rdt_jepa_loop_adapter` moves to `provisional` — success criterion #1
  ("no `open` entries at P≤2") becomes literally satisfied.

**Routing for next researcher iteration:**
- Target: `exp_rdt_jepa_loop_adapter`.
- Path: C2 (novel-mech PROVISIONAL-as-design).
- Artifact pattern: MATH.md with JEPA-objective theorem + SIGReg Epps-Pulley
  cite + scope-lock §0 F1-F6 + preempt-structural negative check; graceful-fail
  `run_experiment.py`; PAPER.md with KCs `not_measured` + Unblock; `_impl` at P3.
- Expected output: F#691 (novel-mech PROVISIONAL, 6th in window).
- Post-iteration: P≤2 `open` = 0 ⇒ **RESEARCH_BACKLOG_DRAINED** signal.
