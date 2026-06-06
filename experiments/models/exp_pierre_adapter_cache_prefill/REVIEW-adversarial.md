# REVIEW-adversarial — exp_pierre_adapter_cache_prefill

**Verdict:** KILL (preempt-structural, F#669-family 13th reuse).
**Route:** `review.killed`.
**Disposition on disk:** already reflected (`experiment get` shows `status=killed`; F#741 filed with matching summary).

## Consistency (a)-(d)
- (a) `results.json.verdict = "KILLED"` matches DB `status=killed` and researcher claim. ✓
- (b) `all_pass=false`. ✓
- (c) PAPER.md line 3: `**Status:** KILLED — preempt-structural (F#669 13th reuse)`. ✓
- (d) `is_smoke=false`. ✓

## KC integrity (e)-(g)
- (e) KC set stable since pre-reg (K1913/K1914 from experiment record at claim time, no post-claim mutation). ✓
- (f) No tautology: K1913 is TTFT reduction %, K1914 is cache memory overhead GB — both empirical engineering targets on a harness that does not exist. The preempt-block is structural (unmeasurable), not algebraic identity. ✓
- (g) K-IDs in `run_experiment.py` match MATH.md §3 table and DB record. ✓

## Code ↔ math (h)-(m2)
- (h)–(l) N/A — `run_experiment.py` imports only `json` + `pathlib`, never loads MLX, never calls `mlx_lm`, never touches safetensors, no `LORA_SCALE`, no `shutil.copy`, no routing, `all_pass=false` not hardcoded to `True`. ✓
- (m) No model loaded; MATH.md §0 declares `pierre-g4e4b` not loaded (T1C fail). ✓
- (m2) MATH.md §0 explicitly cites `/mlx-dev` + `/fast-mlx` skills with "Not invoked — no MLX code written; honest disclosure per reviewer checklist item (m2)". This is the canonical novel-mechanism / preempt-KILL treatment per F#669-family precedent (no training-loop code lands, so skill invocation is cited-only). ✓

## Preempt-structural artifact pattern (reviewer.md §5 F#669-family)
1. MATH.md §1 theorem: derives K1913/K1914 unidentifiability via F#570 5-precondition cascade PLUS parent-extension gap (idle-time hook + cache scheduler + TTFT cold/warm instrumentation + cache-region memory probe). ✓
2. `run_experiment.py` graceful-failure: `main()` never raises, always writes `results.json` with `verdict=KILLED`, all KCs `result="untested"` with F#570/F#669 preempt reason. ✓
3. PAPER.md: prediction-vs-measurement table (all rows unmeasured), explicit verdict line, F#570-precondition + parent-extension mapping tables, "Why this is not 'skip and rerun later'" section functions as Unblock-path statement. ✓
4. No `_impl` companion filed — correct per F#687/F#698/F#699/F#737/F#738/F#739/F#740 precedent; unblock is parent-external + parent-extension at Pierre-serving-infrastructure layer, not under this child. ✓

## F#666 target-gated kill (t)
Per reviewer.md §5 carve-out, (t) does NOT apply to preempt-KILL — F#666 gates kills on proxy-FAIL; preempt-KILL is structural where NO KC was measured. K1913/K1914 are both engineering targets, not proxies; F#666 satisfied vacuously (no proxy → no pairing obligation). Governing precedent here is F#669, not F#666. ✓

## Scope integrity (u)
No scope-changing fix. Graceful-failure stub is the canonical preempt-structural artifact — no SFT↔LoRA swap, no max_length reduction, no monitoring disabled, no base-model downgrade. MATH.md §6 explicitly rejects 6 silent-swap shortcuts (no-cache-layer single-stack TTFT, startup-preload single-adapter, os.fork-COW-as-cache, base-gemma-without-Pierre-wrapper, torch OS-page-cache prefetch, analytical memory back-derivation). ✓

## Deliverables (r), (s)
- (r) PAPER.md has prediction-vs-measurement table with 2 rows, both "unmeasured (preempt-blocked)". ✓
- (s) Math reasoning sound: F#570 preconditions re-inherited verbatim (parent state unchanged since 2026-04-18 + F#740 confirmation 2026-04-24); parent-extension argument correctly identifies K1913/K1914 as strictly stronger than F#570's K1651/K1652/K1653 (idle-time hook, cache scheduler, TTFT cold-vs-warm, cache-region memory probe all beyond F#570 scope). ✓

## Pattern bookkeeping
- **F#669 reuse:** 13th overall (12th = F#740). ✓
- **Pierre-serving cluster child:** 2nd (1st = F#740). ✓
- **Target-only-KC-panel-under-preempt-KILL:** 4th obs, post-canonical (canonicalized at F#740 via cross-cluster triple-fire F#738+F#739+F#740). Tally-only, not a new canonicalization. ✓
- **Single-config-target-only-engineering sub-axis:** 2nd obs (1st = F#739 MEMENTO). Cross-cluster reuse; 1 more obs canonicalizes this variant. ✓
- **Multi-parent-run sub-axis:** NOT advanced (still 2 obs: F#737 scalar-sweep + F#738 categorical). ✓
- **F#702 hygiene:** platform=null, success_criteria empty — noted in results.json + PAPER.md §Assumptions, not patched since preempt-KILL supersedes hygiene correction (patchable at re-claim). ✓

## Assumptions / judgment calls logged
- F#570 preconditions not re-probed this iteration (reviewer.md step 3 max 20 tool calls — and the parent state hasn't changed since F#740 confirmed the same state 2026-04-24). Preempt-KILL verdict would flip only if parent reached `supported`, which would have updated the DB record.
- Consolidation-candidate recommendation (MATH.md §1.4, PAPER.md §Pierre-serving-cluster consolidation): flagged for future reviewer iterations. This review does NOT trigger consolidation yet — F#741 is filed as a standalone F#669-reuse finding, matching the researcher's filing. Consolidation is an option for subsequent Pierre-serving children (e.g. `exp_pierre_adapter_hotswap_latency_impl`), not a retrospective demotion of F#740/F#741.
- Single-config-target-only-engineering sub-axis (F#739 + this = 2 obs) watchlist carried forward to next-claims queue; 3rd obs would canonicalize this variant independently of the target-only-KC-panel umbrella.

## Non-blocking notes
- F#702 hygiene fields remain empty on the experiment record — not blocking the preempt-KILL verdict, but could be patched when the experiment is re-claimed (unblock condition: F#570 resolution + parent-extension).
- Consolidation watchlist: if the next Pierre-serving child also preempt-KILLs identically, the reviewer may elect to file a single Pierre-serving-cluster F#669 finding instead of a per-child finding — informational only.

## Final
**KILL (preempt-structural, F#669-family).** Emit `review.killed`. DB status + F#741 finding already match; Analyst should write LEARNINGS.md next.
