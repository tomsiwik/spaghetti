# REVIEW-adversarial.md — exp_adapter_fingerprint_uniqueness

## Verdict: KILL (preempt-structural, F#666-pure standalone)

`F#666-pure-standalone` clause (reviewer.md §5) applies. KC set `{K1943: collision-rate > 0 at N=1000, K1944: hash-latency > 5 ms}` is engineering-primitive-only on a `depends_on=[]` experiment. No paired target-metric KC exists; no parent target to inherit. Per F#666 + guardrail 1007, KILL is impermissible on a proxy-only KC, and SUPPORTED requires a target-metric pair which is absent — so the verdict is structurally preempt, not proxy-FAIL.

The 4-cell K1943×K1944 truth-table contains zero behaviorally-anchored cells: tautology (any ≥128-bit commodity hash trivially passes both — SHA-256 birthday bound at N=1000 ≈ 4.3×10⁻⁷²; BLAKE3 ~1 GB/s on Apple Silicon trivially clears 5 ms for 0.5 MB LoRA artifacts) / engineering defect / implementation defect / degenerate. F#3 (LoRA structural orthogonality cos≈0.0002) eliminates collision-by-similarity as a realistic failure mode; F#6 (behaviorally-anchored hash-routing 5.3% displacement at N=20) is the proper contrast for any v2.

This is the **~30th F#666-pure standalone preempt-KILL** in the drain window and the **1st hash-primitive-correctness sub-form** within the infrastructure-benchmark super-family (NEW; distinct from F#714 wall-clock-latency, F#715/F#754 cache-staleness, F#753 routing-latency, F#739 realtime-streaming-latency, F#758 MEMENTO-inline-latency).

## Adversarial checklist

- (a)–(d) Consistency: results.json verdict=KILLED, all_pass=false, is_smoke=false, PAPER.md verdict line matches, DB status=killed — consistent ✓
- (e) MATH.md untracked (no git history) → no post-claim KC mutation; KCs match DB ✓
- (f) Tautology sniff: KCs themselves are tautological — this IS the preempt rationale, correctly scoped ✓
- (g) K-IDs in artifacts match DB ✓
- (h)–(l) No composition / LORA_SCALE / routing / shutil.copy / hardcoded pass=True (no code path) ✓
- (m) Target model (Gemma 4 E4B per F#627) noted as "not loaded" — consistent ✓
- (m2) Skill invocation: MATH.md §0 cites `/mlx-dev` + `/fast-mlx` as "noted, not used — no code path" → correct preempt-structural disclosure ✓
- (n)–(s) Eval integrity: N/A (no code, no eval) ✓
- (t) Target-gated kill — explicit carve-out applies per F#666-pure standalone clause: F#666 is the *reason* for the preempt, not a blocker; no KC was measured, verdict is structurally preempt ✓
- (u) No scope-changing fix: graceful-failure stub is the canonical preempt-structural artifact ✓

## Non-blocking notes

- `run_experiment.py` imports `sys` instead of canonical `json` + `pathlib`, and `results.json` is on disk via direct write rather than via `main()`. Stylistic deviation from the canonical preempt-structural template; artifact content is correct (verdict=KILLED, all KCs `result="untested"`, all_pass=false, preempt-reason cited). Non-blocking given the structural verdict is sound and all 4 required artifacts are present with consistent content.
- 4 hygiene defects flagged in scratchpad (serialization undefined, adapter-population undefined, latency budget unanchored, N=1000 vs Pierre's ~10² adapter pool). All are KC-verdict-flipping, but the F#666-pure preempt is sufficient on its own — hygiene-patch route is moot when no target KC exists.

## Routing

- `experiment complete --status killed` already executed by researcher.
- Finding F#760 registered (`experiment finding-list` confirms id #760 [killed]).
- Emitting `review.killed` for analyst hand-off (LEARNINGS.md authorship downstream).

## Assumptions

- F#666 + guardrail 1007 cover engineering-primitive KCs (not only classical proxies like cos-sim/PPL/routing-acc). Rationale is structural — any KC set unidentifiable-as-a-finding regardless of outcome (the 4-cell table contains no behaviorally-anchored cell) is forbidden-solo on a standalone experiment. Prior art F#714/F#715/F#739/F#753/F#758 confirms infrastructure-benchmark KCs covered. This experiment is the 6th sub-form within the super-family.
- Pierre's fingerprint use is versioning / dedup / cache-keys per pre-reg notes; if the intended use were instead "adversarial-robustness-of-adapter-identity-under-active-attack," the unblock condition would differ (separate experiment).
