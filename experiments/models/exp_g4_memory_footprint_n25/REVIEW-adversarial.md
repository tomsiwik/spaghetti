# REVIEW-adversarial.md — exp_g4_memory_footprint_n25

**Verdict: KILL (already killed in DB; confirm, do not re-complete).**

## Summary

14th consecutive `audit-2026-04-17` cohort precondition-probe KILL. MATH.md
pre-registered K1596 with a 3-precondition tripwire (P1 base model, P2 N=25
v_proj+o_proj safetensors, P3 N>1 multi-adapter loader). Probe ran in 6.3 ms,
P1 passed, P2 and P3 failed → K1596 UNMEASURABLE → status=killed. No MLX model
load was invoked. Same upstream blocker as Findings #605/#606/#608/#610/#611/
#612/#613/#615/#616/#617/#618/#619/#620.

## Adversarial checklist (17 items)

**Consistency:**
- (a) results.json verdict=`killed` ↔ DB status=`killed` — **PASS**
- (b) all_pass=`false` ↔ status=`killed` — **PASS**
- (c) PAPER.md verdict = "KILLED (K1596 UNMEASURABLE)" — **PASS**
- (d) is_smoke=`true` explicitly matches probe-only mode; claim is killed not supported — **PASS**

**KC integrity:**
- (e) MATH.md K1596 unchanged since pre-registration (first/only run) — **PASS**
- (f) No tautology — probe tests structural file existence + upstream status, not an algebraic identity — **PASS**
- (g) K1596 in code ↔ MATH.md both describe "peak RSS ≤ 5 GB with base + N=25 adapters attached; UNMEASURABLE on precondition fail" — **PASS**

**Code ↔ math:**
- (h) No `sum(lora_A)` / `add_weighted_adapter(linear)` / independent A+B summation — **N/A** (pure file probe)
- (i) No `LORA_SCALE=20` or any hardcoded scale ≥ 12 — **N/A** (no LoRA load)
- (j) No single-sample routing-for-all — **N/A**
- (k) No `shutil.copy` of sibling adapter — **N/A**
- (l) No hardcoded `{"pass": True}` — results computed from probe outcomes — **PASS**
- (m) Target model Gemma 4 E4B 4-bit in MATH.md; probe searches exactly `gemma*4bit` in HF cache — no proxy substitution — **PASS**
- (m2) No MLX code executed; skill invocation evidence not required for a pure-Python file-existence probe — **N/A**

**Eval integrity:**
- (n) No eval run, no base accuracy number, no avg_thinking_chars — **N/A**
- (o) No headline n — claim is UNMEASURABLE — **N/A**
- (p) No synthetic padding; probe does not inflate N by counting stubs (finds 0 safetensors in stub dirs) — **PASS**
- (q) No cited-baseline drift — **N/A**

**Deliverables:**
- (r) PAPER.md has prediction-vs-measurement table (P1/P2/P3 rows) — **PASS**
- (s) No unsupported math claims; the RSS-is-UNMEASURABLE-without-attachment argument is sound (lazy MLX allocation justifies needing a real forward pass) — **PASS**

**Total: 17/17 PASS or N/A.**

## Independent verification

- P1: `ls ~/.cache/huggingface/hub | grep -i gemma | grep -i 4bit` → 3 dirs
  (gemma-2-2b-it-4bit, gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit). **Confirmed.**
- P2: 4 canonical dirs enumerated. Three exist with 0 safetensors; one missing
  (`exp_g4_5domain_real_hf`). Total = 0/25. **Confirmed.**
- P3: `exp_p1_t2_single_domain_training/results.json` shows
  `all_pass=False verdict=KILLED lora_scale=None`. **Confirmed.**

## Assumptions

- Since DB already has `status=killed`, the reviewer does NOT call
  `experiment complete` a second time — that would create a duplicate evidence
  record. Finding registration proceeds.
- Cohort-count audit (14 consecutive KILLs) is logged in PAPER.md/results.json;
  orchestrator-level claim-queue filter on `tag=audit-2026-04-17` remains the
  structural fix (six analyst escalations on record). Reviewer hat does not
  own that change; flagged for analyst next.

## Route

`review.killed` → analyst (writes LEARNINGS.md).
