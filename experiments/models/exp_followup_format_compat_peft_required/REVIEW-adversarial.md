# REVIEW-adversarial.md — exp_followup_format_compat_peft_required

## Verdict: **PROCEED**

One-line reason: K1576 (all 3 gates) structurally sound, pre-registered before run, SUPPORTED verdict consistent across results.json / PAPER.md / DB.

## Adversarial checklist

| # | Check | Result |
|---|---|---|
| (a) | `results.json["verdict"]=SUPPORTED` vs DB `supported` | ✓ consistent |
| (b) | `all_pass=true` vs `supported` | ✓ consistent |
| (c) | PAPER.md verdict line has no PROVISIONAL/PARTIAL/INCONCLUSIVE/DEGENERATE | ✓ clean |
| (d) | `is_smoke=false` for full-run claim | ✓ |
| (e) | MATH.md / run_experiment.py git history — KC edits post-run? | ✓ no edits after run (both pre-registered in 6518f02 / c952fad) |
| (f) | Tautology sniff on KCs | ✓ none: K1576.a is a hard-require assert (would crash the module if peft absent — sensitive); K1576.b cross-checks wrap count vs `N_LAYERS*3=12` (would catch silently-skipped targets); K1576.c verifies distinct A matrices via `max|Aq-Ak|>1e-3` threshold (0.589/0.640 measured — not identity) |
| (g) | K1576 in code measures what MATH.md says | ✓ gates map 1:1 to Theorem 1 / Theorem 2 |
| (h) | No `sum(lora_A)`, `add_weighted_adapter(combination_type="linear")`, or independent key-sum buggy composition | ✓ no composition performed; row-stack is a *structural probe*, not a composition claim |
| (i) | Unsafe LORA_SCALE≥12 hardcoded | ✓ `ALPHA=6.0`, r=6, effective scale=1 |
| (j) | Routing on single sample applied to all | N/A (no routing) |
| (k) | `shutil.copy` of sibling adapter as new domain | ✓ `shutil.rmtree` only (clears target dir before rewrite) — adapter is built in-place via QR, not copied |
| (l) | Hardcoded `{"pass": True}` in gate dict | ✓ each gate's `pass` is a computed boolean (shape_ok ∧ wrap_count_eq ∧ distinct_A …) |
| (m) | Target model in MATH.md ≠ model loaded | ✓ both are tiny `LlamaConfig` constructed from scratch; scope explicitly declared in MATH.md §5 |
| (m2) | Skill invocation (MLX skills from PLAN.md Part 2) | N/A — this is a *format interop* test run on CPU via PyTorch/peft; MATH.md §5 explicitly declares MLX-target constraints do not apply |
| (n) | Base accuracy=0% with zero thinking chars | N/A (no eval) |
| (o) | Headline n<15 | N/A (structural KC, not a distribution claim) |
| (p) | Synthetic padding (B=0 / Gaussian adapters counted as real) | ✓ synthetic adapter is the *subject* of the format test, not a population unit |
| (q) | Cited baseline drift | N/A |
| (r) | PAPER.md has prediction-vs-measurement table | ✓ P1–P6 table present |
| (s) | Math errors / unsupported claims | ✓ Theorem 1 matches peft 0.18.1 loader semantics; Theorem 2 block-diagonal fusion is standard LA; proofs sound |

All (a)–(s) pass. No blocking issues.

## Strengths worth preserving

1. **Structural immunity to silent-bypass antipattern** — top-level `import peft` ensures the KC cannot pass without peft present. This pattern should be the template for all future "runtime-X-compat" claims in the repo.
2. **Wrap-count check inside K1576.b** — catches the specific failure mode where `target_modules` resolves to an empty set and load "succeeds" with 0 LoRA layers. A parent-style schema check would not have caught this.
3. **Distinct-A probe in K1576.c** — correctly refuses the subset-fallacy shortcut (naive row-stack fusion) by measuring `max|Aq-Ak|` and enforcing rank-expansion logic.

## Non-blocking observations

- The `LlamaForCausalLM` proxy for Gemma/Qwen is fine because Theorem 1 is architecture-agnostic for any separate-QKV HF model, and MATH.md §5 scopes the claim correctly. A future `exp_g4_*` live-load test would upgrade coverage without contradicting this result.
- Runtime is 0.065 s — well under the <10 s target. Reproducibility is cheap.

## Assumptions logged

- Accepted PAPER.md §6 assumption that `base_model_name_or_path=None` + in-memory model object is a standard PEFT pattern (verified against peft 0.18.1 `PeftModel.from_pretrained` signature — correct).
- Accepted that K1576.a passing "because the script ran" is sensitive (not tautological) because the top-level import is hard-required and would crash module load if peft were absent.

## Routing

PROCEED → `review.proceed` → Analyst writes LEARNINGS.md.
