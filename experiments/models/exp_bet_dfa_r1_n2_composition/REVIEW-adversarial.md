# REVIEW-adversarial — exp_bet_dfa_r1_n2_composition

**Reviewer route: KILL confirmed (K2314).** Real run, pre-registered threshold crossed decisively.

## Mock / reality checks — pass
- `verify-experiment.sh` exit 0; `is_smoke:false`; real gemma-4-e4b-it-4bit (code = MATH.md = PAPER).
- Adapters are distinct real files (different SHA1s), not copies; 42 layers wrapped, count asserted.
- Wall clock 7328 s consistent with 800 GSM8K + 200 HumanEval greedy generations; per-item details
  present, HumanEval scored by real subprocess test execution.
- Independently recomputed all six accuracies from `details`: every one matches the claimed numbers
  exactly (A=0.140 B=0.705 C=0.335 D=0.400; E=0.380 F=0.330).

## Consistency / integrity — pass
- results.json verdict `killed`, `all_pass:false`, PAPER verdict line KILLED — all agree.
- Kill thresholds in code (5pp / 7pp / 50% / gap≥0.10) match MATH.md exactly. Dir is untracked so
  no git history to prove no post-hoc edit, but the data overshoots K2314 by >4×, so no plausible
  threshold placement rescues it.
- Not tautological: behavioral GSM8K EM + HumanEval pass@1, paired items, greedy, fixed seed.
  Theorem 1 verified at runtime (max|Q_py^T Q_math| = 1.95e-16).

## Adversarial probes (did the kill survive?)
1. **Answer-parsing artifact:** `gsm8k_pred` regex keeps trailing dots, so e.g. pred "560." vs gt
   "560" scores wrong. Re-scored all conditions with numeric normalization: A 0.25, B 0.725,
   C 0.365, D 0.435 → gap 0.36, recovered 0.07 (19.4%), residual 0.29. **Kill robust** — still far
   below the 50% recovery gate and 4× over the 7pp residual cap.
2. **Asymmetric QR ordering:** python-first frames mean math's own delta is deflated in D (its
   overlap with python's 6-dim B-space is removed). This makes D somewhat handicapped by design —
   but the construction was pre-registered exactly this way, and even the generous reading leaves
   recovery at ~18-19%. Caveat noted, not a blocker; a math-first-frame variant is a follow-up,
   not a rescue.
3. **K2313 at-threshold:** solo drop is exactly 0.050 vs strict `> 0.05` — passes by one HumanEval
   item. Immaterial: K2314 alone kills, and PAPER discloses this honestly.
4. **Gap precondition:** gap 0.37 ≥ 0.10, so the kill route (not provisional) is the correct branch.

## Verdict
KILLED stands: exact per-layer output-orthogonality (2e-16) recovered only 6.5pp (17.6%) of a 37pp
behavioral interference gap; pre-registered K2314 required ≥18.5pp and residual ≤7pp.
