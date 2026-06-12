# REVIEW-adversarial — exp_pgolf_random_frame_compose

**Route: KILL (confirmed).** Real run, both pre-registered kill criteria crossed by wide margins.

## Mock / not-real checks — PASS
- `verify-experiment.sh` exit 0 ("REAL result ok, model-backed").
- No mocks: real MLX 0.31.1 training, real corpora (tinyshakespeare cached on disk at
  `data/tinyshakespeare.txt`, 1.1 MB; python stdlib globbed live). `is_smoke: false`.
- Independent re-execution by reviewer (SMOKE=1, repo venv): pipeline runs end-to-end and
  produces qualitatively consistent numbers (gap +0.21, interference ~0.2 BPB). Smoke path
  correctly emits PROVISIONAL only. Real results.json restored after the check.

## Consistency — PASS
- results.json `verdict: KILLED`, `all_pass: false`; PAPER.md verdict line agrees; numbers in
  PAPER.md tables match results.json to all printed digits.
- Internal cross-check that a fabricator would likely miss: prose solo BPB is bit-identical
  across shared/disjoint arms (2.467735…) — exactly what the code's determinism implies, since
  the prose frame is Q[:, :r] in both arms.

## Integrity — PASS (one note)
- Thresholds in code (0.08 / 0.20 / 0.02) match MATH.md exactly. Directory is untracked in git,
  so threshold history is unverifiable — but moot: gap 0.1757 is 2.2× the kill threshold and
  cut −2.4% is 22 pp below the required +20%; no plausible goalpost placement flips the verdict.
- No tautology: composition is Σᵢ Bᵢ(Aᵢx) with two genuinely trained adapters; solo models
  contain only their own adapter (no gradient leak); the shared-vs-disjoint contrast is a real
  two-arm comparison, and the pre-registered validity gate (I_shared ≥ 0.02) passed at 0.2528.
- Fair K2320 control: dense and random-frame arms share batch seed, steps, LR, schedule.

## Evidence quality
Behavioral (held-out BPB per domain), not a proxy. Interference is large and real (≈0.25 BPB);
exact train-time output-frame orthogonality bought nothing (−2.4%). Single seed is the only
weakness, but margins dwarf plausible seed noise and the result matches the prior at Gemma
scale (R1 17.6% recovery). Kill stands.
