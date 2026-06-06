# E4: Activation Arithmetic Composition — Adversarial Review

## Verdict: KILL

**Override**: `is_smoke: true` → KILL. Failure is method-level, not sample-level. Explicit decompose prompting itself degrades accuracy by -15pp (10% vs 25% base). This is the prompt, not the injection — more samples cannot rescue a mechanism where the intervention itself is counterproductive. Same precedent as E1 (#801), E2 (#802), E3 (#806), E6 (#804).

## Adversarial Checklist

| Check | Result | Notes |
|---|---|---|
| (a) verdict consistency | PASS | results.json=KILLED, PAPER.md=PROVISIONAL, override to KILL |
| (b) all_pass vs claim | PASS | all_pass=false, no supported claim |
| (c) PAPER.md verdict | PASS | Says PROVISIONAL with strong kill signal |
| (d) is_smoke | OVERRIDE | is_smoke=true → KILL (method-level failure) |
| (e) KC mutation | PASS | KCs match MATH.md pre-reg |
| (f) tautology | FLAG | K2024 measures "change" not "improvement" — passes vacuously with degradation (-5pp). Tautological-proxy: F#666 |
| (g) K-ID match | PASS | Code measures same quantities as MATH.md |
| (h) composition bugs | N/A | No LoRA weight operations |
| (i) LORA_SCALE | N/A | α=8.0 is ActAdd injection, not LoRA scale |
| (j) routing | N/A | |
| (k) shutil.copy | PASS | Not present |
| (l) hardcoded pass | PASS | Not present |
| (m) model match | PASS | MATH.md=Gemma 4 E4B, code=mlx-community/gemma-4-e4b-it-4bit |
| (m2) skill invocation | PASS | /mlx-dev + /fast-mlx cited; mx.eval, mx.clear_cache used correctly |
| (n) base acc=0% | PASS | base=25% |
| (o) headline n | PASS | N=20 (>15) |
| (p) synthetic padding | PASS | |
| (r) pred-vs-measurement | PASS | Table present in PAPER.md |
| (s) math errors | PASS | |
| (t) target-gated (F#666) | FLAG | K_struct (proxy) FAIL + K2024 (target) technically PASS but degradation. See below. |

## Kill Rationale

### 1. Method-level failure: strategy forcing is counterproductive
Both intervention mechanisms degrade accuracy on Gemma 4 E4B GSM8K:
- **Explicit decompose prompt**: -15pp (10% vs 25% base)
- **ActAdd injection (L=7, α=8.0)**: -5pp (20% vs 25% base)

The prompt-based degradation proves the issue is with strategy forcing itself, not the injection mechanism. The model's default reasoning outperforms any explicit strategy instruction.

### 2. K2024 tautological-proxy (F#666 variant)
K2024 pre-registered as "accuracy **change**" (not improvement), threshold >2pp. Measured |Δ|=5pp → technically PASS. But the change is degradation (-5pp). A KC that passes by making things worse is tautological — it provides no evidence for the hypothesis that ActAdd can beneficially steer reasoning.

### 3. Convergence of evidence across E1/E4/E6
Three independent approaches to strategy forcing all fail on reasoning:
- E1 (#801): Mean-diff extraction captures format, not strategy
- E6 (#804): Hedgehog distillation antagonistic for reasoning
- E4: Contrastive ActAdd injection degrades accuracy

Each uses a different mechanism (mean-diff, attention-matching, contrastive activation injection) and all converge on the same conclusion: strategy forcing is counterproductive for reasoning on Gemma 4 E4B.

## Assumptions
- K2024 PASS with degradation is treated as a kill signal despite technically meeting the pre-registered threshold. Rationale: the hypothesis was that steering would improve reasoning, not merely change it; a KC that passes by degradation provides no support.

## Positive findings (non-blocking)
- Contrastive extraction partially fixes E1: cos 0.76 vs 0.99 (early layers L3-9 discriminate)
- Layer-dependent strategy encoding: early layers carry strategy-specific info, late layers converge
- Both findings are valuable for E11 contrastive extraction work
