# MATH.md — exp_sigreg_threshold_sweep

## Status: PREEMPT-STRUCTURAL-KILL

This experiment is preempt-killed before any code is run because three independent
structural guardrails each block it on their own. Below we prove each independently:
any one of them is sufficient to justify the KILL; all three hold concurrently.

## §1.1 Theorem 1 — F#666-pure (both KCs are proxy; no target-gated pairing)

**Statement.** Let $\mathcal{K} = \{K_{1890}, K_{1891}\}$ be the pre-registered kill
criteria. $K_{1890}$: FPR at threshold $\tau{=}0.05$ is $\geq 30\%$ on synthetic
non-collapse events. $K_{1891}$: FNR at threshold $\tau{=}0.20$ misses actual
collapse events in a ground-truth labelled set. Both are classification-accuracy
proxies per guardrail 1007 (canonical "routing match rate / canary drift / FNR /
classification accuracy as proxy" family). The target-level question — does threshold
$\tau^{*}$ yield better *downstream-task accuracy* or prevent *behavioural collapse*
in a real composed run? — is absent from $\mathcal{K}$.

**Proof.** Per F#666 and guardrail 1007 (KILL requires BOTH proxy and target to fail;
SUPPORTED requires BOTH to pass), a KC set with $|\mathcal{K}|{=}2$ where both are
proxy admits neither verdict. No target KC exists to pair with. KC augmentation is
required before any run produces verdict-eligible data. Per
`mem-antipattern-f666-pure-standalone-preempt-kill`, standalone proxy-only KC sets
are preempt-killed structurally. **QED.**

**F#666-pure census:** this instance is the **18th standalone F#666-pure** drain-window
preempt-KILL.

## §1.2 Theorem 2 — §5 tautological-inter-variant-delta (intra-detector-threshold-delta)

**Statement.** The sweep $\tau \in \{0.05, 0.10, 0.15, 0.20\}$ on a single fixed
detector produces a monotonic operating-point curve: FPR is monotonically
non-increasing in $\tau$ and FNR is monotonically non-decreasing in $\tau$. Any
inter-$\tau$ rank-ordering by FPR or FNR is therefore trivial (determined by detector
geometry, not by any signal about what $\tau$ matters for the downstream goal).

**Proof.** For any monotone score function $s(\cdot)$ and decision rule
$\mathbb{1}[s(x) \geq \tau]$, raising $\tau$ weakly reduces the positive set.
Hence raising $\tau$ weakly reduces FPR and weakly raises FNR. The rank order
$\tau_1 < \tau_2 < \tau_3 < \tau_4 \Rightarrow \text{FPR}(\tau_1) \geq \text{FPR}(\tau_4) \wedge \text{FNR}(\tau_1) \leq \text{FNR}(\tau_4)$
is a tautology of the detector — no external target is anchoring the comparison.
Per `mem-antipattern-tautological-inter-variant-delta` (§5 of reviewer.md) this
is a preempt-KILL. **QED.**

**§5 census:** this is §5's **12th** drain-window instance and the **2nd instance
of the intra-instantiation sub-variant** (first was F#712's
intra-adapter-rank-delta; this is intra-detector-threshold-delta).

## §1.3 Theorem 3 — F#669 parent-target-unverified (parent F#713 PROVISIONAL)

**Statement.** The "default" threshold $\tau{=}0.10$ which this sweep implicitly
anchors against is supplied by parent mechanism F#713
(`exp_sigreg_composition_monitor`, PROVISIONAL design-lock, empirical deferred).
No empirical verification of the SIGReg Epps-Pulley statistic on real N-composition
collapse events has been filed; only a design-lock on three pre-registered surfaces.
Hence any statement "threshold X is better/worse than default 0.10" references an
untested RHS baseline.

**Proof.** F#669 promotion-gate states: *if a child experiment's RHS baseline is a
PROVISIONAL parent's untested output, the child is preempt-KILLed because (a) a
difference against an unverified baseline is information-free and (b) the child
cannot produce a stronger claim than its parent's status permits*. Parent F#713
is PROVISIONAL at time of claim. **QED.**

**F#669 reuse census:** this is the **8th F#669-family reuse** in the drain window
(F#687, F#698, F#699, F#727, F#728, F#729 + 2 pre-drain-window).

**Cross-parent observation:** F#728/F#729 exhibited the triple-fire composition
(F#666-pure + §5 + F#669) with parent F#682 (JEPA). This instance exhibits the
same triple-fire composition but with parent F#713 (SIGReg). **1st cross-parent
instance** of the structural/parent-dependent triple-fire sub-composition.

## §2 KC set as written

- K1890 — proxy (FPR on detector; canonical guardrail 1007)
- K1891 — proxy (FNR on detector; canonical guardrail 1007)

No target-gated KC. |Proxy|=2, |Target|=0.

## §3 What a non-preempt version would need

1. Convert K1890/K1891 to be *paired* with a target metric. Example target:
   "At the threshold chosen, downstream N>3 composition run retains task accuracy
   $\geq$ baseline $-\epsilon$". FPR/FNR alone is insufficient.
2. Wait for parent F#713 `_impl` to provide an empirical ground-truth event set
   before sweeping thresholds against that set.
3. Pick two candidate thresholds from a Neyman-Pearson / ROC argument external to
   the sweep itself (i.e. do not let the sweep define its own optimum without
   external anchoring).

## §4 Actions taken

- Artifacts written: MATH.md (this file), run_experiment.py (no-op scaffold),
  PAPER.md, REVIEW-adversarial.md (blank per preempt convention), LEARNINGS.md.
- DB updated: status=killed, K1890 and K1891 marked inconclusive,
  success_criteria populated (F#702 hygiene-patch applied).
- No `_impl` follow-up filed — per the F#687/F#698/F#699/F#727/F#728/F#729 precedent,
  preempt-structural KILLs do not spawn `_impl` children.

## §5 Antipattern audit

- F#666-pure: **FIRES** (both proxy, no target)
- §5 tautological-inter-variant-delta: **FIRES** (intra-detector-threshold-delta,
  2nd intra-instantiation sub-variant)
- F#669 parent-target-unverified: **FIRES** (parent F#713 PROVISIONAL)
- Composition math / LORA_SCALE / shutil.copy / hardcoded pass / eval truncation /
  proxy-model / eval-template — N/A (no run)
- F#702 hygiene-patch applied (platform, dir, success_criteria populated)

## §6 References

- F#666 (proxy/target pairing guardrail) — `mem-antipattern-f666-pure-standalone-preempt-kill`
- F#703 (analyst escalation: explicit F#666-pure clause in reviewer.md §5)
- F#669 (parent-target-unverified) — `mem-promotion-same-parent-repeat-blocker` (F#682 analog)
- F#712 (§5 intra-instantiation sub-variant precedent — intra-adapter-rank-delta)
- F#713 (parent SIGReg PROVISIONAL design-lock)
- F#728/F#729 (structural/parent-dependent triple-fire sub-composition; both F#682-parent)
- Guardrail 1007 (target-gated KILL; classification accuracy / FPR / FNR as proxy)

## §7 Triple-fire ledger

Triple-fire sub-composition (F#666-pure + §5 + F#669):
- F#728 (parent F#682): 1st
- F#729 (parent F#682): 2nd
- **F#730 (parent F#713): 3rd — 1st cross-parent instance**
