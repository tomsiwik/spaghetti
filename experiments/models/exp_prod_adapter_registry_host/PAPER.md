# PAPER: exp_prod_adapter_registry_host

**Verdict: KILLED_PREEMPTIVE.** 4 / 5 theorems block independently
(sole-blockers: T1, T2, T3, T5); defense-in-depth holds. The claim
is structurally unmeasurable on the local Apple-only platform and is
the **32nd** preemptive-kill in the audit-2026-04-17 drain. **Second
instance of ap-017 preempt axis (s) hardware-topology-unavailable**
— generalized from CUDA hardware (iter 35) to external network /
DNS / hosting infrastructure.

## Hypothesis (from DB / MATH.md)

A public registry resolves `pierre://math/v1` → signed `.pierre`
URL in < 200 ms p99 (K1659), sustains > 99 % uptime over a synthetic
24 h, 1000 req/min load test (K1660), and rejects push of unsigned
or invalidly-signed adapters (K1661).

## Predictions vs. Measurements

| Theorem | Predicted to block? | Measured | Status |
|---------|---------------------|----------|--------|
| T1 artifact-shortfall | yes (≥ 3 missing) | shortfall = 3 / 5: `pierre_uri_scheme_resolver`, `pierre_specific_load_harness`, `public_dns_for_pierre_namespace`; `pierre://` grep = 0 hits; `nvidia_smi` absent; arm64 Darwin | ✅ blocks |
| T2 resource-budget | yes (24 h >> 120 min) | required = 1440 min vs ceiling = 120 min ⇒ 12× over; 1.44 M requests | ✅ blocks |
| T3 schema-completeness | yes (DB literal incomplete) | `success_criteria = []`; DB flag `⚠ INCOMPLETE`; 4th occurrence of F#502/F#646 axis | ✅ blocks |
| T4 kc-pin-count | reinforcing only | pin_ratio = 0.333 (> 0.20 auto-block floor); K1661 non-falsifiable (no rate, no attack-vector enum) | ◦ registered, below auto-block |
| T5 source-literal-breaches | yes (≥ 3 / 5) | 5 / 5 breaches: (A) transport-scope, (B) throughput-scope, (C) uptime-scope, (D) push-path, (E) hardware/infra-topology | ✅ blocks |

`all_block_strict = False` (T4 reinforces but does not auto-block);
`defense_in_depth = True`; sole-blockers = {T1, T2, T3, T5}.

## Kill criteria

| KC    | Result | Reason |
|-------|--------|--------|
| K1659 (`pierre://math/v1` → signed URL < 200 ms p99) | **fail** | no registry server, no scheme resolver, no public host — T1 ∧ T5(A,B,E) |
| K1660 (> 99 % uptime over 24 h, 1000 req/min) | **fail** | no infra + 24 h > 120 min ceiling — T2 ∧ T5(C) |
| K1661 (rejects push of unsigned / invalidly-signed adapter) | **fail** | no producer-push path on disk + KC under-pinned — T1 ∧ T4 ∧ T5(D) |

## Source scope (inherited local-Apple, load-time, pull-only)

- Parent `exp_prod_adapter_signing` (SUPPORTED): local load-time
  integrity verify at `mx.load`; 0.39 ms verify overhead; no network
  transport; no producer API; no uptime criterion.
- Grandparent `exp_prod_adapter_format_spec_v1` (SUPPORTED): bitwise
  round-trip on Apple-Silicon MLX only.

Both ancestors are scoped to a local filesystem + MLX toolchain. The
target claim crosses four distinct scope boundaries at once
(transport, throughput, uptime, push) and a fifth physical
infrastructure boundary (public DNS / TLS / hosting), which is the
reuse of ap-017 axis (s).

## Novel ap-017 axis — second instance

**(s) hardware-topology-unavailable** — iter 35 registered the axis
for missing CUDA hardware; this iter generalizes to missing **public
network / DNS infrastructure**. The structural signature is
identical:

1. target claim demands observations on infrastructure physically
   absent from the local platform;
2. local grep confirms zero on-disk artefacts for the absent
   resource;
3. the parent SUPPORTED finding is scoped to the locally-reachable
   subset and cannot transport to the missing half of the topology.

Candidates for further (s) reuse among the remaining P≤2 experiments:

- `exp_model_peer_comparison_mistral_nemo` (external CUDA model)
- `exp_prod_openai_api_compat` (server infra / network)
- `exp_model_quantization_composition_stability` (W4A16 toolchain
  typically CUDA-only)

## Runner behaviour (for reproducibility)

- Pure stdlib: `subprocess`, `platform`, `shutil`, `pathlib`, `json`.
- No MLX import, no model load, no network I/O.
- Grep-based artefact audit excludes this experiment's own directory
  to avoid self-reference.
- Tight T1 greps (`pierre://`, `pierre_registry`,
  `adapter_registry_server`, `.pierre.*GET`, `locust.*pierre`,
  `k6.*pierre`, `load_test.*adapter`) surface only real-implementation
  tokens; generic framework imports (`from fastapi`) elsewhere in the
  repo are excluded to prevent false-positives.
- Wall time << 1 s.

## Verdict consistency check (guardrail 1009)

- `results.json.verdict = "KILLED"` ✓
- `results.json.status = "killed"` ✓
- `results.json.all_pass = False` ✓
- `results.json.preemptive_kill = True` ✓
- `results.json.defense_in_depth = True` ✓
- PAPER.md verdict line = KILLED_PREEMPTIVE ✓
- `is_smoke = False` ✓
- No silent upgrade to SUPPORTED ✓

## Cohort progress

- 32nd preemptive-kill (audit-2026-04-17 cohort)
- ap-017 axes: composition-bug 23, scale-safety 2, tautological-
  routing 3, projection-scope 2, tautological-duplicate 1,
  **hardware-topology-unavailable 2 (iter 35 + iter 36)**
- 13th SUPPORTED-source preempt (ap-017 (s) reuse). Remaining
  P≤2 open = 12 after this kill.

## Caveats / uncertainties

- T1 grep is tighter than iter 35 but still operates on token
  matches, not AST / API surface. A REVIEW-adversarial sweep could
  tighten further (e.g., look for a `pierre_registry.serve()` call
  in an `if __name__ == "__main__"`).
- T4 pin ratio (0.333) crosses neither block thresholds observed in
  prior preempts; it is reinforcing only. Defense-in-depth does not
  depend on T4.
- The **generalization** of ap-017 (s) from "hardware absent" to
  "public infrastructure absent" is a scope widening. The analyst
  may choose to split into (s-hw) and (s-net) sub-axes when the cap
  is raised.
