# MATH.md — exp_prod_adapter_registry_host

## Preemptive-Kill Via 5-Theorem Stack

**Verdict: KILLED_PREEMPTIVE** (T1∧T2∧T3∧T5 each alone block; T4
reinforces). 32nd cohort preempt; 2nd occurrence of ap-017 axis
**(s) hardware-topology-unavailable** (generalized from CUDA hardware
in loader_portability to public-network infrastructure here).

## Target claim (per DB)

A public adapter registry resolves `pierre://math/v1` to a signed
`.pierre` artifact URL in < 200 ms p99 (K1659), sustains > 99 %
uptime over a synthetic 24 h, 1000-req/min load test (K1660), and
rejects push of unsigned or invalidly-signed adapters (K1661).

## Parent/source finding

`exp_prod_adapter_signing` (SUPPORTED, K1639/K1640/K1641 pass,
verification overhead 0.39 ms). The parent's PAPER and MATH freeze
the scope explicitly:

- Platform: local Apple M5 Pro, MLX 0.31.1 + cryptography 46.0.7
- Boundary: **load-time** integrity check at the `mx.load` API
  (`save_signed`, `verify_file`, `load_verified`)
- Observable: per-file byte-level tamper rejection on a local
  filesystem; no network transport; no producer/push path exercised

Depends-on chain: registry_host → signing (SUPPORTED) →
format_spec_v1 (SUPPORTED). Both ancestors are local-Apple,
load-time, pull-only.

## 5-Theorem Stack

### T1 — Artifact shortfall (hardware-topology-unavailable, reuse)

Required artefacts for the target claim:

- Registry HTTP(S) server implementation for `.pierre` artefacts
  *(✗ absent — repo grep for `pierre://` returns 0 files; no server
  binary on disk; no `aiohttp` / `fastapi` / `axum` adapter-registry
  service anywhere in the tree)*
- DNS / URL-scheme resolver for `pierre://` *(✗ absent — `pierre`
  is not a registered URI scheme in RFC 7595 registries; no local
  scheme handler in the repo)*
- Public-facing host + TLS cert + routable FQDN *(✗ absent — local
  Apple M5 Pro is not a public endpoint; no control of DNS for any
  `pierre` namespace)*
- 24-h synthetic load generator at 1000 req/min *(✗ absent — no
  `locust` / `wrk` / `k6` harness for `.pierre` resolution on disk)*
- Signing pipeline output *(✓ SUPPORTED ancestor produces signed
  files locally; but end-to-end producer→registry→consumer path is
  not wired)*

shortfall = 4 missing artifact categories out of 5 required. The
single available input (signed `.pierre` bytes) is not sufficient
for an *availability-over-network* claim.

### T2 — Resource budget (time ceiling crosses micro cap)

Claim requires **24 h** of continuous synthetic load. Micro ceiling
per PLAN.md is 120 min; macro single-run budgets are also below
24 h of sustained real-traffic simulation. Arithmetic:

- 24 h = 1440 min >> 120 min ceiling  ⇒  12× over
- 1000 req/min × 1440 min = 1.44 × 10⁶ requests
- Each request touches: DNS + TLS + HTTP(S) + signed-blob fetch —
  structurally requires a live network stack not present locally

Budget is simultaneously a *time* ceiling and a *physical-topology*
ceiling (no public network); either alone disqualifies.

### T3 — DB schema completeness

Literal DB annotation at claim time:

```
success_criteria: [] # MISSING
⚠ INCOMPLETE: missing success_criteria
```

F#502/F#646 schema-completeness-vs-instance-fix axis. This is the
4th occurrence of the same schema-incomplete preempt after
tfidf_routing_no_alias, flywheel_real_users, and loader_portability.
Patching success-criteria on this row alone does not address the
cohort-wide pattern.

### T4 — KC pin count (reinforcing, not sole blocker)

Three KC:

| KC    | Text                                                     | Pins present |
|-------|----------------------------------------------------------|--------------|
| 1659  | `pierre://math/v1` → signed URL in < 200 ms p99           | ε (200 ms) + latency-aggregation (p99) = 2 |
| 1660  | > 99 % uptime over 24 h, 1000 req/min                     | ε (99 %) + duration (24 h) + rate (1000/min) = 3 |
| 1661  | Rejects push of unsigned / invalidly-signed adapter       | ε absent (no rate threshold) + enumeration of "invalidly-signed" absent = 0 |

Full-pin template {baseline, pool, enum, rescale, ε}. Total = 5/15 =
0.333 ratio. K1661 lacks an enumeration of attack vectors (byte flip
in sig, forged-key, stripped-manifest, wrong-signer, etc.) and is
therefore under-specified; it is not strictly non-falsifiable but it
is under-pinned (similar to F#645 N-scale subcategory-aggregation).

T4 reinforces T3 (both schema-axis) but by itself is below the ≤ 0.20
auto-block threshold used in loader_portability.

### T5 — Source-finding LITERAL breaches

Five independent scope/semantic gaps between source
(`exp_prod_adapter_signing`, SUPPORTED) and target (registry_host):

**(A) Transport-scope breach: load-time → serve-time.** Source
verifies integrity at the `mx.load` / `verify_file` boundary on the
*consumer* after the bytes are already on the local filesystem.
Target claims resolution and serving over a network-transport
boundary with a distinct failure surface (DNS spoofing, TLS MITM,
CDN cache poisoning, HTTP 5xx error budget) that the source never
exercised.

**(B) Throughput-scope breach: 0.39 ms local → < 200 ms p99 network.**
K1641 measured 0.39 ms verification overhead on a local file.
K1659 requires < 200 ms *p99* end-to-end including DNS + TLS + HTTP
round-trip + blob fetch. The three orders-of-magnitude delta is
dominated by components the source does not touch.

**(C) Uptime-scope breach: undeclared in source.** Source has zero
uptime / availability criterion (it is a pure-function verifier).
Target K1660 demands > 99 %/24 h — an SRE-level operational
guarantee. Source ratification says nothing about host reliability,
redundancy, or rollover; it cannot transport.

**(D) Push-path breach: pull-only → producer API.** Source tests the
*consumer-load* path (K1639: tamper rejection at `verify_file`).
Target K1661 tests the *producer-push* path (server refuses to
ingest unsigned adapters). These are different API surfaces with
different attack vectors (PUT/POST authentication, upload quota,
registry-write ACL) that the source never defined or exercised.

**(E) Hardware/infra-topology breach (ap-017 s, reuse).** The claim
requires a routable public endpoint (`pierre://math/v1`). The local
Apple M5 Pro has no public DNS; no `pierre` URI scheme is registered
at IANA; no TLD resolver for `pierre` exists. This is the same
physical-absence pattern as loader_portability (CUDA hardware
absent) generalized to external **network-infrastructure** absence.
Second instance of ap-017 (s); axis is reusable across all
external-infra PROD experiments.

## Defense-in-depth

T1 alone blocks (4/5 artefacts missing). T2 alone blocks (24 h >>
120 min ceiling AND no public network). T3 alone blocks (DB literal
incomplete). T5 alone blocks (5 independent source-scope breaches;
any one breaks transport). T4 is reinforcing. `all_block = True
(4/5 block; T4 below auto-block threshold but registered)`;
`defense_in_depth = True (≥ 3 block independently)`.

## Reuse of ap-017 axis (s) hardware-topology-unavailable

First registered in exp_prod_adapter_loader_portability (CUDA/CPU
hardware absent). Generalizes here to **external network / DNS /
hosting infrastructure absent**. Both instances share the same
structural signature: target claim demands observations on
infrastructure physically absent from the local platform; local
grep confirms zero on-disk artefacts for the absent resource; the
parent SUPPORTED finding is scoped to the locally-reachable subset
and cannot transport to the missing half of the topology.

Remaining P≤2 candidates for (s) reuse after this iter:
`exp_model_peer_comparison_mistral_nemo` (external CUDA model),
`exp_prod_openai_api_compat` (server infra / network),
`exp_model_quantization_composition_stability` (W4A16 toolchain
typically CUDA-only).

## Kill criteria (pre-registered for this drain)

- **K1659** result = **fail** (no server, no `pierre://` resolver,
  no public host; T1 ∧ T5(A,B,E))
- **K1660** result = **fail** (no infra, 24 h > 120 min ceiling;
  T2 ∧ T5(C))
- **K1661** result = **fail** (no producer-push path on disk;
  T1 ∧ T5(D))

## Success criterion for this drain

This drain SUCCEEDS iff ≥ 3 of 5 theorems block independently, each
with evidence in `results.json`. Exits as KILLED_PREEMPTIVE, never
silently upgraded to SUPPORTED (guardrail 1009).

## Assumptions (per guardrail 1007)

1. Local platform = Apple Silicon only; PLAN.md Part 2 names no
   remote-hosting runner; no public DNS control.
2. Ancestor experiments `exp_prod_adapter_signing` (SUPPORTED) and
   `exp_prod_adapter_format_spec_v1` (SUPPORTED) are authoritative
   for source scope (local, load-time, pull-only).
3. Repo grep for `pierre://` URI scheme returns 0 structural
   matches (confirmed at claim time).
4. `pierre` is not a registered URI scheme at IANA (no public
   resolver exists for third parties).
