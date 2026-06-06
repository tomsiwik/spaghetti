# PAPER.md: exp_bench_aime_2026

## Type: Guided Exploration
**Status:** KILLED — infrastructure-blocked (not falsification of Theorems 1–3)

---

## Verdict

This attempt is **KILLED**. The AIME 2026 benchmark was not measured. Two independent
infrastructure blockers prevent `run_experiment.py` from producing any evidence; a third,
less severe, code bug would also surface once the first two are fixed. The pre-registered
kill criteria K1417/K1418/K1419 are recorded as FAIL because none can be evaluated — this
is the honest reading per PLAN.md §1 (no silent upgrades, no KC reformulation).

Theorems 1–2 in `MATH.md` (quantization-adjusted base range 34.9–38.3% pass@2; GSM8K→AIME
adapter uplift ≈0pp) are **not** falsified. They remain pre-registered predictions that a
future rerun must verify once the blockers below are cleared.

---

## Prediction vs Measurement Table

| # | Prediction | Value | Actual | Pass? |
|---|---|---|---|---|
| Theorem 1 | Base E4B-4bit pass@2 AIME 2026 | 30–40% | — (not run) | N/A |
| Theorem 2 | Math adapter AIME uplift | < 5pp (≈ 0) | — (not run) | N/A |
| K1417 | Base within 10pp of 42.5% | EXPECTED PASS (~37%) | not measured | FAIL (unmeasured) |
| K1418 | Math adapter ≥ base + 10pp | EXPECTED FAIL | not measured | FAIL (unmeasured) |
| K1419 | Eval completes in < 2h (n=2 seeds) | EXPECTED PASS | — (harness absent) | FAIL |

---

## Blockers (authoritative list — also in `results.json`)

### 1. MathArena harness not installed (P0)

`micro/models/reference_implementations/matharena/` exists but is empty. `run_experiment.py`
calls `subprocess.run([python, MATHARENA_DIR/'scripts'/'run.py', ...])`; that entry point
does not exist on disk. The experiment exits before `mlx_lm.server` is even started.

*Remediation:* `git clone https://github.com/eth-sri/matharena` into the reference dir and
`uv pip install -e` it. Verify `scripts/run.py --help` and the presence of
`configs/competitions/aime/aime_2026.yaml` before retry.

### 2. Math adapter weights absent (P0)

`adapters/registry.json` points at
`micro/models/exp_p1_t2_single_domain_training/adapters/math/`. That directory contains
only `adapter_config.json`; `adapters.safetensors` is missing. `mlx_lm.server
--adapter-path …` would fail at load.

*Side effect:* Finding #421 (math adapter reached 82% GSM8K) depends on weights that are
not currently persisted anywhere under `micro/models/` — any composition/benchmark
experiment that claims to use "the math adapter" inherits this gap.

*Remediation:* rerun `exp_p1_t2_single_domain_training` with weight-persistence verified,
or retract Finding #421's headline number until weights are recovered.

### 3. `find_math_adapter()` iterates the wrong object (low severity, but still breaks)

```python
registry = json.loads(registry_path.read_text())   # dict with keys schema_version, base_model, adapters
for entry in registry:                             # iterates the keys, not the adapters list
    if entry.get("domain") == "math": ...
```

This would return `None` even if weights existed. Fix: `for entry in registry["adapters"]:`.
Documented here so the next runner doesn't also trip on it.

---

## Assumptions (logged per researcher hat autonomy rule)

1. **No network install is attempted within this hat activation.** `uv pip install -e matharena`
   plus the fresh clone, weight retraining, and a 30-problem × 2-seed run together exceed the
   researcher's 30-minute / 40-tool-call envelope. The honest call is KILLED + documented
   blockers, not a partial install that leaves the tree dirtier without producing evidence.
2. **The pre-registered kill criteria are not relaxed or reworded.** If they cannot be
   measured, they are marked FAIL, not "inconclusive". This is the PLAN.md §1
   verdict-consistency discipline applied to the unusual "harness missing" case.
3. **The downstream experiments that cite a "math adapter" (exp_m2p_composition_n5,
   exp_model_peer_comparison_*, etc.) inherit blocker #2** until those weights are restored.
   Flagging this here so the reviewer can propagate the concern via `experiment finding-add`.

---

## Next steps (for the worker that picks this up post-fix)

1. Clone and install MathArena; confirm `scripts/run.py --help` and `configs/competitions/aime/aime_2026.yaml`.
2. Rerun `exp_p1_t2_single_domain_training` with weight persistence (or locate an alternative
   math-adapter checkpoint); update `adapters/registry.json` only after `adapters.safetensors`
   is on disk and loads via `mlx_lm.server --adapter-path`.
3. Fix `find_math_adapter()` to iterate `registry["adapters"]`.
4. Re-claim `exp_bench_aime_2026` as a NEW attempt (not an edit of this killed verdict —
   PLAN.md §1 forbids silent upgrades). MATH.md predictions stay; results.json is rewritten.

---

## Evidence

```json
{
  "ran": false,
  "reason": "infrastructure-blocked",
  "blockers": ["matharena_harness_absent", "math_adapter_weights_absent", "find_math_adapter_iterates_dict_keys"]
}
```
