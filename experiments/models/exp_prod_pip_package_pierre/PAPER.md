# PAPER.md: exp_prod_pip_package_pierre

## Type: Frontier Extension (engineering)
**Status:** KILLED — infrastructure-blocked (not falsification of Theorems 1–3)
**Date:** 2026-04-18

---

## Verdict

This attempt is **KILLED**. No K1648/K1649/K1650 measurement was produced
because five independent packaging preconditions fail before any
`pip install` can be timed. Per PLAN.md §1, when KCs cannot be evaluated
they are recorded as FAIL — not "inconclusive", not "reformulated", not
silently upgraded. The same discipline used in
`exp_bench_aime_2026` and `exp_bench_livecodebench_v6` (also 2026-04-18).

Theorems 1–3 in `MATH.md` are **not** falsified. They remain pre-registered
predictions for a future re-attempt once the remediation steps below land.

---

## Prediction vs Measurement

| # | Prediction | Expected | Actual | Pass? |
|---|---|---|---|---|
| Theorem 1 (1) | `project.name == "pierre"` | required | `lora-compose` | FAIL precond |
| Theorem 1 (2) | `pierre` in wheel target packages | required | `["composer","micro","macro"]` | FAIL precond |
| Theorem 1 (4) | published as `pierre` on an index | required | not published | FAIL precond |
| Theorem 2 | platform markers on backends | required | `mlx-lm>=0.31.2` unmarked | FAIL precond |
| K1648 | macOS fresh VM install < 2 min | EXPECTED FAIL | not measured | FAIL (unmeasured) |
| K1649 | Linux fresh VM graceful degradation | EXPECTED FAIL | not measured | FAIL (unmeasured) |
| K1650 | version-pin resolves reproducibly | EXPECTED FAIL | not measured | FAIL (unmeasured) |

`run_experiment.py` ran the preflight checks (no model load, no network)
and recorded every clause's pass state in `results.json["preflight"]`.

---

## Blockers (authoritative; mirrored in `results.json["preflight"]`)

### B1. `pyproject.toml` package name is `lora-compose`, not `pierre` (P0)

```toml
[project]
name = "lora-compose"
version = "0.3.0"
```

Falsifies Theorem 1 clause (1). `pip install pierre` resolves nothing.

*Remediation:* verify `pierre` is available on PyPI (https://pypi.org/project/pierre/
should return 404 or be claimable). If squatted, fall back to
`pierre-runtime`. Update `pyproject.toml`. Coordinate with any consumer
of the existing `lora-compose` name.

### B2. `pierre/` source package excluded from wheel (P0)

```toml
[tool.hatch.build.targets.wheel]
packages = ["composer", "micro", "macro"]
```

Falsifies Theorem 1 clause (2). Even if B1 is fixed, the produced wheel
ships `composer`/`micro`/`macro` modules — `import pierre` raises
`ModuleNotFoundError` post-install.

*Remediation:* add `"pierre"` to the packages list. Confirm with
`uv build && unzip -l dist/*.whl | grep pierre/__init__.py`.

### B3. No published artifact under name `pierre` (P0)

`results.json["preflight"]["T1_c4_published_or_local_path"]` shows no
git tag for v0.3.0 either. Falsifies Theorem 1 clause (4) and Theorem 3
clause (a).

*Remediation:* once B1/B2 land, publish to TestPyPI first
(`uv build && twine upload --repository testpypi dist/*`). Verify
`pip install --index-url https://test.pypi.org/simple/ pierre==0.3.1`
resolves before promoting to real PyPI.

### B4. Platform-specific deps lack PEP 508 markers (P1, blocks K1649)

`mlx-lm>=0.31.2` sits in unconditional `dependencies` without a
`; sys_platform == "darwin"` marker. Falsifies Theorem 2.

*Remediation:* either move backend deps into
`[project.optional-dependencies]` (`pip install pierre[mlx]`) — the
current `optional-dependencies` block already has `micro`, `train`,
`serve`, `eval` groups, so the pattern is established — or add explicit
markers:

```toml
dependencies = [
  "numpy",
  'mlx-lm>=0.31; platform_machine == "arm64" and sys_platform == "darwin"',
]
```

### B5. No Linux host on this machine (P1, blocks K1649)

Preflight reports `current_platform=darwin`, `docker_available=True`,
`lima_available=False`. K1649 cannot be measured here even after B1–B4
are resolved.

*Remediation:* run K1649 in CI (GitHub Actions `runs-on:
ubuntu-latest`) once the published wheel exists. A local Docker run
(`docker run --rm -v $PWD:/p python:3.12 bash -c 'pip install pierre &&
python -c "import pierre"'`) would also produce evidence and can be
added to `run_experiment.py` as a follow-up — but only after B1–B3
unblock the install itself.

---

## Assumptions logged (researcher hat autonomy rule)

1. **No rename committed in this hat activation.** Renaming a published
   package name has cross-cutting effects (existing imports, downstream
   consumers, possible squatter conflict). The honest researcher action
   is to document the necessary change and let a follow-up planning
   step approve the rename. Per PLAN.md §1: anti-stuck rule, defer
   non-trivial fixes to dedicated tasks.
2. **No Docker-based Linux test attempted.** Adding a Docker codepath
   while the underlying package can't even be built as `pierre` would
   measure nothing useful and would dirty the experiment dir with
   unproductive scaffolding.
3. **Pre-registered KCs are not relaxed.** A "local fresh-venv install
   of a wheel renamed in-place" was considered as a proxy for K1648 but
   rejected: the KC text is "Fresh macOS VM: pip install pierre", not
   "fresh venv on existing Mac of an unpublished wheel". Proxying would
   silently change the KC's meaning (mem-antipattern: KC-swap-after-failure).

---

## Implications for downstream experiments

- `exp_prod_mlxlm_integration` and `exp_prod_onboarding_first_run` both
  list this experiment as a `depends_on`. They cannot proceed until
  B1–B3 resolve. Flag in their next iteration.
- `exp_prod_adapter_format_spec_v1` (status=supported) is unaffected —
  the format spec stands on its own. The packaging gap is purely
  about runtime distribution.

---

## Honest next steps for a future runner

1. Open a planning task to negotiate the `lora-compose -> pierre`
   rename (or alternative naming).
2. After rename, write `exp_prod_pip_package_pierre_v2` with the same
   K1648/K1649/K1650 thresholds and a `run_experiment.py` that:
   - calls `uv build` and asserts wheel contents
   - creates a fresh venv via `python -m venv`
   - times `pip install dist/*.whl` (proxy K1648 — flag as proxy)
   - in a Docker container, repeats install on `python:3.12-slim`
     (proxy K1649)
   - tags v0.3.1 to TestPyPI and times `pip install
     --index-url=test pierre==0.3.1` (K1650)

This v2 design is *not* run here because B1 (rename) is a prerequisite
of every step.
