# MATH.md: pip install pierre — packaging soundness

## Type: Frontier Extension (engineering, not research)

## Context

This experiment is product/distribution work, not a model-research claim.
The "math" is software-engineering reasoning over PEP-517 builds, PEP-508
dependency resolution, and PEP-600 (manylinux) wheel compatibility. We
state the soundness conditions explicitly so the kill criteria are
falsifiable, not aspirational.

`exp_prod_adapter_format_spec_v1` (status=supported) defines `.pierre`
adapter file format v1. This experiment asks the prerequisite
distribution question: **can a user install the runtime that consumes
those files via `pip install pierre`?**

Current repo state (verified 2026-04-18):
- `pyproject.toml` declares package `name = "lora-compose"` v0.3.0.
- `tool.hatch.build.targets.wheel = ["composer", "micro", "macro"]` —
  the `pierre/` directory is NOT included in the wheel.
- No PyPI release of either `pierre` or `lora-compose` exists at
  install-test time.
- Repo has only macOS host (M5 Pro 48GB); no Linux VM available.

These are not assumptions for the proof — they are the input constraints
that determine which kill criteria are even testable.

---

## Theorem 1 (packaging soundness)

**Claim.** A pip-installable Python package named `pierre` exists and is
importable inside a fresh virtualenv on a supported platform iff:
1. **Build:** `pyproject.toml` declares `name = "pierre"` and a
   `[build-system]` block whose backend can produce a PEP-517 wheel from
   the source tree.
2. **Inclusion:** the `pierre/` source package (containing
   `__init__.py`) is listed in the wheel build targets.
3. **Resolvable deps:** every entry in `[project.dependencies]` resolves
   under PEP-508 on the target platform — including any backend
   (`mlx-lm` on Apple Silicon, `torch` on CUDA/CPU).
4. **Distribution:** the wheel is reachable to `pip` (PyPI, a private
   index, or a local path).

If any of (1)–(4) fails, `pip install pierre` errors out before the
import smoke test begins and K1648/K1649/K1650 are unmeasured FAILs.

**Proof sketch.** Each clause is an independent necessary condition:
- (1) `pip install` requires a metadata source; without `name = pierre`,
  `pip install pierre` resolves nothing. The build backend (`hatchling`,
  `setuptools`, etc.) is the only mechanism that converts the source
  tree into installable artifacts (PEP 517 §3).
- (2) An empty wheel installs but `import pierre` raises
  `ModuleNotFoundError`. Wheel-target inclusion is the only mechanism
  that places `pierre/__init__.py` under `site-packages/` (PEP 427 §1).
- (3) A failing transitive dep aborts install (PEP 517 §6, PEP 600 for
  manylinux tag matching). On Linux, `mlx-lm` has no manylinux wheel —
  the resolver must select an alternative backend or reject the install.
- (4) Without an index entry or path, `pip install pierre` returns
  `Could not find a version that satisfies the requirement pierre`. ∎

## Theorem 2 (cross-platform backend autodetect)

**Claim.** A single source distribution can install on macOS-arm64 and
Linux-x86_64 iff platform-specific backends are declared as **optional
extras** (PEP 508 markers), not unconditional `dependencies`.

Reasoning: `mlx`/`mlx-lm` are macOS-arm64 only; PyPI marks them with
`Requires-Python` and platform tags such that pip on Linux refuses to
install. If they sit in `dependencies` (unconditional), Linux install
fails at resolution. The fix is environment markers, e.g.

```toml
dependencies = [
  "numpy",
  'mlx-lm>=0.31; platform_machine == "arm64" and sys_platform == "darwin"',
  'torch>=2.4;  platform_machine == "x86_64"',
]
```

or moving backends to `[project.optional-dependencies]` selected by an
extras spec (`pip install pierre[mlx]` / `pip install pierre[cuda]`).

**Prediction.** Without explicit markers, K1649 (Linux graceful
degradation) FAILs at the resolver step — never reaching the
import-time backend autodetect that the experiment proposes.

## Theorem 3 (reproducible version pinning)

**Claim.** `pip install pierre==X.Y.Z` resolves identically across
identical Python/platform pairs iff (a) the package is published with
that exact version on the index, and (b) the resolver is given a fully
constrained dependency set (lock file or `==`-pinned transitive deps).

Without (a), the install fails. Without (b), transitive resolution is
non-deterministic across pip versions (different backtracking orders).

---

## Pre-registered Kill Criteria (immutable — do not edit post-run)

| KC | ID | Threshold | Maps to |
|---|---|---|---|
| K1648 | 1648 | Fresh macOS VM: `pip install pierre` → working `import pierre` + smoke inference in < 2 min | Theorem 1 (1)+(2)+(3 mac)+(4) |
| K1649 | 1649 | Fresh Linux VM: `pip install pierre` → graceful CPU/CUDA degradation OR clear error if no backend | Theorem 2 |
| K1650 | 1650 | `pip install pierre==X.Y.Z` resolves reproducibly | Theorem 3 |

KC selection rationale: each Theorem corresponds to one KC, so a
specific clause's failure pinpoints which precondition is missing.

---

## Predictions for THIS attempt (pre-run, given current repo state)

| KC | Expected | Why |
|---|---|---|
| K1648 | FAIL (unmeasured) | Theorem 1 clauses (1), (2), (4) all violated: pyproject name is `lora-compose`; `pierre/` not in wheel targets; no published distribution. |
| K1649 | FAIL (unmeasured) | Theorem 2 path is moot while Theorem 1 fails. Also: no Linux host available to test even after Theorem 1 is satisfied. |
| K1650 | FAIL (unmeasured) | Theorem 3 (a) violated: nothing published. Even local `pip install ./dist/*.whl` cannot exercise version-resolution determinism. |

The honest reading is: **all three KCs FAIL because the experiment cannot
be executed against current repo state**. This is identical to the
exp_bench_aime_2026 / exp_bench_livecodebench_v6 pattern (infrastructure
absent → unmeasured FAIL, status `killed`, not falsification of the
underlying theorem).

---

## Remediation paths (ordered, gating)

To unblock a future re-attempt:

1. **Rename + repackage.** Update `pyproject.toml`:
   - `name = "pierre"` (or `name = "pierre-runtime"` if `pierre` is
     squatted on PyPI — verify before renaming).
   - Add `pierre` to `tool.hatch.build.targets.wheel` packages.
   - Move `mlx-lm`, `torch`, `vllm` etc. into platform-marked extras
     per Theorem 2.
2. **Build the wheel locally.** `uv build`, inspect `dist/*.whl` with
   `unzip -l` to confirm `pierre/__init__.py` is inside.
3. **Local fresh-venv install.** `uv venv /tmp/pierre-test && source
   /tmp/pierre-test/bin/activate && pip install dist/*.whl` — measure
   wall-clock, confirm `python -c "import pierre"` succeeds. This
   satisfies a *local* analogue of K1648 but not the "fresh VM" form.
4. **Linux verification.** Requires a Linux host — out of scope for
   this hat / this machine. Defer to CI (e.g., GitHub Actions
   `runs-on: ubuntu-latest`) once steps 1-3 are clean.
5. **Publish.** TestPyPI first, then PyPI. Only then can K1650 be
   exercised against a real index.

Until step 1 is done, this experiment cannot be re-run with any chance
of measuring its KCs.
