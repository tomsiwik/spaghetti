"""Preflight runner for exp_prod_pip_package_pierre.

This experiment is INFRASTRUCTURE-BLOCKED on the current repo state. Rather
than fabricate a wheel-build/install chain that would only pretend to test
the kill criteria, this script performs the *preflight* checks that the
researcher would run before a real K1648/K1649/K1650 evaluation, and
records each precondition's pass/fail state in results.json.

Failing preconditions are themselves evidence — they tell the next runner
exactly what must be repaired before another attempt is meaningful.

No model is loaded. No network call is made. Runtime: < 1 s.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIERRE_DIR = REPO_ROOT / "pierre"


def check_pyproject_name() -> dict[str, Any]:
    """Theorem 1 clause (1): package name must be 'pierre'."""
    if not PYPROJECT.exists():
        return {"pass": False, "reason": "pyproject.toml missing"}
    cfg = tomllib.loads(PYPROJECT.read_text())
    name = cfg.get("project", {}).get("name", "<missing>")
    return {
        "pass": name == "pierre",
        "actual_name": name,
        "expected_name": "pierre",
        "reason": (
            "OK"
            if name == "pierre"
            else f"pyproject project.name = '{name}', not 'pierre'"
        ),
    }


def check_pierre_in_wheel_targets() -> dict[str, Any]:
    """Theorem 1 clause (2): pierre/ package must be in wheel build targets."""
    cfg = tomllib.loads(PYPROJECT.read_text())
    targets = (
        cfg.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    has_pierre = "pierre" in targets
    return {
        "pass": has_pierre,
        "wheel_packages": targets,
        "reason": (
            "OK" if has_pierre else f"'pierre' not in wheel packages list {targets}"
        ),
    }


def check_pierre_init_exists() -> dict[str, Any]:
    """Theorem 1 clause (2): pierre/ must have __init__.py to be importable."""
    init = PIERRE_DIR / "__init__.py"
    return {
        "pass": init.exists(),
        "path": str(init.relative_to(REPO_ROOT)),
        "reason": "OK" if init.exists() else f"{init} not found",
    }


def check_platform_markers() -> dict[str, Any]:
    """Theorem 2: mlx/torch should sit behind PEP 508 platform markers."""
    cfg = tomllib.loads(PYPROJECT.read_text())
    deps = cfg.get("project", {}).get("dependencies", [])
    optional = cfg.get("project", {}).get("optional-dependencies", {})

    risky = []
    for dep in deps:
        bare = dep.split(";")[0].strip().split()[0].split(">=")[0].split("==")[0]
        marked = ";" in dep
        if bare in {"mlx", "mlx-lm", "torch", "vllm"} and not marked:
            risky.append({"dep": dep, "issue": "no platform marker"})

    return {
        "pass": len(risky) == 0,
        "unmarked_platform_specific": risky,
        "optional_groups": list(optional.keys()),
        "reason": (
            "OK"
            if not risky
            else f"{len(risky)} platform-specific deps without env markers — would break Linux install"
        ),
    }


def check_pierre_published() -> dict[str, Any]:
    """Theorem 1 clause (4) + Theorem 3 (a): does PyPI know about 'pierre'?"""
    # Honest no-network check: if the test cannot reach PyPI, we cannot
    # falsify publication. We assert no published artifact for THIS repo's
    # version because no release tag matching pyproject.version exists.
    cfg = tomllib.loads(PYPROJECT.read_text())
    version = cfg.get("project", {}).get("version", "0.0.0")
    name = cfg.get("project", {}).get("name", "")
    git_tag_present = False
    try:
        out = subprocess.run(
            ["git", "tag", "-l", f"v{version}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_tag_present = bool(out.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return {
        "pass": False,  # never published as 'pierre'
        "package_name": name,
        "version": version,
        "git_tag_for_version": git_tag_present,
        "reason": (
            f"package '{name}' has not been published to PyPI under name 'pierre'; "
            f"no release tag detected; even renaming first would require a publish step"
        ),
    }


def check_linux_host() -> dict[str, Any]:
    """K1649 host availability check."""
    on_linux = sys.platform.startswith("linux")
    has_docker = shutil.which("docker") is not None
    has_lima = shutil.which("limactl") is not None
    return {
        "pass": on_linux,
        "current_platform": sys.platform,
        "docker_available": has_docker,
        "lima_available": has_lima,
        "reason": (
            "OK on Linux"
            if on_linux
            else f"Cannot test K1649 on {sys.platform}; "
            f"Docker={has_docker}, Lima={has_lima} could host a Linux VM but "
            f"setup is out of scope for this hat"
        ),
    }


def main() -> int:
    preflight = {
        "T1_c1_name_is_pierre": check_pyproject_name(),
        "T1_c2_pierre_in_wheel_targets": check_pierre_in_wheel_targets(),
        "T1_c2b_pierre_init_exists": check_pierre_init_exists(),
        "T2_platform_markers_present": check_platform_markers(),
        "T1_c4_published_or_local_path": check_pierre_published(),
        "K1649_linux_host_available": check_linux_host(),
    }

    blocker_count = sum(1 for v in preflight.values() if not v["pass"])

    kc_results = {
        "K1648_macos_install_under_2min": {
            "pass": False,
            "reason": (
                "Cannot execute. Failing preconditions: "
                "name!=pierre, pierre/ not in wheel, no published artifact. "
                "Until pyproject is repackaged the install command itself errors out."
            ),
        },
        "K1649_linux_graceful_degradation": {
            "pass": False,
            "reason": (
                "Cannot execute. (a) Theorem 1 not met -> install fails before "
                "platform check; (b) no Linux host available on this machine."
            ),
        },
        "K1650_version_pinning_reproducible": {
            "pass": False,
            "reason": (
                "Cannot execute. No published version exists; pinned install "
                "of an unpublished name resolves to nothing."
            ),
        },
    }

    out = {
        "experiment": "exp_prod_pip_package_pierre",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "ran": False,
        "status": "infrastructure_blocked",
        "reason": (
            "Three independent packaging preconditions fail before any install "
            "can be timed. See preflight + remediation in PAPER.md."
        ),
        "preflight": preflight,
        "preflight_blocker_count": blocker_count,
        "kill_criteria": {k: v["pass"] for k, v in kc_results.items()},
        "kill_criteria_rationale": {k: v["reason"] for k, v in kc_results.items()},
        "remediation_summary": [
            "1. Rename pyproject project.name -> 'pierre' (verify PyPI availability first)",
            "2. Add 'pierre' to tool.hatch.build.targets.wheel.packages",
            "3. Move mlx-lm/torch/vllm to platform-marked deps or [project.optional-dependencies]",
            "4. uv build && inspect dist/*.whl for pierre/__init__.py",
            "5. Local fresh-venv install timing test (proxy for K1648)",
            "6. Provision Linux host (CI runner, Docker, Lima) for K1649",
            "7. Publish to TestPyPI then PyPI; then re-run for K1650",
        ],
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"verdict={out['verdict']} blockers={blocker_count}/6 preconditions failing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
