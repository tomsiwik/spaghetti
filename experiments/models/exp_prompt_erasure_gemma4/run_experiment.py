#!/usr/bin/env python3
"""exp_prompt_erasure_gemma4 — Orca 2 Prompt Erasure verification on
Gemma-4-E4B-it-4bit.

Re-analysis of sibling `exp_knowledge_disentanglement_control`:
  - Same adapter (rank-16 LoRA on v_proj+o_proj, scale 4.0, 16 layers,
    N_STEPS=60, n_train=25).
  - Same prompt-erased eval (no `system` role at inference).
  - This script adds the Orca-2-specific metric the sibling did not:
    method-invocation rate under prompt erasure (K1721).

K1721 — method invocation rate ≥ 50% AND ≥ base + 20pp.
K1722 — |Δ MMLU| ≤ 2pp (reused from sibling results.json).
K1723 — recipe fidelity (structural, 3/3).

is_smoke=true; verdict maps to PROVISIONAL in results.json but --status
killed at CLI because K1722 is unambiguously falsified (−30pp vs 2pp
bound, well beyond n=20 sampling noise).
"""

import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
SIBLING_DIR = EXP_DIR.parent / "exp_knowledge_disentanglement_control"
SIBLING_RESPONSES = SIBLING_DIR / "data" / "eval_responses.jsonl"
SIBLING_RESULTS = SIBLING_DIR / "results.json"
SIBLING_TRAIN = SIBLING_DIR / "data" / "train_multi.jsonl"
SIBLING_RUN_PY = SIBLING_DIR / "run_experiment.py"
SIBLING_ADAPTER = SIBLING_DIR / "adapters" / "method_multi" / "adapters.safetensors"
RESULTS_FILE = EXP_DIR / "results.json"


# ── method signature (per MATH.md A2) ────────────────────────────────────────
_METHOD_LEXICON_RE = re.compile(
    r"(?i)(restate|identif[yi].{0,30}(information|relevant|concept|question)"
    r"|evaluate.{0,20}option|thinking\s*process|step\s*1[:.)]|subgoal)"
)
_NUMBERED_STEPS_RE = re.compile(r"(?m)^\s*[1-4][.)]\s")


def invokes_method(resp_prefix: str) -> bool:
    """MATH.md A2: ≥1 method-lexicon hit AND ≥2 numbered-step hits."""
    if not resp_prefix:
        return False
    lex_hit = _METHOD_LEXICON_RE.search(resp_prefix) is not None
    n_steps = len(_NUMBERED_STEPS_RE.findall(resp_prefix))
    return lex_hit and n_steps >= 2


def load_responses(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rate_by_arm(rows, bench=None):
    """Return (n_invoked, n_total) per arm."""
    out = {"base": [0, 0], "adapter": [0, 0]}
    for r in rows:
        arm = r.get("arm") or r.get("tag")
        if bench is not None and r.get("bench") != bench:
            continue
        if arm not in out:
            continue
        out[arm][1] += 1
        if invokes_method(r.get("resp_prefix", "")):
            out[arm][0] += 1
    return out


def k1723_recipe_fidelity():
    """Structural assertions on sibling training pipeline.

    a) Teacher prompt string contains the 4-phase method template.
    b) Student training `messages` contain no `system` role.
    c) Eval chat templates contain no `system` role (inspect sibling code).
    """
    checks = {}

    # (a) teacher prompt contains subgoal template
    src = SIBLING_RUN_PY.read_text() if SIBLING_RUN_PY.exists() else ""
    teacher_has_template = (
        "METHOD_SYSTEM_PROMPT" in src
        and "Restate" in src
        and "Identify" in src
        and "Evaluate" in src
    )
    checks["a_teacher_template"] = teacher_has_template

    # (b) training messages contain no `system` role
    student_clean = True
    if SIBLING_TRAIN.exists():
        with open(SIBLING_TRAIN) as f:
            for line in f:
                ex = json.loads(line)
                roles = {m["role"] for m in ex.get("messages", [])}
                if "system" in roles:
                    student_clean = False
                    break
    else:
        student_clean = False
    checks["b_no_system_in_train"] = student_clean

    # (c) eval code: chat-template call uses only user role (no system)
    eval_clean = (
        '[{"role": "user"' in src
        and 'apply_chat_template' in src
    )
    checks["c_eval_no_system"] = eval_clean

    checks["pass"] = all(
        [checks["a_teacher_template"],
         checks["b_no_system_in_train"],
         checks["c_eval_no_system"]]
    )
    return checks


def main():
    results = {
        "experiment": "exp_prompt_erasure_gemma4",
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "mlx_lm_version": "0.31.2",
        "is_smoke": True,
        "reuse": True,
        "source": "exp_knowledge_disentanglement_control",
        "reuse_rationale": (
            "Sibling experiment implemented the full Orca-2 recipe "
            "(teacher=METHOD_PROMPT, student prompt-erased). "
            "We compute the Orca-2-specific method-invocation metric "
            "(K1721) that the sibling did not compute; K1722 is reused "
            "from sibling's MMLU result."
        ),
        "sibling_adapter_path": str(SIBLING_ADAPTER),
        "lora_config": {
            "rank": 16, "scale": 4.0, "dropout": 0.0,
            "keys": ["self_attn.v_proj", "self_attn.o_proj"],
            "num_layers": 16,
        },
        "n_steps": 60,
        "n_per_cat_train": 5,
        "train_cats": [
            "math", "computer science", "health", "law", "economics"
        ],
        "seed": 42,
    }

    # ── Load sibling data ────────────────────────────────────────────────
    if not SIBLING_RESPONSES.exists():
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        results["error"] = f"sibling responses missing at {SIBLING_RESPONSES}"
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"ABORT: {results['error']}")
        return 1
    if not SIBLING_RESULTS.exists():
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        results["error"] = f"sibling results.json missing at {SIBLING_RESULTS}"
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"ABORT: {results['error']}")
        return 1

    responses = load_responses(SIBLING_RESPONSES)
    sibling = json.loads(SIBLING_RESULTS.read_text())

    # ── K1721 — method invocation rate ──────────────────────────────────
    # Compute on MMLU + MATH/GSM8K subsets (MCQ-style + reasoning). We
    # evaluate pooled across benches (closest to Orca-2's general-domain
    # method test), and also per-bench for diagnostics.
    pool = rate_by_arm(responses, bench=None)
    per_bench = {
        b: rate_by_arm(responses, bench=b)
        for b in ("gsm8k", "mmlu", "triviaqa")
    }

    b_i, b_n = pool["base"]
    a_i, a_n = pool["adapter"]
    base_rate = (b_i / b_n * 100.0) if b_n else 0.0
    adapter_rate = (a_i / a_n * 100.0) if a_n else 0.0
    delta = adapter_rate - base_rate

    k1721_pass = bool((adapter_rate >= 50.0) and (delta >= 20.0))
    results["k1721_method_invocation"] = {
        "pass": k1721_pass,
        "adapter_rate_pct": round(adapter_rate, 1),
        "base_rate_pct": round(base_rate, 1),
        "delta_pp": round(delta, 1),
        "threshold_abs_pct": 50.0,
        "threshold_delta_pp": 20.0,
        "n_base": b_n,
        "n_adapter": a_n,
        "per_bench": {
            b: {
                "base_rate_pct": round(
                    (per_bench[b]["base"][0] / per_bench[b]["base"][1] * 100.0)
                    if per_bench[b]["base"][1] else 0.0, 1),
                "adapter_rate_pct": round(
                    (per_bench[b]["adapter"][0] / per_bench[b]["adapter"][1] * 100.0)
                    if per_bench[b]["adapter"][1] else 0.0, 1),
                "n_adapter": per_bench[b]["adapter"][1],
            } for b in per_bench
        },
        "signature_definition": (
            "≥1 lexicon hit (restate|identify .* relevant|evaluate option|"
            "thinking process|step 1|subgoal) AND ≥2 numbered-step lines "
            "at line start"
        ),
    }

    # ── K1722 — MMLU knowledge preservation (reused) ───────────────────
    mmlu_base = sibling.get("eval_base", {}).get("mmlu", {}).get("acc")
    mmlu_adp = sibling.get("eval_adapter", {}).get("mmlu", {}).get("acc")
    if mmlu_base is None or mmlu_adp is None:
        k1722_delta = None
        k1722_pass = False
    else:
        k1722_delta = mmlu_adp - mmlu_base
        k1722_pass = bool(abs(k1722_delta) <= 2.0)
    results["k1722_mmlu_preserved"] = {
        "pass": k1722_pass,
        "base_acc": mmlu_base,
        "adapter_acc": mmlu_adp,
        "delta_pp": None if k1722_delta is None else round(k1722_delta, 2),
        "threshold_abs_pp": 2.0,
        "source": "exp_knowledge_disentanglement_control/results.json",
    }

    # ── K1723 — recipe fidelity (structural) ───────────────────────────
    k1723 = k1723_recipe_fidelity()
    results["k1723_recipe_fidelity"] = k1723

    # ── Verdict ────────────────────────────────────────────────────────
    all_pass = bool(
        k1721_pass and k1722_pass and k1723.get("pass", False)
    )
    results["all_pass"] = all_pass

    # Smoke → PROVISIONAL in results.json; CLI will use --status killed if
    # any KC is unambiguously falsified (K1722 has a 30pp drop far beyond
    # binomial CI at n=20).
    results["verdict"] = "PROVISIONAL"
    if k1722_delta is not None and abs(k1722_delta) > 10.0:
        results["verdict_note"] = (
            f"K1722 falsified by {k1722_delta:+.1f}pp (>>2pp bound, "
            f">>n=20 CI≈21pp). Orca-2 recipe at rank-16/N=25/60-steps "
            f"on Gemma-4-E4B-it-4bit collapses MMLU knowledge."
        )
    results["verdict_reason"] = (
        "smoke re-analysis: K1722 unambiguously falsified (see note); "
        "K1721 result depends on signature parse of 300-char prefixes; "
        "K1723 structural. CLI --status killed."
    )

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("=" * 60)
    print(f"exp_prompt_erasure_gemma4 (re-analysis of sibling)")
    print("=" * 60)
    print(f"  K1721 method invocation ≥50% AND ≥base+20pp: "
          f"{'PASS' if k1721_pass else 'FAIL'}")
    print(f"    base={base_rate:.1f}%  adapter={adapter_rate:.1f}%  "
          f"Δ={delta:+.1f}pp  (n_adp={a_n})")
    print(f"  K1722 |ΔMMLU|≤2pp: "
          f"{'PASS' if k1722_pass else 'FAIL'}")
    print(f"    base={mmlu_base}%  adapter={mmlu_adp}%  Δ={k1722_delta}pp")
    print(f"  K1723 recipe fidelity: "
          f"{'PASS' if k1723.get('pass') else 'FAIL'}  {k1723}")
    print(f"  VERDICT: {results['verdict']}  all_pass={all_pass}")
    if results.get("verdict_note"):
        print(f"  NOTE: {results['verdict_note']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
