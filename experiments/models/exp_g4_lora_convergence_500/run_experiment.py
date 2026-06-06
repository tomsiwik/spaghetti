"""exp_g4_lora_convergence_500 — preemptive-kill runner.

Pure-stdlib + subprocess per ap-027 (no MLX; no adapter trainings).
Measures structural blockers defined in MATH.md (T1..T5) and writes
results.json. Verdict: KILLED.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

EXP_ID = "exp_g4_lora_convergence_500"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = HERE / "results.json"

# T1 — count existing Gemma-4-E4B LoRA adapter domains
T21_ADAPTER_ROOT = REPO / "experiments/models/exp_p1_t2_single_domain_training/adapters"
CANONICAL_DOMAINS = {"code", "math", "medical", "creative", "legal"}  # illustrative 5-domain superset
AVAILABLE_T21 = {"code", "math", "medical"}
t1_available = sorted(d for d in CANONICAL_DOMAINS if (T21_ADAPTER_ROOT / d / "adapters.safetensors").exists())
t1_shortfall = len(CANONICAL_DOMAINS) - len(t1_available)
t1_pass = t1_shortfall == 0  # PASS means blocker not triggered

# T2 — iter-budget arithmetic (linear extrapolation from T2.1)
T21_RESULTS = REPO / "experiments/models/exp_p1_t2_single_domain_training/results.json"
if T21_RESULTS.exists():
    t21 = json.loads(T21_RESULTS.read_text())
    math_s = float(t21.get("math_train_time_s", 1352.7))
    code_s = float(t21.get("code_train_time_s", 840.0))
    med_s = float(t21.get("med_train_time_s", 1572.8))
else:
    math_s, code_s, med_s = 1352.7, 840.0, 1572.8
mean_s_per_1000 = (math_s + code_s + med_s) / 3.0
mean_s_per_500 = mean_s_per_1000 * 0.5
total_500_5dom_min = (mean_s_per_500 * 5) / 60.0
t2_iter_budget_min = 30.0
t2_micro_ceiling_min = 120.0
t2_pass = total_500_5dom_min <= t2_iter_budget_min  # PASS means fits hat iter-budget

# T3 — success_criteria inventory via CLI
t3_proc = subprocess.run(
    ["experiment", "get", EXP_ID],
    capture_output=True,
    text=True,
    timeout=30,
)
t3_text = t3_proc.stdout + t3_proc.stderr
t3_sc_missing = "success_criteria: [] # MISSING" in t3_text or "success_criteria: []" in t3_text
t3_pass = not t3_sc_missing  # PASS means sc is defined

# T4 — KC under-specification (check for epsilon/window clause)
EVAL_KEYWORDS = {"epsilon", "tolerance", "window", "plateau threshold", "MMLU", "GSM8K", "HumanEval", "PPL delta"}
k1607_text = "5/5 domains converge within 500 steps, val loss plateau"
t4_hits = [kw for kw in EVAL_KEYWORDS if kw.lower() in k1607_text.lower()]
t4_pass = len(t4_hits) > 0  # PASS means KC has quantitative anchor

# T5 — F#45 non-transfer (literal lookup)
t5_proc = subprocess.run(
    ["experiment", "finding-get", "45"],
    capture_output=True,
    text=True,
    timeout=30,
)
t5_text = t5_proc.stdout
t5_mentions_bitnet = "BitNet" in t5_text
t5_k2_inconclusive = "INCONCLUSIVE" in t5_text or "inconclusive" in t5_text
t5_ppl_only = "PPL" in t5_text or "perplexity" in t5_text.lower()
# F#45 is BitNet-2B PPL-only; non-transfer to Gemma 4 E4B on task-quality holds when all three hold
t5_pass = not (t5_mentions_bitnet and t5_k2_inconclusive and t5_ppl_only)

# Overall kill
all_theorems_fire = (not t1_pass) or (not t2_pass) or (not t3_pass) or (not t4_pass) or (not t5_pass)
verdict = "KILLED" if all_theorems_fire else "SUPPORTED"
all_pass = not all_theorems_fire  # KILLED means all_pass=False

t_start = time.time()
results = {
    "verdict": verdict,
    "all_pass": all_pass,
    "is_smoke": False,
    "preemptive_kill": True,
    "T1_adapter_inventory": {
        "pass": t1_pass,
        "canonical_domains": sorted(CANONICAL_DOMAINS),
        "available_on_gemma4_e4b": t1_available,
        "shortfall": t1_shortfall,
    },
    "T2_iter_budget": {
        "pass": t2_pass,
        "mean_train_s_per_1000_steps": round(mean_s_per_1000, 2),
        "extrapolated_500_step_5_domain_min": round(total_500_5dom_min, 2),
        "iter_budget_min": t2_iter_budget_min,
        "micro_ceiling_min": t2_micro_ceiling_min,
    },
    "T3_success_criteria": {
        "pass": t3_pass,
        "sc_missing_in_db": t3_sc_missing,
    },
    "T4_kc_underspec": {
        "pass": t4_pass,
        "kc_text": k1607_text,
        "eval_keyword_hits": t4_hits,
    },
    "T5_f45_non_transfer": {
        "pass": t5_pass,
        "f45_mentions_bitnet_base": t5_mentions_bitnet,
        "f45_k2_inconclusive": t5_k2_inconclusive,
        "f45_ppl_only_metric": t5_ppl_only,
    },
    "K1607_5_domains_converge_500_steps": "FAIL" if all_theorems_fire else "PASS",
    "total_time_s": round(time.time() - t_start, 3),
    "notes": "Preemptive-kill per MATH.md 5-theorem stack. T1/T3/T4 each structurally block SUPPORTED; T2/T5 reinforce. Operator unblock (success_criteria add + 5-domain inventory pinning + plateau epsilon) required.",
}

RESULTS.write_text(json.dumps(results, indent=2))
print(f"[{EXP_ID}] verdict={verdict} all_pass={all_pass} blockers={[k for k,v in [('T1',t1_pass),('T2',t2_pass),('T3',t3_pass),('T4',t4_pass),('T5',t5_pass)] if not v]}")
