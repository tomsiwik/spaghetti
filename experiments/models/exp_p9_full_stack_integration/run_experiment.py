"""
P9.G0: Full Stack Integration — Current System Capability Assessment

Kill criteria:
  K1387: Full stack (routed math adapter) outperforms base Gemma 4 by >= 15pp on GSM8K
  K1388: Oracle-routed system outperforms base on mixed-domain MMLU-Pro by >= 10pp
  K1389: Adapter footprint reported (EXPECTED FAIL: 25MB >> 5MB target, documents the gap)

Revised scope (from original CMoE+TT-LoRA+DES design → killed dependencies):
  - Use existing knowledge adapters (q_proj, r=6, 5MB each)
  - Simulate oracle routing (select correct domain adapter)
  - Test adapter composition (math + medical, α=0.5)
  - Establish current system baseline before P11 improvements

See MATH.md for formal theorems and predictions.
"""

import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REGISTRY_PATH = REPO_ROOT / "data" / "adapters" / "registry.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = "--smoke" in sys.argv or __import__("os").environ.get("SMOKE_TEST", "0") == "1"
GSM8K_N = 5 if IS_SMOKE else 50
MMLU_N_PER_CAT = 2 if IS_SMOKE else 20
COMPOSITION_N = 3 if IS_SMOKE else 30

ADAPTER_MATH = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/math"
ADAPTER_CODE = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/code"
ADAPTER_MEDICAL = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical"
ADAPTER_LEGAL = REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/legal"
ADAPTER_FINANCE = REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/finance"

MMLU_PATH = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"

# Domain → adapter path + MMLU-Pro category mapping
# MMLU-Pro categories: math, biology, chemistry, physics, engineering,
#                      health, law, business, economics, psychology,
#                      other, computer science, history, philosophy
DOMAIN_MAP = {
    "math":    {"adapter": ADAPTER_MATH,    "mmlu_cats": ["math", "engineering"]},
    "health":  {"adapter": ADAPTER_MEDICAL, "mmlu_cats": ["health"]},
    "law":     {"adapter": ADAPTER_LEGAL,   "mmlu_cats": ["law"]},
    "finance": {"adapter": ADAPTER_FINANCE, "mmlu_cats": ["economics", "business"]},
}


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def strip_thinking(response):
    """Remove Gemma 4 thinking tokens. Uses p10/W4A16-validated regex."""
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_mcq_answer(response):
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    upper = answer_text.upper()
    if re.match(r'^[A-J]$', upper.strip()):
        return upper.strip(), thinking_chars
    for pattern in [r'answer[:\s]+([A-J])\b', r'\b([A-J])\b(?:\s*$|\s*[.)])',
                    r'(?:^|\s)([A-J])(?:\s*$|\s*[.)])']:
        m = re.search(pattern, upper, re.MULTILINE)
        if m:
            return m.group(1), thinking_chars
    m = re.search(r'\b([A-J])\b', upper)
    if m:
        return m.group(1), thinking_chars
    return None, thinking_chars


def eval_gsm8k(model, tokenizer, n=None, seed=SEED) -> dict:
    """Evaluate on GSM8K (generation task). Returns accuracy dict."""
    from mlx_lm import generate

    n = n or GSM8K_N
    gsm_path = EXPERIMENT_DIR / "data" / "gsm8k_test.jsonl"
    gsm_path.parent.mkdir(parents=True, exist_ok=True)

    if not gsm_path.exists():
        log("  Fetching GSM8K test data via datasets library...")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        with open(gsm_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"  Saved {len(rows)} GSM8K examples")

    gsm_data = []
    with open(gsm_path) as f:
        for line in f:
            gsm_data.append(json.loads(line.strip()))

    rng = np.random.RandomState(seed)
    n = min(n, len(gsm_data))
    sample = [gsm_data[i] for i in rng.choice(len(gsm_data), n, replace=False)]

    def get_gt(answer_str):
        m = re.search(r'####\s*([\d,.-]+)', answer_str)
        return m.group(1).replace(",", "").strip() if m else answer_str.strip()

    def extract_number(text):
        cleaned, _ = strip_thinking(text)
        m = re.search(r'####\s*([\d,.-]+)', cleaned)
        if m:
            return m.group(1).replace(",", "").strip()
        nums = re.findall(r'-?[\d,]+\.?\d*', cleaned)
        return nums[-1].replace(",", "").strip() if nums else None

    correct = 0
    for item in sample:
        gt = get_gt(item["answer"])
        messages = [{"role": "user",
                     "content": f"Solve this math problem step by step.\n\n{item['question']}\n\nAnswer:"}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
            predicted = extract_number(response)
            if predicted and gt and predicted == gt:
                correct += 1
            del response
            mx.eval()
        except Exception as e:
            log(f"    GSM8K error: {e}")

    acc = correct / n * 100 if n > 0 else 0
    log(f"  GSM8K: {acc:.1f}% ({correct}/{n})")
    return {"accuracy": round(acc, 1), "correct": correct, "total": n}


def eval_mmlu_pro_subset(model, tokenizer, categories: list, n_per_cat: int = None,
                          seed=SEED) -> dict:
    """Evaluate MMLU-Pro on specific categories, no thinking."""
    from mlx_lm import generate

    n_per_cat = n_per_cat or MMLU_N_PER_CAT
    if not MMLU_PATH.exists():
        log(f"  WARNING: MMLU-Pro data not found at {MMLU_PATH}")
        return {"accuracy": None, "error": "MMLU data missing"}

    df = pd.read_parquet(MMLU_PATH)
    OPTION_LETTERS = "ABCDEFGHIJ"
    rng = np.random.RandomState(seed)

    correct_total = 0
    total = 0
    per_cat = {}

    for cat in categories:
        cat_df = df[df["category"] == cat]
        if len(cat_df) == 0:
            log(f"  WARNING: category '{cat}' not found in MMLU-Pro")
            continue
        n_sample = min(n_per_cat, len(cat_df))
        idxs = rng.choice(len(cat_df), n_sample, replace=False)
        sample = cat_df.iloc[idxs]

        cat_correct = 0
        for _, row in sample.iterrows():
            options = row.get("options", [])
            n_opts = len(options)
            option_text = "\n".join(
                f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
            correct_letter = OPTION_LETTERS[int(row["answer_index"])]
            user_content = (
                f"Answer the following multiple choice question. "
                f"Select the single best answer ({OPTION_LETTERS[0]}-{OPTION_LETTERS[n_opts-1]}).\n\n"
                f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
            )
            messages = [{"role": "user", "content": user_content}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                response = generate(model, tokenizer, prompt=prompt, max_tokens=128)
                predicted, _ = parse_mcq_answer(response)
                if predicted == correct_letter:
                    cat_correct += 1
                del response
                mx.eval()
            except Exception as e:
                log(f"    MMLU error: {e}")

        correct_total += cat_correct
        total += n_sample
        cat_acc = cat_correct / n_sample * 100 if n_sample > 0 else 0
        per_cat[cat] = {"correct": cat_correct, "total": n_sample, "acc": round(cat_acc, 1)}
        log(f"  [{cat}] {cat_acc:.0f}% ({cat_correct}/{n_sample})")

    acc = correct_total / total * 100 if total > 0 else 0
    log(f"  Subtotal: {acc:.1f}% ({correct_total}/{total})")
    return {"accuracy": round(acc, 1), "correct": correct_total, "total": total,
            "per_category": per_cat}


def compose_adapters(path1: Path, path2: Path, alpha: float = 0.5) -> Path:
    """Compose two LoRA adapters by averaging their weights.

    Returns path to a temp dir with composed adapters.safetensors + adapter_config.json.
    Caller must clean up the temp dir.

    Formula: composed_delta(x) = alpha * (B1 @ A1) x + (1-alpha) * (B2 @ A2) x
    Implemented by averaging lora_a and lora_b matrices separately.
    Note: (alpha*B1 + (1-alpha)*B2)(alpha*A1 + (1-alpha)*A2) ≠ alpha*B1@A1 + (1-alpha)*B2@A2
    We store composed_lora_b = alpha*B1 + (1-alpha)*B2, composed_lora_a = A1 (dominant adapter).
    For equal-weight blending in expectation, this is the parameter-space average.
    """
    weights1 = dict(mx.load(str(path1 / "adapters.safetensors")).items())
    weights2 = dict(mx.load(str(path2 / "adapters.safetensors")).items())
    mx.eval()

    all_keys = set(list(weights1.keys()) + list(weights2.keys()))
    composed = {}
    for key in all_keys:
        if key in weights1 and key in weights2:
            composed[key] = alpha * weights1[key] + (1 - alpha) * weights2[key]
        elif key in weights1:
            composed[key] = alpha * weights1[key]
        else:
            composed[key] = (1 - alpha) * weights2[key]

    mx.eval()

    tmp_dir = Path(f"/tmp/composed_adapter_{int(time.time())}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tmp_dir / "adapters.safetensors"), composed)
    shutil.copy(path1 / "adapter_config.json", tmp_dir / "adapter_config.json")

    del weights1, weights2, composed
    gc.collect()
    mx.clear_cache()
    log(f"  Composed adapter saved to {tmp_dir}")
    return tmp_dir


# ─────────────────────────────────────────────────────────────────
# Phase 1: GSM8K — base vs math adapter  (K1387)
# ─────────────────────────────────────────────────────────────────

def phase_gsm8k():
    """Base model vs math adapter on GSM8K. Verifies K1387."""
    from mlx_lm import load

    log("\n" + "=" * 60)
    log("Phase 1: GSM8K — base vs math adapter (K1387)")
    log("=" * 60)

    results = {}

    # 1a: Base model
    log("\n[1a] Base model GSM8K")
    model, tokenizer = load(MODEL_ID)
    log_memory("base-loaded")
    results["base"] = eval_gsm8k(model, tokenizer)
    cleanup(model, tokenizer)

    # 1b: Math adapter
    log("\n[1b] Math adapter GSM8K")
    if not ADAPTER_MATH.exists():
        log(f"  ERROR: math adapter not found at {ADAPTER_MATH}")
        results["math_adapter"] = {"accuracy": None, "error": "adapter_not_found"}
    else:
        model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER_MATH))
        log_memory("math-adapter-loaded")
        results["math_adapter"] = eval_gsm8k(model, tokenizer)
        cleanup(model, tokenizer)

    # K1387 verdict
    base_acc = results["base"].get("accuracy")
    math_acc = results["math_adapter"].get("accuracy")
    if base_acc is not None and math_acc is not None:
        delta = math_acc - base_acc
        k1387_pass = delta >= 15.0
        log(f"\n  K1387: math={math_acc:.1f}% base={base_acc:.1f}% delta={delta:+.1f}pp "
            f"(need +15pp) → {'PASS' if k1387_pass else 'FAIL'}")
        results["k1387"] = {"delta_pp": round(delta, 1), "pass": k1387_pass}

    return results


# ─────────────────────────────────────────────────────────────────
# Phase 2: Oracle routing on MMLU-Pro domains  (K1388)
# ─────────────────────────────────────────────────────────────────

def phase_oracle_routing():
    """Oracle routing: select correct domain adapter per domain group.

    Tests: does routing to the correct domain adapter improve over base?
    K1388: oracle-routed system outperforms base by >= 10pp on mixed MMLU-Pro.
    """
    from mlx_lm import load

    log("\n" + "=" * 60)
    log("Phase 2: Oracle routing on MMLU-Pro (K1388)")
    log("=" * 60)

    if not MMLU_PATH.exists():
        log(f"  ERROR: MMLU data not found at {MMLU_PATH}")
        return {"error": "mmlu_data_missing"}

    results = {}

    # 2a: Base model on all domains
    log("\n[2a] Base model on all domain categories")
    model, tokenizer = load(MODEL_ID)
    log_memory("base-loaded")
    base_results = {}
    for domain, info in DOMAIN_MAP.items():
        cats = info["mmlu_cats"]
        log(f"  Domain: {domain} (categories: {cats})")
        r = eval_mmlu_pro_subset(model, tokenizer, cats)
        base_results[domain] = r
    cleanup(model, tokenizer)

    base_total_correct = sum(r.get("correct", 0) for r in base_results.values())
    base_total_q = sum(r.get("total", 0) for r in base_results.values())
    base_mixed_acc = base_total_correct / base_total_q * 100 if base_total_q > 0 else 0
    log(f"\n  Base model mixed total: {base_mixed_acc:.1f}% ({base_total_correct}/{base_total_q})")
    results["base"] = {"per_domain": base_results, "mixed_acc": round(base_mixed_acc, 1),
                       "total_correct": base_total_correct, "total_q": base_total_q}

    # 2b: Oracle routing — each domain uses its own adapter
    log("\n[2b] Oracle routing (correct adapter per domain)")
    oracle_domain_results = {}
    for domain, info in DOMAIN_MAP.items():
        adapter_path = info["adapter"]
        cats = info["mmlu_cats"]
        log(f"\n  Domain: {domain}, adapter: {adapter_path.name}")
        if not adapter_path.exists():
            log(f"  ERROR: adapter not found at {adapter_path}")
            oracle_domain_results[domain] = {"accuracy": None, "error": "adapter_not_found"}
            continue
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
        log_memory(f"{domain}-adapter-loaded")
        r = eval_mmlu_pro_subset(model, tokenizer, cats)
        oracle_domain_results[domain] = r
        cleanup(model, tokenizer)

    oracle_total_correct = sum(r.get("correct", 0) for r in oracle_domain_results.values())
    oracle_total_q = sum(r.get("total", 0) for r in oracle_domain_results.values())
    oracle_mixed_acc = oracle_total_correct / oracle_total_q * 100 if oracle_total_q > 0 else 0
    log(f"\n  Oracle routing mixed total: {oracle_mixed_acc:.1f}% ({oracle_total_correct}/{oracle_total_q})")
    results["oracle"] = {"per_domain": oracle_domain_results, "mixed_acc": round(oracle_mixed_acc, 1),
                         "total_correct": oracle_total_correct, "total_q": oracle_total_q}

    # K1388 verdict
    delta = oracle_mixed_acc - base_mixed_acc
    k1388_pass = delta >= 10.0
    log(f"\n  K1388: oracle={oracle_mixed_acc:.1f}% base={base_mixed_acc:.1f}% "
        f"delta={delta:+.1f}pp (need +10pp) → {'PASS' if k1388_pass else 'FAIL'}")
    results["k1388"] = {"oracle_acc": round(oracle_mixed_acc, 1),
                         "base_acc": round(base_mixed_acc, 1),
                         "delta_pp": round(delta, 1), "pass": k1388_pass}

    return results


# ─────────────────────────────────────────────────────────────────
# Phase 3: Adapter composition (math + medical, α=0.5)  (Theorem 2)
# ─────────────────────────────────────────────────────────────────

def phase_composition():
    """Test α=0.5 composition of math + medical adapters.

    Measures: does composition degrade vs single adapter?
    Tests on both math (GSM8K) and medical (MMLU health) to check cross-domain
    interference.
    """
    from mlx_lm import load

    log("\n" + "=" * 60)
    log("Phase 3: Adapter composition test (α=0.5 math+medical)")
    log("=" * 60)

    if not ADAPTER_MATH.exists() or not ADAPTER_MEDICAL.exists():
        log("  ERROR: Math or medical adapter missing, skipping composition test")
        return {"error": "adapter_missing"}

    results = {}
    composed_dir = None

    try:
        # Create composed adapter
        log("\n[3a] Creating composed adapter (α=0.5 math + 0.5 medical)")
        composed_dir = compose_adapters(ADAPTER_MATH, ADAPTER_MEDICAL, alpha=0.5)

        # 3b: Test all three variants on math (GSM8K) + medical (MMLU health)
        n_gsm = COMPOSITION_N
        n_mmlu = max(1, COMPOSITION_N // 2)

        for variant_name, adapter_path in [
            ("math_only", ADAPTER_MATH),
            ("medical_only", ADAPTER_MEDICAL),
            ("composed", composed_dir),
        ]:
            log(f"\n[3b-{variant_name}] Loading adapter: {adapter_path}")
            model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
            log_memory(f"{variant_name}-loaded")

            # Math eval: GSM8K
            log(f"  Math eval (GSM8K n={n_gsm})")
            math_r = eval_gsm8k(model, tokenizer, n=n_gsm, seed=SEED + 100)

            # Medical eval: MMLU health
            log(f"  Medical eval (MMLU health n={n_mmlu}/cat)")
            med_r = eval_mmlu_pro_subset(model, tokenizer, ["health"], n_per_cat=n_mmlu,
                                          seed=SEED + 200)

            results[variant_name] = {"gsm8k": math_r, "mmlu_health": med_r}
            cleanup(model, tokenizer)

        # Compute interference metrics
        if "math_only" in results and "composed" in results:
            math_solo = results["math_only"]["gsm8k"].get("accuracy", 0) or 0
            math_comp = results["composed"]["gsm8k"].get("accuracy", 0) or 0
            med_solo = results["medical_only"]["mmlu_health"].get("accuracy", 0) or 0
            med_comp = results["composed"]["mmlu_health"].get("accuracy", 0) or 0

            results["composition_interference"] = {
                "math_degradation_pp": round(math_comp - math_solo, 1),
                "medical_degradation_pp": round(med_comp - med_solo, 1),
                "theorem2_prediction": "degradation within 5-15pp at α=0.5",
            }
            log(f"\n  Composition interference:")
            log(f"    Math: {math_solo:.1f}% → {math_comp:.1f}% (Δ={math_comp-math_solo:+.1f}pp)")
            log(f"    Medical: {med_solo:.1f}% → {med_comp:.1f}% (Δ={med_comp-med_solo:+.1f}pp)")

    finally:
        if composed_dir and composed_dir.exists():
            shutil.rmtree(composed_dir, ignore_errors=True)
            log(f"  Cleaned up composed adapter dir")

    return results


# ─────────────────────────────────────────────────────────────────
# Phase 4: Footprint audit  (K1389)
# ─────────────────────────────────────────────────────────────────

def phase_footprint():
    """Measure total adapter footprint. K1389 EXPECTED FAIL (25MB >> 5MB target)."""
    log("\n" + "=" * 60)
    log("Phase 4: Adapter footprint audit (K1389 — expected FAIL)")
    log("=" * 60)

    adapters = {
        "math": ADAPTER_MATH,
        "code": ADAPTER_CODE,
        "medical": ADAPTER_MEDICAL,
        "legal": ADAPTER_LEGAL,
        "finance": ADAPTER_FINANCE,
    }

    total_bytes = 0
    per_adapter = {}
    for name, path in adapters.items():
        if not path.exists():
            log(f"  {name}: NOT FOUND")
            per_adapter[name] = {"size_mb": None, "error": "not_found"}
            continue
        # Sum all .safetensors files
        size = sum(f.stat().st_size for f in path.rglob("*.safetensors"))
        size_mb = size / 1024**2
        total_bytes += size
        per_adapter[name] = {"size_mb": round(size_mb, 2)}
        log(f"  {name}: {size_mb:.2f} MB")

    total_mb = total_bytes / 1024**2
    k1389_pass = total_mb < 5.0
    log(f"\n  Total: {total_mb:.2f} MB  (target: < 5 MB)")
    log(f"  K1389: {'PASS' if k1389_pass else 'FAIL (expected — TT-LoRA compression not achieved)'}")

    return {
        "per_adapter_mb": per_adapter,
        "total_mb": round(total_mb, 2),
        "target_mb": 5.0,
        "k1389_pass": k1389_pass,
        "note": (
            "EXPECTED FAIL: current adapters are full-LoRA (5MB each). "
            "TT-LoRA (180KB) was killed due to MCQ failure (exp_p9_ttlora_moe_router). "
            "Next path: larger-rank FFN+attention adapters trained on classification."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("P9.G0: Full Stack Integration — Current System Capability")
    log(f"Model: {MODEL_ID}")
    log(f"Smoke mode: {IS_SMOKE}")
    log("=" * 60)

    t_start = time.time()
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": IS_SMOKE}

    # Phase 1: GSM8K baseline + math adapter (K1387)
    results["phase1_gsm8k"] = phase_gsm8k()

    # Phase 2: Oracle routing on MMLU-Pro (K1388)
    results["phase2_routing"] = phase_oracle_routing()

    # Phase 3: Composition test (Theorem 2 verification)
    results["phase3_composition"] = phase_composition()

    # Phase 4: Footprint audit (K1389)
    results["phase4_footprint"] = phase_footprint()

    # Summary
    elapsed = time.time() - t_start
    results["elapsed_s"] = round(elapsed, 1)
    log(f"\n{'='*60}")
    log(f"Total elapsed: {elapsed/60:.1f} min")

    # Kill criteria summary
    k1387 = results["phase1_gsm8k"].get("k1387", {})
    k1388 = results["phase2_routing"].get("k1388", {})
    k1389 = results["phase4_footprint"]
    log("\nKill criteria summary:")
    log(f"  K1387 (math >= base + 15pp): {k1387.get('pass', 'N/A')} "
        f"(Δ={k1387.get('delta_pp', 'N/A')}pp)")
    log(f"  K1388 (routed >= base + 10pp): {k1388.get('pass', 'N/A')} "
        f"(Δ={k1388.get('delta_pp', 'N/A')}pp)")
    log(f"  K1389 (footprint < 5MB): {k1389.get('k1389_pass', 'N/A')} "
        f"(actual: {k1389.get('total_mb', 'N/A')} MB)")

    results["kill_summary"] = {
        "K1387": k1387,
        "K1388": k1388,
        "K1389": {"pass": k1389.get("k1389_pass"), "total_mb": k1389.get("total_mb")},
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
