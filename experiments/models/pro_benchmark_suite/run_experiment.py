#!/usr/bin/env python3
"""Pierre Pro: benchmark suite — does scale=5 composition preserve general benchmarks?

Verification experiment. Finding #329-330 proved scale=5 preserves MMLU on BitNet-2B.
Finding #324 killed scale=20 (catastrophic benchmark destruction). This replicates
on Qwen3-4B-4bit.

Configurations:
  1. base — Qwen3-4B-4bit, no adapters
  2. single_math — math adapter at scale=5
  3. composed_n5 — N=5 NRE composition at scale=5
  4. composed_n5_dare — N=5 NRE + DARE p=0.5 at scale=5

Kill criteria:
  K822: Composed N=5 (scale=5) loses to base on ALL 3 benchmarks → killed
Success:
  S81: Composed N=5 within 5pp of base on at least 2/3 benchmarks
"""

import gc
import json
import math
import os
import re
import ast
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_DIR = EXPERIMENT_DIR.parent / "pro_base_validation"
INIT_DIR = EXPERIMENT_DIR.parent / "pro_grassmannian_init"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"

LORA_RANK = 16
LORA_SCALE = 5.0  # NOT 20! Finding #329-330: scale=5 preserves MMLU
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def log(m):
    print(m, flush=True)


def cleanup(*o):
    for x in o:
        del x
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ---- Adapter utilities ----

class RuntimeLoRA(nn.Module):
    """y = base(x) + alpha * (x @ A) @ B"""
    def __init__(self, base, A, B, alpha):
        super().__init__()
        self.base = base
        self.lora_a = A.astype(mx.bfloat16)
        self.lora_b = B.astype(mx.bfloat16)
        self.alpha = alpha
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        y = self.base(x)
        return y + ((x @ self.lora_a) @ self.lora_b * self.alpha).astype(y.dtype)


def attach_adapter(model, skeleton, adapter_b, domain_idx, scale):
    """Attach adapter to all target keys via RuntimeLoRA wrapping."""
    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            B = adapter_b[bk].astype(mx.bfloat16)
            updates.append((key, RuntimeLoRA(m, A, B, scale)))
            count += 1
        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


def compose_adapters_nre(adapter_bs):
    """NRE composition: average B-matrices with norm rescaling (Finding #225)."""
    all_keys = set()
    for ab in adapter_bs:
        all_keys.update(ab.keys())

    composed = {}
    for key in all_keys:
        tensors = [ab[key].astype(mx.float32) for ab in adapter_bs if key in ab]
        if len(tensors) == 1:
            composed[key] = tensors[0].astype(mx.bfloat16)
            continue
        mean = sum(tensors) / len(tensors)
        source_norm = mx.mean(mx.stack([mx.linalg.norm(t.reshape(-1)) for t in tensors]))
        mean_norm = mx.linalg.norm(mean.reshape(-1))
        mx.eval(source_norm, mean_norm)
        if mean_norm.item() > 1e-8:
            composed[key] = (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
        else:
            composed[key] = mean.astype(mx.bfloat16)
    return composed


def dare_sparsify(adapter_b, p=0.5, seed=42):
    """DARE: randomly drop p fraction of B params, rescale by 1/(1-p). Finding #266."""
    rng = np.random.RandomState(seed)
    sparsified = {}
    for key, val in adapter_b.items():
        val_np = np.array(val)
        mask = rng.binomial(1, 1.0 - p, size=val_np.shape).astype(np.float32)
        sparsified[key] = mx.array(val_np * mask / (1.0 - p)).astype(mx.bfloat16)
    return sparsified


# ---- Benchmarks ----

# MMLU: balanced answer distribution (6A, 6B, 6C, 6D = 24 questions)
MMLU_QS = [
    # Answer A (6 questions)
    ("Habeas corpus protects against:", "A) Unlawful detention\nB) Self-incrimination\nC) Double jeopardy\nD) Cruel punishment", "A"),
    ("Which planet is closest to the Sun?", "A) Mercury\nB) Venus\nC) Earth\nD) Mars", "A"),
    ("The powerhouse of the cell is the:", "A) Mitochondria\nB) Nucleus\nC) Ribosome\nD) Golgi apparatus", "A"),
    ("HTML stands for:", "A) HyperText Markup Language\nB) High Transfer Machine Language\nC) HyperText Machine Learning\nD) High Text Markup Language", "A"),
    ("The boiling point of water at sea level is:", "A) 100 degrees Celsius\nB) 90 degrees Celsius\nC) 110 degrees Celsius\nD) 80 degrees Celsius", "A"),
    ("Which blood type is the universal donor?", "A) O negative\nB) A positive\nC) AB positive\nD) B negative", "A"),
    # Answer B (6 questions)
    ("What is the derivative of x^3?", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3", "B"),
    ("GDP measures:", "A) Government debt\nB) Total economic output\nC) Inflation rate\nD) Trade balance", "B"),
    ("In Python, a list is:", "A) Immutable\nB) Mutable\nC) A primitive type\nD) Fixed size", "B"),
    ("Newton's second law states F equals:", "A) mv\nB) ma\nC) mg\nD) mc^2", "B"),
    ("The chemical formula for water is:", "A) CO2\nB) H2O\nC) NaCl\nD) O2", "B"),
    ("Which organ produces insulin?", "A) Liver\nB) Pancreas\nC) Kidney\nD) Heart", "B"),
    # Answer C (6 questions)
    ("O(n log n) is the average complexity of:", "A) Bubble sort\nB) Linear search\nC) Merge sort\nD) Hash lookup", "C"),
    ("The largest organ in the human body is:", "A) Liver\nB) Brain\nC) Skin\nD) Heart", "C"),
    ("Which gas makes up most of Earth's atmosphere?", "A) Oxygen\nB) Carbon dioxide\nC) Nitrogen\nD) Argon", "C"),
    ("The speed of light in vacuum is approximately:", "A) 3x10^6 m/s\nB) 3x10^10 m/s\nC) 3x10^8 m/s\nD) 3x10^4 m/s", "C"),
    ("TCP stands for:", "A) Transfer Control Process\nB) Total Communication Protocol\nC) Transmission Control Protocol\nD) Technical Connection Protocol", "C"),
    ("Which amendment guarantees free speech?", "A) Second\nB) Fourth\nC) First\nD) Fifth", "C"),
    # Answer D (6 questions)
    ("The Pythagorean theorem relates:", "A) Angles of a triangle\nB) Area of a circle\nC) Volume of a sphere\nD) Sides of a right triangle", "D"),
    ("DNA stands for:", "A) DiNucleic Acid\nB) DeNatured Acid\nC) DiNitrogen Acid\nD) Deoxyribonucleic Acid", "D"),
    ("Which vitamin is produced by sunlight exposure?", "A) Vitamin A\nB) Vitamin B12\nC) Vitamin C\nD) Vitamin D", "D"),
    ("The SI unit of electrical resistance is:", "A) Ampere\nB) Volt\nC) Watt\nD) Ohm", "D"),
    ("Which data structure uses FIFO ordering?", "A) Stack\nB) Tree\nC) Hash map\nD) Queue", "D"),
    ("The Krebs cycle occurs in the:", "A) Nucleus\nB) Cytoplasm\nC) Endoplasmic reticulum\nD) Mitochondria", "D"),
]

GSM8K = [
    ("Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest for $2 each. How much daily?", "18"),
    ("A robe takes 2 bolts of blue fiber and half that much white. How many total?", "3"),
    ("If 5 shirts cost $100, how much do 8 shirts cost?", "160"),
    ("A train travels 60 mph for 3 hours. How far?", "180"),
    ("A store sells apples for $2 each. If you buy 5 and get a 10% discount, how much do you pay?", "9"),
    ("Tom has 3 times as many marbles as Jerry. Jerry has 4. How many does Tom have?", "12"),
    ("A baker makes 48 cookies. He puts them in boxes of 6. How many boxes?", "8"),
    ("Sarah earns $15 per hour. She works 8 hours a day, 5 days a week. Weekly earnings?", "600"),
    ("A rectangle has length 12 and width 5. What is its perimeter?", "34"),
    ("If a car uses 5 gallons per 100 miles, how many gallons for 350 miles?", "17"),  # 17.5 rounds
    ("A school has 360 students split equally into 12 classes. How many per class?", "30"),
    ("John buys 3 notebooks at $4 each and 2 pens at $2 each. Total cost?", "16"),
    ("A pool fills at 3 gallons per minute. How many gallons in 45 minutes?", "135"),
    ("If 8 workers can build a wall in 6 days, how many days for 12 workers?", "4"),
    ("A shirt costs $25 after a 50% discount. What was the original price?", "50"),
    ("Maria reads 30 pages per hour. A book has 450 pages. How many hours to finish?", "15"),
    ("A bus travels 240 miles in 4 hours. What is its average speed in mph?", "60"),
    ("If you save $50 per month for 2 years, how much total?", "1200"),
    ("A triangle has sides 3, 4, and 5. What is its perimeter?", "12"),
    ("A recipe needs 2 cups of flour for 12 cookies. How many cups for 36 cookies?", "6"),
]

CODE_PROMPTS = [
    "Write a Python function to compute factorial recursively.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to find the maximum element in a list.",
    "Write a Python class for a simple stack with push and pop.",
    "Write a Python function to reverse a linked list.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function to check if a number is prime.",
    "Write a Python function to flatten a nested list.",
    "Write a Python function to merge two sorted lists.",
    "Write a Python function to count vowels in a string.",
]


def eval_mmlu(model, tokenizer):
    """Logit-based MMLU evaluation. No generation needed."""
    correct = 0
    for q, choices, answer in MMLU_QS:
        prompt = f"Q: {q}\n{choices}\nAnswer:"
        tokens = tokenizer.encode(prompt)[:512]
        logits = model(mx.array(tokens)[None, :])
        mx.eval(logits)
        last = logits[0, -1]
        # Get logit for each answer letter
        preds = {}
        for letter in "ABCD":
            token_ids = tokenizer.encode(f" {letter}")
            if token_ids:
                preds[letter] = last[token_ids[0]].item()
        if preds and max(preds, key=preds.get) == answer:
            correct += 1
        del logits
    return correct, len(MMLU_QS)


def eval_gsm8k(model, tokenizer):
    """Generation-based math reasoning. Uses chat template."""
    correct = 0
    sampler = make_sampler(temp=0.0)
    for q, a in GSM8K:
        try:
            prompt = format_chat_prompt(tokenizer, f"{q}\nSolve step by step. Give only the final number.")
            out = mlx_generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=200, sampler=sampler, verbose=False,
            )
            # Look for numbers in the output, check last number
            nums = re.findall(r'[\d]+', out)
            if nums and nums[-1] == a:
                correct += 1
        except Exception:
            pass
    return correct, len(GSM8K)


def format_chat_prompt(tokenizer, user_msg):
    """Format using tokenizer's chat template if available, else basic format."""
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_msg}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"User: {user_msg}\nAssistant:"


def eval_code(model, tokenizer):
    """Generation + syntax parse for code. Uses native chat template."""
    correct = 0
    sampler = make_sampler(temp=0.0)
    for prompt in CODE_PROMPTS:
        try:
            formatted = format_chat_prompt(tokenizer, prompt)
            out = mlx_generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=300, sampler=sampler, verbose=False,
            )
            # Extract code from markdown blocks or raw output
            blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', out, re.DOTALL)
            code = '\n'.join(blocks) if blocks else '\n'.join(
                l for l in out.split('\n') if l.strip() and not l.startswith('#'))
            ast.parse(code)
            correct += 1
        except Exception:
            pass
    return correct, len(CODE_PROMPTS)


def run_benchmarks(model, tokenizer, config_name):
    """Run all 3 benchmarks, return results dict."""
    log(f"\n=== {config_name} ===")
    mmlu_c, mmlu_t = eval_mmlu(model, tokenizer)
    log(f"  MMLU: {mmlu_c}/{mmlu_t} ({mmlu_c/mmlu_t:.1%})")

    gsm_c, gsm_t = eval_gsm8k(model, tokenizer)
    log(f"  GSM8K: {gsm_c}/{gsm_t} ({gsm_c/gsm_t:.1%})")

    code_c, code_t = eval_code(model, tokenizer)
    log(f"  Code: {code_c}/{code_t} ({code_c/code_t:.1%})")

    return {
        "mmlu": {"correct": mmlu_c, "total": mmlu_t, "acc": round(mmlu_c / mmlu_t, 3)},
        "gsm8k": {"correct": gsm_c, "total": gsm_t, "acc": round(gsm_c / gsm_t, 3)},
        "code": {"correct": code_c, "total": code_t, "acc": round(code_c / code_t, 3)},
    }


def main():
    t0 = time.time()
    log("Pierre Pro: Benchmark Suite (scale=5 verification)")
    log("=" * 60)
    mx.random.seed(SEED)

    # Load model config
    base_data = {}
    if (BASE_DIR / "results.json").exists():
        base_data = json.loads((BASE_DIR / "results.json").read_text())
    model_id = base_data.get("model_id", "mlx-community/Qwen3-4B-4bit")
    log(f"Model: {model_id}")

    # Load skeleton
    skeleton_path = INIT_DIR / "grassmannian_skeleton_n5.npz"
    skeleton = {}
    if skeleton_path.exists():
        skeleton = dict(np.load(str(skeleton_path)))
        log(f"Skeleton loaded: {len(skeleton)} keys")
    else:
        log("WARNING: No skeleton found, adapter configs will be skipped")

    # Load all adapter B-matrices
    all_adapters = {}
    for domain in DOMAINS:
        p = ADAPTER_DIR / domain / "adapter.npz"
        if p.exists():
            all_adapters[domain] = dict(mx.load(str(p)))
    log(f"Adapters loaded: {list(all_adapters.keys())}")

    results = {"model_id": model_id, "scale": LORA_SCALE, "benchmarks": {}}

    # ---- Config 1: Base model (no adapters) ----
    model, tok = load(model_id)
    results["benchmarks"]["base"] = run_benchmarks(model, tok, "base")
    cleanup(model, tok)

    # ---- Config 2: Single math adapter at scale=5 ----
    if skeleton and "math" in all_adapters:
        model, tok = load(model_id)
        n_attached = attach_adapter(model, skeleton, all_adapters["math"], 2, LORA_SCALE)
        log(f"  Math adapter attached: {n_attached} modules")
        results["benchmarks"]["single_math_s5"] = run_benchmarks(model, tok, "single_math (scale=5)")
        cleanup(model, tok)

    # ---- Config 3: Composed N=5 NRE at scale=5 ----
    if skeleton and len(all_adapters) == 5:
        model, tok = load(model_id)
        composed = compose_adapters_nre(list(all_adapters.values()))
        n_attached = attach_adapter(model, skeleton, composed, 0, LORA_SCALE)
        log(f"  Composed N=5 attached: {n_attached} modules")
        results["benchmarks"]["composed_n5_s5"] = run_benchmarks(model, tok, "composed_n5 (scale=5)")
        del composed
        cleanup(model, tok)

    # ---- Config 4: Composed N=5 + DARE at scale=5 ----
    if skeleton and len(all_adapters) == 5:
        model, tok = load(model_id)
        dare_adapters = [dare_sparsify(ab, p=0.5, seed=SEED + i) for i, ab in enumerate(all_adapters.values())]
        composed_dare = compose_adapters_nre(dare_adapters)
        n_attached = attach_adapter(model, skeleton, composed_dare, 0, LORA_SCALE)
        log(f"  Composed N=5 DARE attached: {n_attached} modules")
        results["benchmarks"]["composed_n5_dare_s5"] = run_benchmarks(model, tok, "composed_n5_dare (scale=5)")
        del dare_adapters, composed_dare
        cleanup(model, tok)

    # ---- Analysis ----
    base = results["benchmarks"].get("base", {})
    log(f"\n{'='*60}")
    log(f"{'Config':<25} {'MMLU':>8} {'GSM8K':>8} {'Code':>8}")
    log(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for c, b in results["benchmarks"].items():
        log(f"{c:<25} {b['mmlu']['acc']:>8.1%} {b['gsm8k']['acc']:>8.1%} {b['code']['acc']:>8.1%}")

    # Deltas vs base
    log(f"\nDeltas vs base:")
    for c, b in results["benchmarks"].items():
        if c == "base":
            continue
        deltas = {}
        for bench in ["mmlu", "gsm8k", "code"]:
            d = b[bench]["acc"] - base.get(bench, {}).get("acc", 0)
            deltas[bench] = d
        log(f"  {c}: MMLU {deltas['mmlu']:+.1%} | GSM8K {deltas['gsm8k']:+.1%} | Code {deltas['code']:+.1%}")

    # Kill criteria: K822 — composed N=5 loses to base on ALL 3 benchmarks
    # "Loses" = strictly worse accuracy
    composed_key = "composed_n5_s5"
    if composed_key in results["benchmarks"]:
        composed = results["benchmarks"][composed_key]
        loses_all = all(
            composed[bench]["acc"] < base[bench]["acc"]
            for bench in ["mmlu", "gsm8k", "code"]
        )
        k822_pass = not loses_all  # PASS = does NOT lose all 3

        # S81: within 5pp on at least 2/3 benchmarks
        within_5pp = sum(
            abs(composed[bench]["acc"] - base[bench]["acc"]) <= 0.05
            for bench in ["mmlu", "gsm8k", "code"]
        )
        s81_pass = within_5pp >= 2
    else:
        k822_pass = False
        s81_pass = False
        loses_all = True
        within_5pp = 0

    results["kill_criteria"] = {
        "K822": {
            "pass": k822_pass,
            "detail": f"Composed N=5 loses ALL 3 benchmarks: {loses_all}",
        },
    }
    results["success_criteria"] = {
        "S81": {
            "pass": s81_pass,
            "detail": f"Within 5pp on {within_5pp}/3 benchmarks (threshold: >=2)",
        },
    }
    results["all_pass"] = k822_pass and s81_pass
    results["total_time_s"] = round(time.time() - t0, 1)

    log(f"\nK822 (not all worse): {'PASS' if k822_pass else 'FAIL'}")
    log(f"S81 (>=2/3 within 5pp): {'PASS' if s81_pass else 'FAIL'}")
    log(f"Total time: {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
