#!/usr/bin/env python3
"""
BitNet-2B Task-Based Evaluation of Ternary Composed Model

Tests whether composed ternary adapters improve TASK-BASED metrics (not just PPL).
This bridges the PPL-to-task gap that has been a critical weakness of the SOLE project.

Hypothesis: BitNet-2B ternary composed model improves task performance (not just PPL).

Kill criteria:
  K1: composed model scores WORSE than base on >40% of task-based metrics
  K2: math adapter shows <3pp accuracy improvement over base on math tasks

Design:
  - Reuse existing trained adapters from bitnet_multiseed_validation (seed 42, 400 steps)
  - 5 domains: medical, code, math, legal, creative
  - Domain-specific task metrics (not just PPL):
    * Math: MATH-500 subset - boxed answer extraction + numerical grading
    * Code: Python code generation - syntax validity (ast.parse)
    * Medical: Medical QA - keyword overlap F1 with reference answers
    * Legal: Legal QA - keyword overlap F1 with reference answers
    * Creative: Story continuation - next-token prediction quality (PPL on held-out)
  - Conditions: base, individual adapters, composed (1/N scaling)
  - All local on Apple Silicon via MLX, $0

Prior findings that transfer:
  - 1/N scaling resolves composition catastrophe (macro proven)
  - PPL does NOT predict task accuracy (micro proven, r=0.08)
  - Multiseed reproducibility: CV=0.5% (micro proven)
  - Ternary adapters compose 4.4% better than FP16 (micro proven)
"""

import ast
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg):
    print(msg, flush=True)


import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 128

ADAPTER_DIR = Path(__file__).parent.parent / "bitnet_multiseed_validation" / "adapters" / "seed42"
DATA_DIR = Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data"
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MATH500_PATH = Path(__file__).parent.parent / "reasoning_expert_distillation" / "math500_test.json"

DOMAINS = ["medical", "code", "math", "legal", "creative"]

# Eval sizes (small for Apple Silicon runtime)
MATH_EVAL_N = 20
CODE_EVAL_N = 20
QA_EVAL_N = 20
CREATIVE_EVAL_N = 25
MAX_GEN_TOKENS = 128  # Short to keep runtime fast


# ===========================================================================
# Ternary weight unpacking
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# TernaryLoRALinear (must match training code exactly)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# Text generation (simple autoregressive, no KV cache for reliability)
# ===========================================================================
def generate_text(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS, temperature=0.0):
    """Generate text token-by-token. Simple but reliable."""
    tokens = tokenizer.encode(prompt)
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[-MAX_SEQ_LENGTH:]

    generated = []
    for _ in range(max_tokens):
        x = mx.array(tokens + generated)[None, :]
        logits = model(x)
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = mx.argmax(next_logits, axis=-1)
        else:
            next_token = mx.random.categorical(next_logits / temperature)

        mx.eval(next_token)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        # Prevent runaway (limit total context)
        if len(tokens) + len(generated) > MAX_SEQ_LENGTH * 3:
            break

    return tokenizer.decode(generated)


# ===========================================================================
# MATH grading (from serverless_eval.py)
# ===========================================================================
RE_BOXED = re.compile(r"\\boxed\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def get_last_boxed(text):
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos - 1] if depth == 0 else text[start:]


def extract_final_answer(text):
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()[:50]


def normalize_text(text):
    text = text.strip()
    for s in ["\\$", "$", "\\left", "\\right", "\\,", "\\text{", "\\mathrm{",
              "\\(", "\\)", "\\{", "\\}"]:
        text = text.replace(s, "")
    text = text.replace("\\ ", " ").replace("\\dfrac", "\\frac")
    text = text.rstrip("}")
    return " ".join(text.split())


def grade_math_answer(predicted, ground_truth):
    pred, gt = normalize_text(predicted), normalize_text(ground_truth)
    if pred == gt:
        return True
    try:
        if "/" in pred and "/" not in gt:
            p = pred.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(gt)) < 1e-6
        elif "/" in gt and "/" not in pred:
            p = gt.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(pred)) < 1e-6
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ===========================================================================
# Domain-specific evaluations
# ===========================================================================
def eval_math(model, tokenizer, n_problems=MATH_EVAL_N):
    """MATH-500 subset: accuracy of extracted boxed answers."""
    if not MATH500_PATH.exists():
        log("  Downloading math500_test.json...")
        import urllib.request
        url = ("https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
               "main/ch03/01_main-chapter-code/math500_test.json")
        MATH500_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, MATH500_PATH)

    with open(MATH500_PATH) as f:
        problems = json.load(f)

    # Pick easiest problems (levels 1-3) since 2B model is small
    easy = [p for p in problems if p.get("level", 5) <= 3]
    selected = easy[:n_problems]
    if len(selected) < n_problems:
        selected = problems[:n_problems]
    log(f"  Math: {len(selected)} problems (levels {set(p.get('level',0) for p in selected)})")

    correct = 0
    details = []
    for i, prob in enumerate(selected):
        prompt = f"Solve: {prob['problem']}\nAnswer:"
        response = generate_text(model, tokenizer, prompt)
        predicted = extract_final_answer(response)
        is_correct = grade_math_answer(predicted, prob["answer"])
        correct += int(is_correct)
        details.append({
            "idx": i, "level": prob.get("level"),
            "predicted": predicted[:30], "answer": prob["answer"][:30],
            "correct": is_correct,
        })
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(selected)}] acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(selected) if selected else 0
    return {"metric": "accuracy", "value": round(acc, 4),
            "correct": correct, "total": len(selected),
            "pct": round(acc * 100, 1), "details": details}


def eval_code(model, tokenizer, n_prompts=CODE_EVAL_N):
    """Code generation: syntax validity of generated Python."""
    data_path = DATA_DIR / "code" / "valid.jsonl"
    if not data_path.exists():
        return {"metric": "syntax_valid_rate", "value": 0, "total": 0, "error": "no data"}

    texts = []
    with open(data_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    prompts = []
    for text in texts[:n_prompts * 3]:
        lines = text.strip().split('\n')
        if len(lines) >= 3:
            n_ctx = max(2, len(lines) // 3)
            prompts.append({"prompt": '\n'.join(lines[:n_ctx]),
                          "reference": '\n'.join(lines[n_ctx:])})
        if len(prompts) >= n_prompts:
            break

    if not prompts:
        return {"metric": "syntax_valid_rate", "value": 0, "total": 0, "error": "no prompts"}

    valid = 0
    details = []
    for i, item in enumerate(prompts):
        gen_prompt = f"```python\n{item['prompt']}\n"
        response = generate_text(model, tokenizer, gen_prompt, max_tokens=MAX_GEN_TOKENS)
        code = item["prompt"] + "\n" + response
        is_valid = False
        if len(response.strip()) > 3:
            try:
                ast.parse(code)
                is_valid = True
            except SyntaxError:
                try:
                    ast.parse(response)
                    is_valid = True
                except SyntaxError:
                    pass
        valid += int(is_valid)
        details.append({"idx": i, "valid": is_valid, "len": len(response)})
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(prompts)}] valid={valid/(i+1)*100:.1f}%")

    rate = valid / len(prompts)
    return {"metric": "syntax_valid_rate", "value": round(rate, 4),
            "valid": valid, "total": len(prompts),
            "pct": round(rate * 100, 1), "details": details}


def compute_keyword_f1(prediction, reference):
    def tokenize(text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 2]
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((pred_tokens & ref_tokens).values())
    prec = common / sum(pred_tokens.values()) if pred_tokens else 0
    rec = common / sum(ref_tokens.values()) if ref_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def eval_qa_domain(model, tokenizer, domain_name, n_questions=QA_EVAL_N):
    """QA evaluation using keyword F1 with reference answers."""
    data_path = DATA_DIR / domain_name / "valid.jsonl"
    if not data_path.exists():
        return {"metric": "keyword_f1", "value": 0, "total": 0, "error": "no data"}

    texts = []
    with open(data_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    qa_pairs = []
    for text in texts:
        sents = re.split(r'[.!?]\s+', text.strip())
        if len(sents) >= 2 and len(sents[0]) > 15:
            q = sents[0].strip()
            r = ' '.join(sents[1:]).strip()
            if len(r) > 20:
                qa_pairs.append({"question": q, "reference": r})
        if len(qa_pairs) >= n_questions:
            break

    if len(qa_pairs) < 5:
        # Fallback: split text chunks
        for text in texts[:n_questions]:
            words = text.split()
            if len(words) > 10:
                qa_pairs.append({
                    "question": ' '.join(words[:len(words)//3]),
                    "reference": ' '.join(words[len(words)//3:])
                })
        qa_pairs = qa_pairs[:n_questions]

    if not qa_pairs:
        return {"metric": "keyword_f1", "value": 0, "total": 0, "error": "no QA pairs"}

    f1_scores = []
    details = []
    for i, qa in enumerate(qa_pairs):
        prompt = f"Question: {qa['question']}\nAnswer:"
        response = generate_text(model, tokenizer, prompt)
        f1 = compute_keyword_f1(response, qa["reference"])
        f1_scores.append(f1)
        details.append({"idx": i, "f1": round(f1, 4), "len": len(response)})
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(qa_pairs)}] avg_f1={sum(f1_scores)/len(f1_scores):.3f}")

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return {"metric": "keyword_f1", "value": round(mean_f1, 4),
            "total": len(qa_pairs), "pct": round(mean_f1 * 100, 1), "details": details}


def eval_creative_ppl(model, tokenizer, n_texts=CREATIVE_EVAL_N):
    """Creative writing via PPL on held-out text (lower = better)."""
    data_path = DATA_DIR / "creative" / "valid.jsonl"
    if not data_path.exists():
        return {"metric": "creative_ppl", "value": float("inf"), "total": 0, "error": "no data"}

    texts = []
    with open(data_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0
    for text in texts[:n_texts]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 4:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return {"metric": "creative_ppl", "value": float("inf"), "total": 0}

    avg_ppl = math.exp(min(total_loss / total_tokens, 100))
    return {"metric": "creative_ppl", "value": round(avg_ppl, 4),
            "total": n_texts, "total_tokens": total_tokens}


# ===========================================================================
# Run all domain evals for one condition
# ===========================================================================
def run_condition(model, tokenizer, condition_name):
    log(f"\n  === Evaluating: {condition_name} ===")
    results = {}

    log(f"  [math] MATH-500 subset...")
    results["math"] = eval_math(model, tokenizer)
    log(f"    -> accuracy: {results['math']['pct']}%")

    log(f"  [code] Python syntax validation...")
    results["code"] = eval_code(model, tokenizer)
    log(f"    -> syntax_valid_rate: {results['code']['pct']}%")

    log(f"  [medical] Medical QA F1...")
    results["medical"] = eval_qa_domain(model, tokenizer, "medical")
    log(f"    -> keyword_f1: {results['medical']['pct']}%")

    log(f"  [legal] Legal QA F1...")
    results["legal"] = eval_qa_domain(model, tokenizer, "legal")
    log(f"    -> keyword_f1: {results['legal']['pct']}%")

    log(f"  [creative] Story PPL...")
    results["creative"] = eval_creative_ppl(model, tokenizer)
    log(f"    -> creative_ppl: {results['creative']['value']}")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    results = {
        "experiment": "bitnet_task_eval",
        "model": MODEL_ID,
        "hypothesis": "BitNet-2B ternary composed model improves task performance",
        "adapter_source": str(ADAPTER_DIR),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("BitNet-2B Task-Based Evaluation")
    log("=" * 70)

    # Phase 0: Load model
    log("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")

    log("  Unpacking ternary weights...")
    t1 = time.time()
    model = replace_bitlinear_with_linear(model)
    log(f"  Unpacked in {time.time() - t1:.1f}s")

    # Check adapters
    for domain in DOMAINS:
        p = ADAPTER_DIR / domain / "adapter.npz"
        if not p.exists():
            log(f"  FATAL: {p} not found")
            return
    log(f"  All 5 adapters found")

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)
    log(f"  Loaded {len(adapters)} adapters")

    # Phase 1: Apply LoRA structure
    log("\n[Phase 1] Applying LoRA structure...")
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Phase 2: Base model evaluation
    log("\n[Phase 2] BASE model evaluation...")
    zero_lora_params(model)
    mx.eval(model.parameters())
    base_results = run_condition(model, tokenizer, "base")
    results["base"] = base_results

    # Phase 3: Individual adapter evaluation (own domain only)
    log("\n[Phase 3] Individual adapter evaluation...")
    individual_results = {}
    for domain in DOMAINS:
        log(f"\n  Loading {domain} adapter...")
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain])
        mx.eval(model.parameters())

        r = {}
        if domain == "math":
            r["math"] = eval_math(model, tokenizer)
            log(f"    {domain} math acc: {r['math']['pct']}%")
        elif domain == "code":
            r["code"] = eval_code(model, tokenizer)
            log(f"    {domain} code valid: {r['code']['pct']}%")
        elif domain == "medical":
            r["medical"] = eval_qa_domain(model, tokenizer, "medical")
            log(f"    {domain} med f1: {r['medical']['pct']}%")
        elif domain == "legal":
            r["legal"] = eval_qa_domain(model, tokenizer, "legal")
            log(f"    {domain} legal f1: {r['legal']['pct']}%")
        elif domain == "creative":
            r["creative"] = eval_creative_ppl(model, tokenizer)
            log(f"    {domain} creative ppl: {r['creative']['value']}")
        individual_results[domain] = r

    results["individual"] = individual_results

    # Phase 4: Composed model (1/N scaling)
    log("\n[Phase 4] COMPOSED model (1/N) evaluation...")
    merged = compose_adapters(list(adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())
    composed_results = run_condition(model, tokenizer, "composed_1_over_N")
    results["composed"] = composed_results

    # Phase 5: Kill criteria
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    comparisons = {}

    # Math: higher accuracy is better
    b = base_results["math"]["value"]
    c = composed_results["math"]["value"]
    comparisons["math"] = {"base": b, "composed": c, "better": c >= b, "delta": round(c - b, 4)}

    # Code: higher valid rate is better
    b = base_results["code"]["value"]
    c = composed_results["code"]["value"]
    comparisons["code"] = {"base": b, "composed": c, "better": c >= b, "delta": round(c - b, 4)}

    # Medical: higher F1 is better
    b = base_results["medical"]["value"]
    c = composed_results["medical"]["value"]
    comparisons["medical"] = {"base": b, "composed": c, "better": c >= b, "delta": round(c - b, 4)}

    # Legal: higher F1 is better
    b = base_results["legal"]["value"]
    c = composed_results["legal"]["value"]
    comparisons["legal"] = {"base": b, "composed": c, "better": c >= b, "delta": round(c - b, 4)}

    # Creative: lower PPL is better
    b = base_results["creative"]["value"]
    c = composed_results["creative"]["value"]
    comparisons["creative"] = {"base": b, "composed": c, "better": c <= b, "delta": round(c - b, 4)}

    n_worse = sum(1 for v in comparisons.values() if not v["better"])
    n_total = len(comparisons)
    worse_pct = n_worse / n_total * 100

    k1_pass = worse_pct <= 40
    log(f"\n  K1: Composed worse than base on <= 40% of metrics")
    log(f"      Worse on {n_worse}/{n_total} ({worse_pct:.0f}%)")
    for d, comp in comparisons.items():
        st = "BETTER" if comp["better"] else "WORSE"
        log(f"        {d}: base={comp['base']:.4f} composed={comp['composed']:.4f} delta={comp['delta']:+.4f} [{st}]")
    log(f"      Verdict: {'PASS' if k1_pass else 'KILL'}")

    # K2: math adapter >= 3pp over base
    base_math_acc = base_results["math"]["value"]
    math_adapter_acc = individual_results.get("math", {}).get("math", {}).get("value", base_math_acc)
    math_pp = (math_adapter_acc - base_math_acc) * 100
    k2_pass = math_pp >= 3.0

    log(f"\n  K2: Math adapter improvement >= 3pp over base")
    log(f"      Base: {base_math_acc*100:.1f}%  Math adapter: {math_adapter_acc*100:.1f}%")
    log(f"      Improvement: {math_pp:+.1f}pp")
    log(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")

    results["comparisons"] = comparisons
    results["kill_criteria"] = {
        "K1": {"n_worse": n_worse, "n_total": n_total, "worse_pct": round(worse_pct, 1), "pass": k1_pass},
        "K2": {"base_acc": round(base_math_acc * 100, 1), "adapter_acc": round(math_adapter_acc * 100, 1),
               "improvement_pp": round(math_pp, 1), "pass": k2_pass},
    }

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    if not k1_pass:
        verdict += " (K1: composed worse on too many metrics)"
    if not k2_pass:
        verdict += " (K2: math adapter < 3pp improvement)"

    results["verdict"] = verdict
    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    log(f"\n  VERDICT: {verdict}")
    log(f"  Total time: {total_time/60:.1f} min")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
