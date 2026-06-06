#!/usr/bin/env python3
"""Expert promotion: freeze solidified expert into base, verify no degradation.

Kill criteria:
  K839: Promoted expert loses >30% quality
  K840: New adapters fail to converge on promoted base

Procedure:
  1. Measure base model: MMLU, medical PPL, behavioral
  2. Promote medical adapter: base_new = base + scale * B^T @ A^T
  3. Verify medical quality preserved on promoted base (PPL, behavioral)
  4. Verify MMLU preserved on promoted base
  5. Train 2 new domain adapters (code, math) on promoted base
  6. Compare convergence speed and quality vs original base training

Finding #320/#330: scale=5 gives 0pp MMLU degradation for single adapter.
Finding #331: sequential promotion from random init = catastrophic. Pre-trained base is key.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from pierre.bench import Experiment, mmlu_eval, ppl, cleanup

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
SKELETON_PATH = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
LORA_RANK = 16
TRAIN_SCALE = 20.0   # original training scale
PROMOTE_SCALE = 5.0   # promotion scale (safe per Finding #320/#330)
NEW_ADAPTER_SCALE = 20.0  # scale for training new adapters on promoted base
TRAIN_ITERS = 300
LEARNING_RATE = 1e-4
MAX_SEQ = 256
SEED = 42
RESPONSE_MARKER = "### Response:\n"

DOMAINS = ["medical", "code", "math", "legal", "finance"]
PROMOTE_DOMAIN = "medical"  # domain to promote
RETRAIN_DOMAINS = ["code", "math"]  # domains to retrain on promoted base

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Same 50 MMLU questions as solidified_composition_mmlu for consistency
MMLU_QUESTIONS = [
    ("physics", "A 2 kg object at 3 m/s collides with a stationary 1 kg object in a perfectly inelastic collision. What is the speed after collision?", "A) 1 m/s\nB) 2 m/s\nC) 3 m/s\nD) 6 m/s", "B"),
    ("physics", "What is the SI unit of electrical resistance?", "A) Volt\nB) Ampere\nC) Ohm\nD) Watt", "C"),
    ("physics", "According to Newton's second law, force equals:", "A) mass times velocity\nB) mass times acceleration\nC) mass times distance\nD) mass times time", "B"),
    ("physics", "What is the speed of light in a vacuum?", "A) 3 x 10^6 m/s\nB) 3 x 10^7 m/s\nC) 3 x 10^8 m/s\nD) 3 x 10^9 m/s", "C"),
    ("physics", "What is the unit of frequency?", "A) Watt\nB) Joule\nC) Hertz\nD) Pascal", "C"),
    ("physics", "A ball is dropped from rest. After 2 seconds of free fall (g=10 m/s^2), its speed is:", "A) 10 m/s\nB) 20 m/s\nC) 30 m/s\nD) 40 m/s", "B"),
    ("chemistry", "What is the molecular formula of glucose?", "A) C6H12O6\nB) C12H22O11\nC) CH3COOH\nD) C2H5OH", "A"),
    ("chemistry", "Which element has the highest electronegativity?", "A) Oxygen\nB) Chlorine\nC) Fluorine\nD) Nitrogen", "C"),
    ("chemistry", "What is the pH of pure water at 25 degrees Celsius?", "A) 0\nB) 1\nC) 7\nD) 14", "C"),
    ("chemistry", "What is the atomic number of carbon?", "A) 4\nB) 6\nC) 8\nD) 12", "B"),
    ("biology", "Which organelle produces ATP in eukaryotic cells?", "A) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi apparatus", "C"),
    ("biology", "What type of bond holds DNA strands together?", "A) Covalent bonds\nB) Ionic bonds\nC) Hydrogen bonds\nD) Metallic bonds", "C"),
    ("biology", "What is the powerhouse of the cell?", "A) Nucleus\nB) Mitochondria\nC) Chloroplast\nD) Endoplasmic reticulum", "B"),
    ("biology", "Which molecule carries amino acids to the ribosome during translation?", "A) mRNA\nB) rRNA\nC) tRNA\nD) DNA", "C"),
    ("math", "What is the derivative of x^3?", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3", "B"),
    ("math", "If log base 2 of x equals 5, what is x?", "A) 10\nB) 25\nC) 32\nD) 64", "C"),
    ("math", "What is the sum of the interior angles of a hexagon?", "A) 360 degrees\nB) 540 degrees\nC) 720 degrees\nD) 900 degrees", "C"),
    ("math", "If f(x) = 2x + 3, what is f(f(1))?", "A) 7\nB) 13\nC) 11\nD) 9", "B"),
    ("math", "What is the value of pi to two decimal places?", "A) 3.12\nB) 3.14\nC) 3.16\nD) 3.18", "B"),
    ("computer_science", "What is the time complexity of binary search?", "A) O(1)\nB) O(n)\nC) O(log n)\nD) O(n log n)", "C"),
    ("computer_science", "Which data structure uses FIFO ordering?", "A) Stack\nB) Queue\nC) Binary tree\nD) Hash table", "B"),
    ("computer_science", "What does SQL stand for?", "A) Structured Query Language\nB) Sequential Query Logic\nC) Standard Query Library\nD) System Query Language", "A"),
    ("computer_science", "In a binary search tree, worst-case search time is:", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
    ("history", "In what year did World War II end?", "A) 1943\nB) 1944\nC) 1945\nD) 1946", "C"),
    ("history", "Who was the first President of the United States?", "A) Thomas Jefferson\nB) John Adams\nC) Benjamin Franklin\nD) George Washington", "D"),
    ("history", "The French Revolution began in:", "A) 1776\nB) 1789\nC) 1799\nD) 1804", "B"),
    ("history", "The Berlin Wall fell in:", "A) 1987\nB) 1988\nC) 1989\nD) 1990", "C"),
    ("history", "Who discovered penicillin?", "A) Louis Pasteur\nB) Alexander Fleming\nC) Joseph Lister\nD) Robert Koch", "B"),
    ("philosophy", "Who wrote 'The Republic'?", "A) Aristotle\nB) Socrates\nC) Plato\nD) Epicurus", "C"),
    ("philosophy", "The categorical imperative is associated with:", "A) John Stuart Mill\nB) Immanuel Kant\nC) David Hume\nD) Friedrich Nietzsche", "B"),
    ("philosophy", "Cogito ergo sum was stated by:", "A) Descartes\nB) Locke\nC) Spinoza\nD) Leibniz", "A"),
    ("literature", "Who wrote Romeo and Juliet?", "A) Charles Dickens\nB) William Shakespeare\nC) Jane Austen\nD) Mark Twain", "B"),
    ("literature", "Who wrote 1984?", "A) Aldous Huxley\nB) George Orwell\nC) Ray Bradbury\nD) H.G. Wells", "B"),
    ("literature", "In which century was Don Quixote first published?", "A) 15th\nB) 16th\nC) 17th\nD) 18th", "C"),
    ("economics", "What does GDP stand for?", "A) General Domestic Product\nB) Gross Domestic Product\nC) Gross Domestic Profit\nD) General Domestic Profit", "B"),
    ("economics", "According to the law of demand, as price increases:", "A) quantity demanded increases\nB) quantity demanded decreases\nC) supply increases\nD) supply decreases", "B"),
    ("economics", "Inflation is defined as:", "A) A decrease in the general price level\nB) An increase in the general price level\nC) A decrease in unemployment\nD) An increase in GDP", "B"),
    ("psychology", "Who is the father of psychoanalysis?", "A) Carl Jung\nB) B.F. Skinner\nC) Sigmund Freud\nD) Ivan Pavlov", "C"),
    ("psychology", "Classical conditioning was discovered by:", "A) B.F. Skinner\nB) Ivan Pavlov\nC) John Watson\nD) Albert Bandura", "B"),
    ("psychology", "Maslow's hierarchy places which need at the base?", "A) Self-actualization\nB) Esteem\nC) Safety\nD) Physiological", "D"),
    ("geography", "What is the largest ocean on Earth?", "A) Atlantic\nB) Indian\nC) Arctic\nD) Pacific", "D"),
    ("geography", "Which continent has the most countries?", "A) Asia\nB) Europe\nC) Africa\nD) South America", "C"),
    ("geography", "What is the longest river in the world?", "A) Amazon\nB) Nile\nC) Mississippi\nD) Yangtze", "B"),
    ("law", "Habeas corpus protects against:", "A) Double jeopardy\nB) Unlawful detention\nC) Self-incrimination\nD) Cruel punishment", "B"),
    ("medicine", "What organ produces insulin?", "A) Liver\nB) Kidney\nC) Pancreas\nD) Spleen", "C"),
    ("medicine", "Normal resting heart rate for adults (bpm)?", "A) 40-60\nB) 60-100\nC) 100-120\nD) 120-140", "B"),
    ("engineering", "Ohm's law is:", "A) V = IR\nB) F = ma\nC) E = mc^2\nD) P = IV", "A"),
    ("astronomy", "Which planet is the Red Planet?", "A) Venus\nB) Mars\nC) Jupiter\nD) Saturn", "B"),
    ("astronomy", "How many planets in our solar system?", "A) 7\nB) 8\nC) 9\nD) 10", "B"),
    ("nutrition", "Which vitamin is produced by sunlight exposure?", "A) Vitamin A\nB) Vitamin B12\nC) Vitamin C\nD) Vitamin D", "D"),
]

# Behavioral prompts for quality assessment
BEHAVIORAL_PROMPTS = {
    "medical": [
        "### Instruction:\nWhat are the common symptoms of type 2 diabetes?\n\n### Response:\n",
        "### Instruction:\nExplain the mechanism of action of metformin.\n\n### Response:\n",
        "### Instruction:\nWhat is the difference between a CT scan and an MRI?\n\n### Response:\n",
    ],
    "code": [
        "### Instruction:\nWrite a Python function to check if a string is a palindrome.\n\n### Response:\n",
        "### Instruction:\nExplain the difference between a list and a tuple in Python.\n\n### Response:\n",
        "### Instruction:\nWrite a function to find the factorial of a number using recursion.\n\n### Response:\n",
    ],
    "math": [
        "### Instruction:\nSolve: If a train travels 120 km in 2 hours, what is its average speed?\n\n### Response:\n",
        "### Instruction:\nWhat is the area of a circle with radius 7 cm?\n\n### Response:\n",
        "### Instruction:\nSimplify the expression: 3x + 5 - x + 2\n\n### Response:\n",
    ],
}

DOMAIN_KEYWORDS = {
    "medical": ["symptoms", "diagnosis", "treatment", "patient", "disease",
                "medication", "clinical", "therapy", "blood", "organ",
                "insulin", "glucose", "scan", "imaging", "doctor"],
    "code": ["def ", "return", "function", "class ", "import", "for ",
             "if ", "while", "print", "variable", "string", "list",
             "parameter", "argument", "recursion"],
    "math": ["=", "answer", "solve", "calculate", "total", "number",
             "equation", "formula", "area", "speed", "km", "radius",
             "pi", "multiply", "divide"],
}

REFUSAL_PHRASES = ["I don't know", "I cannot", "as an AI", "I'm sorry",
                   "I'm unable", "I am not able"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if hasattr(o, 'item'): return o.item()
        return super().default(o)


def log(m):
    print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


def score_behavioral(text, domain):
    """Score domain-specific behavioral quality 0-1 (from pro_sft_5_adapters)."""
    if not text or len(text.strip()) < 20:
        return 0.0
    score = 0.2
    text_lower = text.lower()
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    keyword_hits = sum(1 for k in keywords if k.lower() in text_lower)
    score += min(keyword_hits * 0.06, 0.5)
    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in text_lower:
            score -= 0.15
    if len(text.strip()) > 100:
        score += 0.1
    if text.count('.') > 1 or text.count('\n') > 1:
        score += 0.1
    words = text.split()
    if len(words) > 30:
        from collections import Counter
        ngram_counts = Counter()
        for i in range(len(words) - 9):
            ngram = " ".join(words[i:i+10])
            ngram_counts[ngram] += 1
        if ngram_counts and max(ngram_counts.values()) > 3:
            score -= 0.2
    return max(0.0, min(1.0, score))


def evaluate_behavioral(model, tokenizer, domain):
    """Generate responses and score behavioral quality for a domain."""
    prompts = BEHAVIORAL_PROMPTS.get(domain, [])
    if not prompts:
        return {"score": 0.0, "responses": [], "n_prompts": 0}
    sampler = make_sampler(temp=0.0)
    scores = []
    responses = []
    for prompt in prompts:
        try:
            generated = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=200, sampler=sampler, verbose=False,
            )
            s = score_behavioral(generated, domain)
            scores.append(s)
            responses.append({
                "prompt": prompt.split("### Response:")[0].strip()[-80:],
                "response_preview": generated[:200],
                "score": round(s, 2),
            })
        except Exception as e:
            log(f"  WARNING: generation failed for {domain}: {e}")
            scores.append(0.0)
            responses.append({"prompt": prompt[:80], "response_preview": str(e), "score": 0.0})
    mean_score = float(np.mean(scores)) if scores else 0.0
    return {"score": round(mean_score, 3), "responses": responses, "n_prompts": len(prompts)}


def load_data(domain, split="valid", n=None):
    texts = []
    path = DATA_DIR / domain / f"{split}.jsonl"
    if not path.exists():
        log(f"  WARNING: data file not found: {path}")
        return []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if n and len(texts) >= n:
                break
    return texts


def tokenize_sft(text, tokenizer, max_len=256):
    """Tokenize with SFT mask: 1 for response tokens, 0 for instruction."""
    response_idx = text.find(RESPONSE_MARKER)
    if response_idx < 0:
        tokens = tokenizer.encode(text)[:max_len]
        return tokens, [1] * len(tokens)
    instruction_part = text[:response_idx + len(RESPONSE_MARKER)]
    instruction_len = len(tokenizer.encode(instruction_part))
    full_tokens = tokenizer.encode(text)[:max_len]
    mask = [0] * min(instruction_len, len(full_tokens))
    mask += [1] * (len(full_tokens) - len(mask))
    return full_tokens, mask


def sft_loss_fn(model, tokens, mask):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    response_mask = mask[:, 1:]
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    return masked_loss.sum() / n_response


# ── QLoRA with frozen Grassmannian A ──────────────────────────────────────

class GrassmannianLoRALinear(nn.Module):
    """QLoRA with frozen Grassmannian A-matrix and trainable B-matrix.

    Can wrap QuantizedLinear, Linear, or another GrassmannianLoRALinear
    (for stacking promoted adapter + new trainable adapter).
    """
    def __init__(self, base_linear, rank=16, scale=20.0, a_init=None):
        super().__init__()
        # Detect dimensions from the base layer, handling various types
        if isinstance(base_linear, GrassmannianLoRALinear):
            # Wrapping an existing LoRA — get dims from its A-matrix
            input_dims = base_linear.lora_a.shape[0]
            output_dims = base_linear.lora_b.shape[1]
        elif hasattr(base_linear, 'weight'):
            output_dims, packed_input = base_linear.weight.shape
            if isinstance(base_linear, nn.QuantizedLinear):
                input_dims = packed_input * (32 // base_linear.bits)
            else:
                input_dims = packed_input
        else:
            raise TypeError(f"Cannot wrap {type(base_linear)} — no weight or lora_a attribute")
        self.linear = base_linear
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(input_dims)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(input_dims, rank))
        self.lora_b = mx.zeros((rank, output_dims))
        self.scale = scale
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return base_out + (self.scale * z).astype(x.dtype)


# ── Phase functions (each self-contained for memory isolation) ─────────────

def phase_base_measurements(skeleton):
    """Phase 1: Measure base model quality (MMLU, medical PPL, behavioral)."""
    log("\n" + "=" * 60)
    log("Phase 1: Base model measurements")
    log("=" * 60)

    model, tok = load(MODEL_ID)

    # MMLU
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    base_mmlu = correct / total if total else 0
    log(f"  Base MMLU: {base_mmlu:.1%} ({correct}/{total})")

    # Medical PPL
    med_texts = load_data("medical", "valid", 25)
    base_med_ppl = ppl(model, tok, med_texts) if med_texts else float('inf')
    log(f"  Base medical PPL: {base_med_ppl:.3f}")

    # Code PPL (for new adapter comparison)
    code_texts = load_data("code", "valid", 25)
    base_code_ppl = ppl(model, tok, code_texts) if code_texts else float('inf')
    log(f"  Base code PPL: {base_code_ppl:.3f}")

    # Math PPL
    math_texts = load_data("math", "valid", 25)
    base_math_ppl = ppl(model, tok, math_texts) if math_texts else float('inf')
    log(f"  Base math PPL: {base_math_ppl:.3f}")

    # Medical behavioral
    base_med_behavioral = evaluate_behavioral(model, tok, "medical")
    log(f"  Base medical behavioral: {base_med_behavioral['score']:.3f}")

    log_memory("post-base-measurements")
    cleanup(model, tok)
    return {
        "mmlu": base_mmlu,
        "mmlu_correct": correct,
        "mmlu_total": total,
        "mmlu_per_subject": per_subject,
        "medical_ppl": base_med_ppl,
        "code_ppl": base_code_ppl,
        "math_ppl": base_math_ppl,
        "medical_behavioral": base_med_behavioral,
    }


def phase_promote_and_measure(skeleton, base_results):
    """Phase 2: Promote medical adapter into base, measure quality.

    Promotion: For each target module in each layer, compute
      W' = W + scale * B^T @ A^T
    and replace the base weight with W'.
    """
    log("\n" + "=" * 60)
    log(f"Phase 2: Promote {PROMOTE_DOMAIN} adapter at scale={PROMOTE_SCALE}")
    log("=" * 60)

    # Load medical adapter B-matrices
    adapter_path = ADAPTER_DIR / PROMOTE_DOMAIN / "adapter.npz"
    if not adapter_path.exists():
        log(f"  ERROR: adapter not found: {adapter_path}")
        return None
    adapter_b = dict(mx.load(str(adapter_path)))
    log(f"  Loaded adapter: {len(adapter_b)} B-matrices")

    # Compute promotion delta norms for analysis
    domain_idx = DOMAINS.index(PROMOTE_DOMAIN)
    delta_norms = []

    model, tok = load(MODEL_ID)

    # Promotion: modify base weights in-place
    n_promoted = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue

            # Get base module
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            A = mx.array(skeleton[ak]).astype(mx.float32)  # (in, rank)
            B = adapter_b[bk].astype(mx.float32)            # (rank, out)

            # Delta = scale * B^T @ A^T = scale * (A @ B)^T
            # LoRA computes: y = x @ A @ B * scale
            # So the effective weight delta is: scale * (A @ B).T for W^T
            # In the weight matrix: W' = W + scale * (B.T @ A.T) applied as additive
            # Since y = x @ W (quantized), and LoRA adds x @ A @ B * scale,
            # the effective delta to the weight is: delta = (A @ B * scale)
            # which gets transposed depending on convention.
            #
            # For QuantizedLinear, we can't modify the quantized weights directly.
            # Instead, we'll use a permanent LoRA with frozen B as the "promoted" adapter.

            delta = PROMOTE_SCALE * (A @ B)  # (in, out)
            delta_norm = mx.linalg.norm(delta.reshape(-1))
            mx.eval(delta_norm)
            delta_norms.append(delta_norm.item())
            n_promoted += 1
            del delta, delta_norm

    log(f"  Computing promotion via permanent LoRA attachment (QuantizedLinear)")
    log(f"  Modules to promote: {n_promoted}")
    if delta_norms:
        mean_norm = np.mean(delta_norms)
        max_norm = np.max(delta_norms)
        log(f"  Delta norms: mean={mean_norm:.4f}, max={max_norm:.4f}")
    else:
        mean_norm = 0.0
        max_norm = 0.0

    # For QuantizedLinear, we cannot directly modify weights.
    # Instead, we attach the adapter as a FROZEN LoRA (both A and B frozen).
    # This is mathematically equivalent to W' = W + delta.
    cleanup(model, tok)
    del adapter_b
    gc.collect()
    mx.clear_cache()

    # Reload and attach frozen LoRA as "promoted" adapter
    model, tok = load(MODEL_ID)
    adapter_b = dict(mx.load(str(adapter_path)))

    n_attached = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            B = adapter_b[bk].astype(mx.bfloat16)

            # Create a frozen LoRA layer (both A and B frozen = permanent promotion)
            lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=PROMOTE_SCALE, a_init=A)
            lora.lora_b = B
            lora.freeze()  # Freeze everything including B
            updates.append((key, lora))
            n_attached += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    model.freeze()  # Everything frozen — this is the "promoted base"
    log(f"  Attached {n_attached} frozen LoRA modules as promotion")

    # Now measure quality on the promoted model
    # Medical PPL
    med_texts = load_data("medical", "valid", 25)
    promoted_med_ppl = ppl(model, tok, med_texts) if med_texts else float('inf')
    log(f"  Promoted medical PPL: {promoted_med_ppl:.3f} (base: {base_results['medical_ppl']:.3f})")

    # Medical behavioral
    promoted_med_behavioral = evaluate_behavioral(model, tok, "medical")
    log(f"  Promoted medical behavioral: {promoted_med_behavioral['score']:.3f} "
        f"(base: {base_results['medical_behavioral']['score']:.3f})")

    # MMLU
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    promoted_mmlu = correct / total if total else 0
    mmlu_deg = (promoted_mmlu - base_results["mmlu"]) * 100
    log(f"  Promoted MMLU: {promoted_mmlu:.1%} ({correct}/{total}, {mmlu_deg:+.1f}pp vs base)")

    # Code PPL (to check non-promoted domains)
    code_texts = load_data("code", "valid", 25)
    promoted_code_ppl = ppl(model, tok, code_texts) if code_texts else float('inf')
    log(f"  Promoted code PPL: {promoted_code_ppl:.3f} (base: {base_results['code_ppl']:.3f})")

    # Math PPL
    math_texts = load_data("math", "valid", 25)
    promoted_math_ppl = ppl(model, tok, math_texts) if math_texts else float('inf')
    log(f"  Promoted math PPL: {promoted_math_ppl:.3f} (base: {base_results['math_ppl']:.3f})")

    med_ppl_ratio = promoted_med_ppl / base_results["medical_ppl"] if base_results["medical_ppl"] > 0 else float('inf')
    code_ppl_ratio = promoted_code_ppl / base_results["code_ppl"] if base_results["code_ppl"] > 0 else float('inf')
    math_ppl_ratio = promoted_math_ppl / base_results["math_ppl"] if base_results["math_ppl"] > 0 else float('inf')

    log(f"  PPL ratios: medical={med_ppl_ratio:.3f}x, code={code_ppl_ratio:.3f}x, math={math_ppl_ratio:.3f}x")

    log_memory("post-promotion-eval")
    cleanup(model, tok, adapter_b)

    return {
        "n_promoted": n_promoted,
        "delta_norms_mean": round(mean_norm, 6),
        "delta_norms_max": round(max_norm, 6),
        "medical_ppl": promoted_med_ppl,
        "medical_ppl_ratio": round(med_ppl_ratio, 4),
        "medical_behavioral": promoted_med_behavioral,
        "mmlu": promoted_mmlu,
        "mmlu_correct": correct,
        "mmlu_total": total,
        "mmlu_degradation_pp": round(mmlu_deg, 2),
        "mmlu_per_subject": per_subject,
        "code_ppl": promoted_code_ppl,
        "code_ppl_ratio": round(code_ppl_ratio, 4),
        "math_ppl": promoted_math_ppl,
        "math_ppl_ratio": round(math_ppl_ratio, 4),
    }


def phase_train_new_adapter_original(domain, domain_idx, skeleton):
    """Phase 3a: Train a new adapter on the ORIGINAL base (control)."""
    log(f"\n" + "=" * 60)
    log(f"Phase 3a: Train {domain} adapter on ORIGINAL base")
    log("=" * 60)
    return _train_adapter(domain, domain_idx, skeleton, promoted=False)


def phase_train_new_adapter_promoted(domain, domain_idx, skeleton):
    """Phase 3b: Train a new adapter on the PROMOTED base."""
    log(f"\n" + "=" * 60)
    log(f"Phase 3b: Train {domain} adapter on PROMOTED base")
    log("=" * 60)
    return _train_adapter(domain, domain_idx, skeleton, promoted=True)


def _train_adapter(domain, domain_idx, skeleton, promoted=False):
    """Train one SFT adapter, optionally on promoted base. Returns metrics."""
    t0 = time.time()

    model, tok = load(MODEL_ID)

    # If promoted: attach the medical adapter as frozen LoRA first
    if promoted:
        med_adapter_b = dict(mx.load(str(ADAPTER_DIR / PROMOTE_DOMAIN / "adapter.npz")))
        med_idx = DOMAINS.index(PROMOTE_DOMAIN)
        n_med = 0
        for li in range(len(model.model.layers)):
            layer = model.model.layers[li]
            updates = []
            for key in TARGET_KEYS:
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{med_idx}"
                if bk not in med_adapter_b or ak not in skeleton:
                    continue
                m = layer
                for part in key.split("."):
                    m = getattr(m, part, None)
                    if m is None:
                        break
                if m is None:
                    continue
                A = mx.array(skeleton[ak]).astype(mx.bfloat16)
                B = med_adapter_b[bk].astype(mx.bfloat16)
                lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=PROMOTE_SCALE, a_init=A)
                lora.lora_b = B
                lora.freeze()
                updates.append((key, lora))
                n_med += 1
            if updates:
                layer.update_modules(tree_unflatten(updates))
        mx.eval(model.parameters())
        log(f"  Attached {n_med} frozen medical LoRA modules (promotion)")
        del med_adapter_b
        gc.collect()

    # Now attach a NEW trainable LoRA for the target domain
    # The new adapter uses DIFFERENT Grassmannian A-matrices (domain_idx != medical_idx)
    n_lora = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey not in skeleton:
                continue

            # Navigate to the current module (may be LoRA-wrapped if promoted)
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            # If already LoRA-wrapped from promotion, wrap the whole thing
            # The new LoRA wraps on top: y = promoted(x) + new_lora(x)
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            new_lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=NEW_ADAPTER_SCALE, a_init=a_mx)
            updates.append((key, new_lora))
            n_lora += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    model.freeze()
    model.unfreeze(keys=["lora_b"], strict=False)

    # Count trainable: should only be the NEW adapter's B-matrices
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  LoRA modules: {n_lora}, trainable params: {trainable_params:,}")

    # Load data
    train_texts = load_data(domain, "train", 400)
    val_texts = load_data(domain, "valid", 50)
    if not train_texts:
        log(f"  SKIP: no training data for {domain}")
        cleanup(model, tok)
        return None

    train_batches = []
    for text in train_texts:
        tokens, mask = tokenize_sft(text, tok, MAX_SEQ)
        if len(tokens) >= 4:
            train_batches.append((tokens, mask))

    val_batches = []
    for text in val_texts:
        tokens, mask = tokenize_sft(text, tok, MAX_SEQ)
        if len(tokens) >= 4:
            val_batches.append((tokens, mask))

    log(f"  Data: {len(train_batches)} train, {len(val_batches)} val")

    # Baseline val loss (zero-init new adapter)
    n_val = min(len(val_batches), 25)
    base_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        base_loss += loss.item()
        del loss
    base_loss /= max(n_val, 1)
    log(f"  Base val loss (zero-init LoRA): {base_loss:.4f}")

    # Train
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, sft_loss_fn)
    initial_loss = None
    losses_log = []
    loss_at_50 = None

    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_batches)
        tokens, mask = train_batches[idx]
        loss, grads = loss_and_grad(model, mx.array([tokens]), mx.array([mask]))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        lv = loss.item()
        if initial_loss is None:
            initial_loss = lv
        if (step + 1) == 50:
            loss_at_50 = lv
        if (step + 1) % 50 == 0:
            losses_log.append({"step": step + 1, "loss": round(lv, 4)})
        if (step + 1) % 100 == 0:
            log(f"  Step {step+1}: loss={lv:.4f}")
    gc.enable()

    # Final val loss
    final_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        final_loss += loss.item()
        del loss
    final_loss /= max(n_val, 1)

    converged = final_loss < base_loss
    pct_change = (base_loss - final_loss) / base_loss * 100 if base_loss > 0 else 0
    log(f"  Val loss: {base_loss:.4f} -> {final_loss:.4f} ({pct_change:+.1f}%, "
        f"{'CONVERGED' if converged else 'FAILED'})")

    train_time = time.time() - t0
    log(f"  Train time: {train_time:.1f}s")

    log_memory(f"post-train-{domain}-{'promoted' if promoted else 'original'}")
    cleanup(model, tok, optimizer)

    return {
        "domain": domain,
        "promoted_base": promoted,
        "base_loss": round(base_loss, 4),
        "final_loss": round(final_loss, 4),
        "initial_train_loss": round(initial_loss, 4) if initial_loss else None,
        "loss_at_50": round(loss_at_50, 4) if loss_at_50 else None,
        "converged": converged,
        "pct_improvement": round(pct_change, 2),
        "train_time_s": round(train_time, 1),
        "n_train": len(train_batches),
        "n_val": n_val,
        "n_lora": n_lora,
        "trainable_params": trainable_params,
        "losses_log": losses_log,
    }


# ── Main orchestrator ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Expert Promotion Experiment")
    log(f"Promote {PROMOTE_DOMAIN} adapter at scale={PROMOTE_SCALE}")
    log(f"Retrain domains: {RETRAIN_DOMAINS}")
    log("=" * 60)
    mx.random.seed(SEED)

    # Load skeleton
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"Skeleton loaded: {len(skeleton)} A-matrices")

    # Phase 1: Base measurements
    base_results = phase_base_measurements(skeleton)

    # Phase 2: Promote and measure
    promotion_results = phase_promote_and_measure(skeleton, base_results)

    # Phase 3: Train new adapters on original and promoted base
    # Train code adapter on original base (control)
    training_results = {}

    for domain in RETRAIN_DOMAINS:
        domain_idx = DOMAINS.index(domain)

        # Original base training
        mx.random.seed(SEED)  # Reset seed for fair comparison
        orig_result = phase_train_new_adapter_original(domain, domain_idx, skeleton)
        training_results[f"{domain}_original"] = orig_result

        # Promoted base training
        mx.random.seed(SEED)  # Reset seed for fair comparison
        prom_result = phase_train_new_adapter_promoted(domain, domain_idx, skeleton)
        training_results[f"{domain}_promoted"] = prom_result

        # Compare
        if orig_result and prom_result:
            speed_ratio = prom_result["train_time_s"] / orig_result["train_time_s"] if orig_result["train_time_s"] > 0 else float('inf')
            loss_ratio = prom_result["final_loss"] / orig_result["final_loss"] if orig_result["final_loss"] > 0 else float('inf')
            log(f"\n  {domain} comparison:")
            log(f"    Train time: {orig_result['train_time_s']:.1f}s (orig) vs {prom_result['train_time_s']:.1f}s (promoted) = {speed_ratio:.2f}x")
            log(f"    Final loss: {orig_result['final_loss']:.4f} (orig) vs {prom_result['final_loss']:.4f} (promoted) = {loss_ratio:.3f}x")
            log(f"    Converged: {orig_result['converged']} (orig) vs {prom_result['converged']} (promoted)")

    # ── Kill criteria assessment ──────────────────────────────────────────

    log("\n" + "=" * 60)
    log("Kill Criteria Assessment")
    log("=" * 60)

    # K839: Promoted expert loses >30% quality
    k839_pass = True
    k839_details = []

    if promotion_results:
        # Medical PPL ratio
        med_ratio = promotion_results["medical_ppl_ratio"]
        k839_details.append(f"medical_ppl_ratio={med_ratio:.3f}")
        if med_ratio > 1.30:
            k839_pass = False
            k839_details.append("FAIL: medical PPL degraded >30%")

        # MMLU degradation
        mmlu_deg = promotion_results["mmlu_degradation_pp"]
        k839_details.append(f"mmlu_degradation={mmlu_deg:+.1f}pp")
        if mmlu_deg < -8:
            k839_pass = False
            k839_details.append("FAIL: MMLU degraded >8pp")

        # Medical behavioral
        base_beh = base_results["medical_behavioral"]["score"]
        prom_beh = promotion_results["medical_behavioral"]["score"]
        beh_ratio = prom_beh / base_beh if base_beh > 0 else 0
        k839_details.append(f"behavioral_ratio={beh_ratio:.3f}")
        if beh_ratio < 0.70:
            k839_pass = False
            k839_details.append("FAIL: behavioral quality <70%")
    else:
        k839_pass = False
        k839_details.append("FAIL: promotion phase failed")

    log(f"  K839: {'PASS' if k839_pass else 'FAIL'} — {'; '.join(k839_details)}")

    # K840: New adapters fail to converge on promoted base
    k840_pass = True
    k840_details = []

    for domain in RETRAIN_DOMAINS:
        orig_key = f"{domain}_original"
        prom_key = f"{domain}_promoted"
        if training_results.get(orig_key) and training_results.get(prom_key):
            orig = training_results[orig_key]
            prom = training_results[prom_key]

            # Check convergence
            if not prom["converged"]:
                k840_pass = False
                k840_details.append(f"FAIL: {domain} did not converge on promoted base")

            # Check quality ratio
            if orig["final_loss"] > 0:
                loss_ratio = prom["final_loss"] / orig["final_loss"]
                k840_details.append(f"{domain}_loss_ratio={loss_ratio:.3f}")
                if loss_ratio > 1.30:
                    k840_pass = False
                    k840_details.append(f"FAIL: {domain} >30% worse on promoted base")

            # Check speed ratio
            if orig["train_time_s"] > 0:
                speed_ratio = prom["train_time_s"] / orig["train_time_s"]
                k840_details.append(f"{domain}_speed_ratio={speed_ratio:.2f}x")
                if speed_ratio > 1.50:
                    k840_details.append(f"WARNING: {domain} >50% slower on promoted base")
        else:
            k840_pass = False
            k840_details.append(f"FAIL: {domain} training failed")

    log(f"  K840: {'PASS' if k840_pass else 'FAIL'} — {'; '.join(k840_details)}")

    all_pass = k839_pass and k840_pass
    log(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ── Save results ──────────────────────────────────────────────────────

    elapsed = time.time() - t0
    results = {
        "config": {
            "model_id": MODEL_ID,
            "promote_domain": PROMOTE_DOMAIN,
            "promote_scale": PROMOTE_SCALE,
            "retrain_domains": RETRAIN_DOMAINS,
            "new_adapter_scale": NEW_ADAPTER_SCALE,
            "lora_rank": LORA_RANK,
            "train_iters": TRAIN_ITERS,
            "lr": LEARNING_RATE,
            "seed": SEED,
        },
        "base": base_results,
        "promotion": promotion_results,
        "training": training_results,
        "kill_criteria": {
            "K839": {"pass": k839_pass, "details": k839_details},
            "K840": {"pass": k840_pass, "details": k840_details},
        },
        "all_pass": all_pass,
        "total_time_s": round(elapsed, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
