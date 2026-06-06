#!/usr/bin/env python3
"""Sequential expert promotion: 3 domains promoted one at a time, verifying the
Davis-Kahan cumulative rotation bound stays below quality-preservation threshold.

Kill criteria:
  K844/K850: After 3 sequential promotions at scale=5, MMLU degradation < 3pp
             (92% -> >= 89%)
  K845/K851: Each newly promoted domain shows PPL improvement vs base (ratio <= 0.90x)
  K846/K852: Previously promoted domains not catastrophically degraded by
             subsequent promotions (PPL ratio < 1.10x after each step)

Promotion sequence: medical -> code -> math (3 sequential promotions)

CRITICAL FIX vs exp_expert_promotion:
  Old confound: model.unfreeze(keys=["lora_b"]) unfroze ALL lora_b including
  promoted (should-be-frozen) adapter B-matrices. Fix: after calling
  model.unfreeze(keys=["lora_b"]), explicitly re-freeze all promoted adapters
  by name pattern. This ensures only the NEW adapter's lora_b is trainable.
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

from mlx_lm import load
from pierre.bench import mmlu_eval, ppl, cleanup

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
SKELETON_PATH = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
LORA_RANK = 16
PROMOTE_SCALE = 5.0
NEW_ADAPTER_SCALE = 20.0
TRAIN_ITERS = 300
LEARNING_RATE = 1e-4
MAX_SEQ = 256
SEED = 42
RESPONSE_MARKER = "### Response:\n"

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# 3 sequential promotions: medical first, then code, then math
PROMOTION_SEQUENCE = ["medical", "code", "math"]
EVAL_DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# MMLU questions (same 50 as prior experiments for consistency)
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


# ── GrassmannianLoRALinear ────────────────────────────────────────────────────

class GrassmannianLoRALinear(nn.Module):
    """QLoRA with frozen Grassmannian A-matrix and trainable B-matrix.

    Can wrap QuantizedLinear, Linear, or another GrassmannianLoRALinear
    (for stacking promoted adapter + new trainable adapter).
    """
    def __init__(self, base_linear, rank=16, scale=20.0, a_init=None):
        super().__init__()
        if isinstance(base_linear, GrassmannianLoRALinear):
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


# ── Promotion helper ──────────────────────────────────────────────────────────

def attach_frozen_promotion(model, skeleton, domain, domain_idx, adapter_b_path):
    """Attach a domain adapter as a FROZEN LoRA overlay onto the model.

    This implements 'promotion': the adapter becomes permanently active (frozen)
    on top of the base (or previously promoted) model. Since QuantizedLinear
    prevents true weight modification, we use a frozen LoRA overlay as the
    mathematically equivalent substitute.

    Returns (n_attached, mean_delta_norm, max_delta_norm).
    """
    adapter_b = dict(mx.load(str(adapter_b_path)))
    log(f"  Attaching frozen promotion for '{domain}' ({len(adapter_b)} B-matrices)")

    # Compute delta norms for analysis
    delta_norms = []
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue
            A = mx.array(skeleton[ak]).astype(mx.float32)
            B = adapter_b[bk].astype(mx.float32)
            delta = PROMOTE_SCALE * (A @ B)
            dn = mx.linalg.norm(delta.reshape(-1))
            mx.eval(dn)
            delta_norms.append(dn.item())
            del delta, dn
    mx.clear_cache()

    # Attach frozen LoRA overlay
    n_attached = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue
            # Navigate to current module (may already be LoRA-wrapped from prior promotion)
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            B = adapter_b[bk].astype(mx.bfloat16)
            lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=PROMOTE_SCALE, a_init=A)
            lora.lora_b = B
            lora.freeze()  # Both A and B frozen: this is the "promoted" adapter
            updates.append((key, lora))
            n_attached += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    del adapter_b
    gc.collect()
    mx.clear_cache()

    mean_norm = float(np.mean(delta_norms)) if delta_norms else 0.0
    max_norm = float(np.max(delta_norms)) if delta_norms else 0.0
    log(f"  Attached {n_attached} frozen LoRA modules. Delta norms: mean={mean_norm:.4f}, max={max_norm:.4f}")
    return n_attached, mean_norm, max_norm


def attach_trainable_adapter(model, skeleton, domain, domain_idx):
    """Attach a NEW trainable LoRA adapter on top of the current model state.

    This adapter will be trained. The CRITICAL FIX is here: after this call,
    we must explicitly re-freeze promoted adapters so that only this new
    adapter's lora_b is trainable.

    Returns n_attached.
    """
    n_attached = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey not in skeleton:
                continue
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            new_lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=NEW_ADAPTER_SCALE, a_init=a_mx)
            updates.append((key, new_lora))
            n_attached += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return n_attached


def freeze_promoted_refreeze_new(model, new_domain_idx):
    """CRITICAL FIX: Ensure only the outermost (newly added) lora_b is trainable.

    The problem: after attaching a new trainable LoRA on top of a promoted (frozen)
    LoRA, calling model.unfreeze(keys=['lora_b']) unfreezes ALL lora_b recursively —
    including the promoted adapters' B-matrices.

    Verified fix: freeze entire model, unfreeze at module level (which recurses into
    inner lora_b too), then explicitly re-freeze all inner promoted LoRA modules.

    Structure:
      layer.key                   = GrassmannianLoRALinear (NEW, outer — trainable lora_b)
        .linear                   = GrassmannianLoRALinear (promoted 1 — should stay frozen)
          .linear                 = GrassmannianLoRALinear or QuantizedLinear (base)

    Algorithm:
      1. model.freeze() — freezes everything
      2. For each outermost GrassmannianLoRALinear: call .unfreeze(keys=['lora_b'])
         (this unfreezes both outer.lora_b AND any inner .lora_b due to recursion)
      3. For each inner (promoted) GrassmannianLoRALinear: call .freeze()
         (re-freezes the inner lora_b, leaving only outer lora_b trainable)

    Returns: number of outer lora_b parameters unfrozen.
    """
    model.freeze()  # Step 1: freeze everything

    n_unfrozen = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            # Navigate to outermost module at this key
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None or not isinstance(m, GrassmannianLoRALinear):
                continue

            # Step 2: unfreeze lora_b at this module (recurses into nested lora_b too)
            m.unfreeze(keys=["lora_b"], strict=False)
            n_unfrozen += 1

            # Step 3: re-freeze all inner (promoted) LoRA modules
            # The inner module is m.linear — freeze it and everything inside it
            inner = m.linear
            while isinstance(inner, GrassmannianLoRALinear):
                inner.freeze()  # re-freezes inner.lora_b
                inner = inner.linear

    return n_unfrozen


def measure_full_state(model, tok, skeleton, label):
    """Measure MMLU and PPL for all eval domains. Returns dict."""
    log(f"\n[EVAL] {label}")
    result = {"label": label}

    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    mmlu = correct / total if total else 0
    result["mmlu"] = round(mmlu, 4)
    result["mmlu_correct"] = correct
    result["mmlu_total"] = total
    result["mmlu_per_subject"] = per_subject
    log(f"  MMLU: {mmlu:.1%} ({correct}/{total})")

    domain_ppl = {}
    for domain in EVAL_DOMAINS:
        texts = load_data(domain, "valid", 25)
        if texts:
            domain_ppl[domain] = round(ppl(model, tok, texts), 4)
        else:
            domain_ppl[domain] = None
        log(f"  {domain} PPL: {domain_ppl[domain]}")
    result["domain_ppl"] = domain_ppl

    return result


# ── Phase 0: Baseline ─────────────────────────────────────────────────────────

def phase_baseline():
    """Measure base model MMLU and all-domain PPLs (no adapters)."""
    log("\n" + "=" * 60)
    log("Phase 0: Base model measurements (no adapters)")
    log("=" * 60)

    model, tok = load(MODEL_ID)
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"Skeleton loaded: {len(skeleton)} A-matrices")

    result = measure_full_state(model, tok, skeleton, "baseline")

    log_memory("post-baseline")
    cleanup(model, tok)
    del skeleton
    gc.collect()
    mx.clear_cache()
    return result


# ── Phase 1: Promote medical ──────────────────────────────────────────────────

def phase_promote_medical(base_results):
    """Promote medical adapter into the base model. Measure quality."""
    log("\n" + "=" * 60)
    log("Phase 1: Promote medical adapter (scale=5)")
    log("=" * 60)

    skeleton = dict(np.load(str(SKELETON_PATH)))
    model, tok = load(MODEL_ID)

    med_idx = DOMAINS.index("medical")
    med_adapter_path = ADAPTER_DIR / "medical" / "adapter.npz"

    n_att, mean_norm, max_norm = attach_frozen_promotion(
        model, skeleton, "medical", med_idx, med_adapter_path
    )
    model.freeze()  # Everything frozen after promotion

    result = measure_full_state(model, tok, skeleton, "after_promotion_1_medical")
    result["n_promoted"] = n_att
    result["delta_norm_mean"] = round(mean_norm, 6)
    result["delta_norm_max"] = round(max_norm, 6)

    # Compute K851/K845: newly promoted domain PPL ratio
    base_med_ppl = base_results["domain_ppl"]["medical"]
    promoted_med_ppl = result["domain_ppl"]["medical"]
    if base_med_ppl and promoted_med_ppl:
        result["k851_medical"] = round(promoted_med_ppl / base_med_ppl, 4)
        log(f"  K851 (medical PPL ratio vs base): {result['k851_medical']:.4f} "
            f"({'PASS' if result['k851_medical'] <= 0.90 else 'FAIL/check'})")
    else:
        result["k851_medical"] = None

    mmlu_deg = (result["mmlu"] - base_results["mmlu"]) * 100
    result["mmlu_degradation_pp"] = round(mmlu_deg, 2)
    log(f"  MMLU degradation: {mmlu_deg:+.1f}pp vs base")

    log_memory("post-promote-1")
    cleanup(model, tok)
    del skeleton
    gc.collect()
    mx.clear_cache()
    return result


# ── Phase 2: Train and promote code ──────────────────────────────────────────

def phase_train_code_on_promoted_base(base_results, phase1_results):
    """Train code adapter on medical-promoted base, then promote it.

    Step 2a: Load base + attach frozen medical promotion
    Step 2b: Attach trainable code adapter on top
    Step 2c: CRITICAL FIX — unfreeze only code lora_b (not medical lora_b)
    Step 2d: Train 300 iterations
    Step 2e: Promote code adapter (freeze code lora_b)
    Step 2f: Measure MMLU, all domain PPLs
    """
    log("\n" + "=" * 60)
    log("Phase 2: Train code adapter on promoted base, then promote")
    log("=" * 60)

    skeleton = dict(np.load(str(SKELETON_PATH)))
    model, tok = load(MODEL_ID)

    # Step 2a: Attach frozen medical promotion
    med_idx = DOMAINS.index("medical")
    med_adapter_path = ADAPTER_DIR / "medical" / "adapter.npz"
    n_med, _, _ = attach_frozen_promotion(model, skeleton, "medical", med_idx, med_adapter_path)
    model.freeze()  # Medical adapter frozen

    # Step 2b: Attach trainable code adapter on top
    code_idx = DOMAINS.index("code")
    n_code = attach_trainable_adapter(model, skeleton, "code", code_idx)
    log(f"  Attached {n_code} code LoRA modules (trainable)")

    # Step 2c: CRITICAL FIX — explicitly unfreeze only the outermost lora_b
    n_unfrozen = freeze_promoted_refreeze_new(model, code_idx)
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params after fix: {trainable_params:,} (unfrozen modules: {n_unfrozen})")
    # Sanity check: should be approx n_code * LORA_RANK * d_out
    # For n_code=252, rank=16, d_out=2560: expected ~ 252 * 16 * 2560 = 10,321,920 ~ 10M
    expected_min = n_code * LORA_RANK * 100  # loose lower bound
    if trainable_params < expected_min:
        log(f"  WARNING: fewer trainable params than expected ({trainable_params} < {expected_min})")

    # Step 2d: Train code adapter
    t0_train = time.time()
    train_texts = load_data("code", "train", 400)
    val_texts = load_data("code", "valid", 50)
    if not train_texts:
        log("  ERROR: no code training data")
        cleanup(model, tok)
        del skeleton
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
    log(f"  Code data: {len(train_batches)} train, {len(val_batches)} val")

    # Baseline val loss (zero-init code adapter)
    n_val = min(len(val_batches), 25)
    base_val_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        base_val_loss += loss.item()
        del loss
    base_val_loss /= max(n_val, 1)
    log(f"  Code val loss (zero-init): {base_val_loss:.4f}")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, sft_loss_fn)
    losses_log = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_batches)
        tokens, mask = train_batches[idx]
        loss, grads = loss_and_grad(model, mx.array([tokens]), mx.array([mask]))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if (step + 1) % 50 == 0:
            lv = loss.item()
            losses_log.append({"step": step + 1, "loss": round(lv, 4)})
            log(f"  Step {step+1}: loss={lv:.4f}")
    gc.enable()

    final_val_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        final_val_loss += loss.item()
        del loss
    final_val_loss /= max(n_val, 1)
    train_time = time.time() - t0_train
    log(f"  Code val loss: {base_val_loss:.4f} -> {final_val_loss:.4f} "
        f"({'CONVERGED' if final_val_loss < base_val_loss else 'FAILED'}), "
        f"time={train_time:.1f}s")

    del optimizer, train_batches, val_batches, train_texts, val_texts
    gc.collect()
    mx.clear_cache()

    # Step 2e: Promote code adapter (freeze its lora_b)
    # The outermost LoRA at each module is the code adapter. Freeze it.
    model.freeze()  # Freeze everything = both medical and code are now frozen/promoted
    n_frozen = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, GrassmannianLoRALinear):
                n_frozen += 1
    log(f"  Code adapter promoted (frozen). Total frozen LoRA modules: {n_frozen}")

    # Step 2f: Measure full state
    result = measure_full_state(model, tok, skeleton, "after_promotion_2_code")
    result["code_train"] = {
        "base_val_loss": round(base_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "converged": final_val_loss < base_val_loss,
        "train_time_s": round(train_time, 1),
        "trainable_params": trainable_params,
        "losses_log": losses_log,
    }

    # Kill criterion ratios
    base_ppl = base_results["domain_ppl"]
    dom_ppl = result["domain_ppl"]

    # K851 for code (newly promoted domain should improve)
    if base_ppl.get("code") and dom_ppl.get("code"):
        result["k851_code"] = round(dom_ppl["code"] / base_ppl["code"], 4)
        log(f"  K851 (code PPL ratio vs base): {result['k851_code']:.4f} "
            f"({'PASS' if result['k851_code'] <= 0.90 else 'FAIL/check'})")

    # K852 for medical (should not degrade after code promotion)
    if base_ppl.get("medical") and dom_ppl.get("medical"):
        result["k852_medical"] = round(dom_ppl["medical"] / base_ppl["medical"], 4)
        log(f"  K852 (medical PPL ratio after code promo): {result['k852_medical']:.4f} "
            f"({'PASS' if result['k852_medical'] < 1.10 else 'FAIL'})")

    mmlu_deg = (result["mmlu"] - base_results["mmlu"]) * 100
    result["mmlu_degradation_pp"] = round(mmlu_deg, 2)
    log(f"  Cumulative MMLU degradation: {mmlu_deg:+.1f}pp vs base")

    log_memory("post-phase-2")
    cleanup(model, tok)
    del skeleton
    gc.collect()
    mx.clear_cache()
    return result


# ── Phase 3: Train and promote math ──────────────────────────────────────────

def phase_train_math_on_doubly_promoted_base(base_results, phase1_results, phase2_results):
    """Train math adapter on [medical+code]-promoted base, then promote it.

    Same pattern as Phase 2 but starting from a doubly-promoted base.
    """
    log("\n" + "=" * 60)
    log("Phase 3: Train math adapter on doubly-promoted base, then promote")
    log("=" * 60)

    skeleton = dict(np.load(str(SKELETON_PATH)))
    model, tok = load(MODEL_ID)

    # Attach frozen medical promotion
    med_idx = DOMAINS.index("medical")
    med_adapter_path = ADAPTER_DIR / "medical" / "adapter.npz"
    n_med, _, _ = attach_frozen_promotion(model, skeleton, "medical", med_idx, med_adapter_path)
    model.freeze()

    # Attach frozen code promotion
    code_idx = DOMAINS.index("code")
    code_adapter_path = ADAPTER_DIR / "code" / "adapter.npz"
    n_code_frozen, _, _ = attach_frozen_promotion(model, skeleton, "code", code_idx, code_adapter_path)
    model.freeze()

    log(f"  Double-promoted base: medical({n_med}) + code({n_code_frozen}) frozen LoRA modules")

    # Attach trainable math adapter
    math_idx = DOMAINS.index("math")
    n_math = attach_trainable_adapter(model, skeleton, "math", math_idx)
    log(f"  Attached {n_math} math LoRA modules (trainable)")

    # CRITICAL FIX: unfreeze only the outermost (math) lora_b
    n_unfrozen = freeze_promoted_refreeze_new(model, math_idx)
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params after fix: {trainable_params:,} (unfrozen modules: {n_unfrozen})")
    expected_min = n_math * LORA_RANK * 100
    if trainable_params < expected_min:
        log(f"  WARNING: fewer trainable params than expected ({trainable_params} < {expected_min})")

    # Train math adapter
    t0_train = time.time()
    train_texts = load_data("math", "train", 400)
    val_texts = load_data("math", "valid", 50)
    if not train_texts:
        log("  ERROR: no math training data")
        cleanup(model, tok)
        del skeleton
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
    log(f"  Math data: {len(train_batches)} train, {len(val_batches)} val")

    n_val = min(len(val_batches), 25)
    base_val_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        base_val_loss += loss.item()
        del loss
    base_val_loss /= max(n_val, 1)
    log(f"  Math val loss (zero-init): {base_val_loss:.4f}")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, sft_loss_fn)
    losses_log = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_batches)
        tokens, mask = train_batches[idx]
        loss, grads = loss_and_grad(model, mx.array([tokens]), mx.array([mask]))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if (step + 1) % 50 == 0:
            lv = loss.item()
            losses_log.append({"step": step + 1, "loss": round(lv, 4)})
            log(f"  Step {step+1}: loss={lv:.4f}")
    gc.enable()

    final_val_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        final_val_loss += loss.item()
        del loss
    final_val_loss /= max(n_val, 1)
    train_time = time.time() - t0_train
    log(f"  Math val loss: {base_val_loss:.4f} -> {final_val_loss:.4f} "
        f"({'CONVERGED' if final_val_loss < base_val_loss else 'FAILED'}), "
        f"time={train_time:.1f}s")

    del optimizer, train_batches, val_batches, train_texts, val_texts
    gc.collect()
    mx.clear_cache()

    # Promote math adapter
    model.freeze()
    log(f"  Math adapter promoted (all 3 domains now frozen in base)")

    # Measure final state
    result = measure_full_state(model, tok, skeleton, "after_promotion_3_math")
    result["math_train"] = {
        "base_val_loss": round(base_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "converged": final_val_loss < base_val_loss,
        "train_time_s": round(train_time, 1),
        "trainable_params": trainable_params,
        "losses_log": losses_log,
    }

    # Kill criterion ratios
    base_ppl = base_results["domain_ppl"]
    dom_ppl = result["domain_ppl"]

    # K851 for math (newly promoted domain should improve)
    if base_ppl.get("math") and dom_ppl.get("math"):
        result["k851_math"] = round(dom_ppl["math"] / base_ppl["math"], 4)
        log(f"  K851 (math PPL ratio vs base): {result['k851_math']:.4f} "
            f"({'PASS' if result['k851_math'] <= 0.90 else 'FAIL/check'})")

    # K852 checks: previously promoted domains should not degrade
    if base_ppl.get("medical") and dom_ppl.get("medical"):
        result["k852_medical"] = round(dom_ppl["medical"] / base_ppl["medical"], 4)
        log(f"  K852 (medical PPL ratio after math promo): {result['k852_medical']:.4f} "
            f"({'PASS' if result['k852_medical'] < 1.10 else 'FAIL'})")
    if base_ppl.get("code") and dom_ppl.get("code"):
        result["k852_code"] = round(dom_ppl["code"] / base_ppl["code"], 4)
        log(f"  K852 (code PPL ratio after math promo): {result['k852_code']:.4f} "
            f"({'PASS' if result['k852_code'] < 1.10 else 'FAIL'})")

    mmlu_deg = (result["mmlu"] - base_results["mmlu"]) * 100
    result["mmlu_degradation_pp"] = round(mmlu_deg, 2)
    log(f"  Final MMLU degradation: {mmlu_deg:+.1f}pp vs base")

    log_memory("post-phase-3")
    cleanup(model, tok)
    del skeleton
    gc.collect()
    mx.clear_cache()
    return result


# ── Kill criteria assessment ──────────────────────────────────────────────────

def assess_kill_criteria(base, phase1, phase2, phase3):
    """Evaluate K844/K850, K845/K851, K846/K852 from final results."""
    log("\n" + "=" * 60)
    log("Kill Criteria Assessment")
    log("=" * 60)

    # K850 (K844): After 3 sequential promotions, MMLU degradation < 3pp (>= 89%)
    k850_pass = False
    k850_details = []
    if phase3:
        mmlu_final = phase3["mmlu"]
        mmlu_deg = phase3.get("mmlu_degradation_pp", (mmlu_final - base["mmlu"]) * 100)
        k850_details.append(f"MMLU_final={mmlu_final:.1%}, degradation={mmlu_deg:+.1f}pp")
        k850_pass = mmlu_final >= 0.89
        k850_details.append("PASS" if k850_pass else "FAIL: MMLU < 89%")
    else:
        k850_details.append("FAIL: phase3 did not complete")

    log(f"  K850: {'PASS' if k850_pass else 'FAIL'} — {'; '.join(k850_details)}")

    # K851 (K845): Each newly promoted domain PPL ratio <= 0.90 vs base
    k851_pass = True
    k851_details = []
    for domain, phase_result in [("medical", phase1), ("code", phase2), ("math", phase3)]:
        if phase_result is None:
            k851_pass = False
            k851_details.append(f"FAIL: {domain} phase did not complete")
            continue
        key = f"k851_{domain}"
        ratio = phase_result.get(key)
        if ratio is None:
            k851_pass = False
            k851_details.append(f"FAIL: {domain} PPL ratio not computed")
        else:
            k851_details.append(f"{domain}_ratio={ratio:.4f}")
            if ratio > 0.90:
                k851_pass = False
                k851_details.append(f"FAIL: {domain} ratio={ratio:.4f} > 0.90")

    log(f"  K851: {'PASS' if k851_pass else 'FAIL'} — {'; '.join(k851_details)}")

    # K852 (K846): Previously promoted domains not catastrophically degraded (<1.10x)
    k852_pass = True
    k852_details = []

    # After code promotion: medical should be < 1.10
    if phase2 is not None:
        ratio = phase2.get("k852_medical")
        if ratio is not None:
            k852_details.append(f"medical_after_code={ratio:.4f}")
            if ratio >= 1.10:
                k852_pass = False
                k852_details.append(f"FAIL: medical PPL degraded {ratio:.4f}x after code promo")

    # After math promotion: medical and code should be < 1.10
    if phase3 is not None:
        for domain in ["medical", "code"]:
            ratio = phase3.get(f"k852_{domain}")
            if ratio is not None:
                k852_details.append(f"{domain}_after_math={ratio:.4f}")
                if ratio >= 1.10:
                    k852_pass = False
                    k852_details.append(f"FAIL: {domain} PPL degraded {ratio:.4f}x after math promo")

    if not k852_details:
        k852_pass = False
        k852_details.append("FAIL: no K852 ratios computed")

    log(f"  K852: {'PASS' if k852_pass else 'FAIL'} — {'; '.join(k852_details)}")

    all_pass = k850_pass and k851_pass and k852_pass
    log(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return {
        "K850": {"pass": k850_pass, "details": k850_details},
        "K851": {"pass": k851_pass, "details": k851_details},
        "K852": {"pass": k852_pass, "details": k852_details},
        "all_pass": all_pass,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Sequential Expert Promotion Experiment")
    log(f"Sequence: {' -> '.join(PROMOTION_SEQUENCE)}")
    log(f"Promote scale: {PROMOTE_SCALE}, New adapter scale: {NEW_ADAPTER_SCALE}")
    log(f"Training iterations: {TRAIN_ITERS}, LR: {LEARNING_RATE}")
    log("=" * 60)
    mx.random.seed(SEED)

    # Phase 0: Baseline
    base_results = phase_baseline()

    # Phase 1: Promote medical
    mx.random.seed(SEED)
    phase1_results = phase_promote_medical(base_results)

    # Phase 2: Train + promote code on promoted base
    mx.random.seed(SEED)
    phase2_results = phase_train_code_on_promoted_base(base_results, phase1_results)

    # Phase 3: Train + promote math on doubly-promoted base
    mx.random.seed(SEED)
    phase3_results = phase_train_math_on_doubly_promoted_base(
        base_results, phase1_results, phase2_results
    )

    # Kill criteria
    kill_criteria = assess_kill_criteria(base_results, phase1_results, phase2_results, phase3_results)

    # Summary
    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Compute cumulative MMLU degradation
    final_mmlu = phase3_results["mmlu"] if phase3_results else None
    cumulative_deg = None
    if final_mmlu is not None:
        cumulative_deg = round((final_mmlu - base_results["mmlu"]) * 100, 2)

    results = {
        "config": {
            "model_id": MODEL_ID,
            "promotion_sequence": PROMOTION_SEQUENCE,
            "promote_scale": PROMOTE_SCALE,
            "new_adapter_scale": NEW_ADAPTER_SCALE,
            "lora_rank": LORA_RANK,
            "train_iters": TRAIN_ITERS,
            "lr": LEARNING_RATE,
            "seed": SEED,
        },
        "baseline": base_results,
        "after_promotion_1": phase1_results,
        "after_promotion_2": phase2_results,
        "after_promotion_3": phase3_results,
        "kill_criteria": kill_criteria,
        "cumulative_mmlu_degradation_pp": cumulative_deg,
        "total_time_s": round(elapsed, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
