#!/usr/bin/env python3
"""
C1.1: PoLAR with Joint Stiefel on Gemma 4 E4B (5-Fix Re-test of T1.5)

5 fixes over killed T1.5:
  1. Both U+V Stiefel (not just U)
  2. 1000 steps (not 200)
  3. Multi-domain synthetic data (not single-domain GSM8K)
  4. Gemma 4 E4B (not Qwen3-4B proxy)
  5. Rank sweep: r=16 (KC07) and r=6 (KC08/KC09)

Kill criteria:
  KC07: sr(PoLAR ΔW) >= 5 at r=16 with multi-domain training
  KC08: PoLAR GSM8K accuracy >= LoRA at matched rank r=6 and 1000 steps
  KC09: Both ||UU^T-I||_F < 0.01 AND ||VV^T-I||_F < 0.01 post-retraction

SMOKE_TEST=1: 10 steps per phase, 5 eval examples.
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_LAYERS = 42
D_IN = 2560       # Gemma 4 E4B hidden_size (q_proj input)
D_OUT = 2048      # Gemma 4 E4B q_proj output (local attention)

SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 256
RETRACT_EVERY = 20    # Steps between Stiefel retractions
GRAD_CLIP = 1.0

# Per-phase steps
STEPS_R16 = 10 if IS_SMOKE else 500    # Phase 1: r=16 KC07
STEPS_R6 = 10 if IS_SMOKE else 1000   # Phase 2+3: r=6 KC08/KC09

N_EVAL_GSM8K = 5 if IS_SMOKE else 30  # KC08 evaluation


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


# ──────────────────────────────────────────────────────────────
# Synthetic multi-domain dataset (no downloads required)
# ──────────────────────────────────────────────────────────────

MATH_EXAMPLES = [
    ("What is 7 × 8?", "56"),
    ("What is 144 / 12?", "12"),
    ("What is 15² ?", "225"),
    ("Solve for x: 3x + 6 = 21", "5"),
    ("What is the area of a rectangle 8 × 5?", "40"),
    ("What is √81?", "9"),
    ("Simplify: 2(3 + 4) - 5", "9"),
    ("What is 2³ + 3²?", "17"),
    ("If a = 4 and b = 7, what is a × b?", "28"),
    ("What is 100 - 37?", "63"),
    ("What is 0.5 × 0.5?", "0.25"),
    ("What is 1000 / 4?", "250"),
    ("What is the LCM of 4 and 6?", "12"),
    ("What is 5! (5 factorial)?", "120"),
    ("What is 2^10?", "1024"),
    ("A train travels 60 km/h for 3 hours. Distance?", "180 km"),
    ("What is 7 + 8 × 2?", "23"),
    ("What is 4/5 + 1/5?", "1"),
    ("What is 3/4 × 8?", "6"),
    ("Solve: 2x - 4 = 10", "7"),
]

CODE_EXAMPLES = [
    ("What does `len([1, 2, 3])` return?", "3"),
    ("What does `'hello'.upper()` return?", "'HELLO'"),
    ("What does `range(5)` produce?", "0, 1, 2, 3, 4"),
    ("What does `[x**2 for x in range(3)]` evaluate to?", "[0, 1, 4]"),
    ("What does `sum([1, 2, 3, 4])` return?", "10"),
    ("What does `'abc'.replace('b', 'x')` return?", "'axc'"),
    ("What does `max([3, 1, 4, 1, 5])` return?", "5"),
    ("What does `sorted([3, 1, 2])` return?", "[1, 2, 3]"),
    ("What does `'abc'[1]` return?", "'b'"),
    ("What does `{'a': 1}.get('b', 0)` return?", "0"),
    ("What does `int('42')` return?", "42"),
    ("What does `list(reversed([1, 2, 3]))` return?", "[3, 2, 1]"),
    ("What does `'hello world'.split()` return?", "['hello', 'world']"),
    ("What does `abs(-7)` return?", "7"),
    ("What does `round(3.7)` return?", "4"),
    ("What does `isinstance(3, int)` return?", "True"),
    ("What is the output of `print('hi')[:2]`?", "SyntaxError — print returns None"),
    ("What does `set([1, 2, 2, 3])` return?", "{1, 2, 3}"),
    ("What does `tuple([1, 2, 3])` return?", "(1, 2, 3)"),
    ("What does `'abc' * 2` return?", "'abcabc'"),
]

LANGUAGE_EXAMPLES = [
    ("What is the past tense of 'run'?", "ran"),
    ("What is the plural of 'mouse'?", "mice"),
    ("What is the synonym of 'happy'?", "joyful"),
    ("What is the antonym of 'hot'?", "cold"),
    ("What part of speech is 'quickly'?", "adverb"),
    ("What is a noun? Give an example.", "A person, place, thing: e.g., 'dog'"),
    ("What is the comparative form of 'good'?", "better"),
    ("What is the passive voice of 'I ate the cake'?", "The cake was eaten by me"),
    ("What is an idiom? Give an example.", "Fixed phrase: 'kick the bucket' means die"),
    ("What is the possessive of 'James'?", "James's"),
    ("What does 'ubiquitous' mean?", "present or found everywhere"),
    ("What is a conjunction? Give an example.", "Word joining clauses: 'and', 'but'"),
    ("What is the plural of 'criterion'?", "criteria"),
    ("What is the meaning of 'ephemeral'?", "lasting for a very short time"),
    ("What part of speech is 'beautiful'?", "adjective"),
    ("Translate to French: 'Hello, how are you?'", "Bonjour, comment allez-vous?"),
    ("What is a metaphor? Give an example.", "'Time is money' — direct comparison"),
    ("What is the superlative of 'bad'?", "worst"),
    ("What is an oxymoron? Give an example.", "Contradiction: 'deafening silence'"),
    ("What is the meaning of 'ambiguous'?", "open to more than one interpretation"),
]

LOGIC_EXAMPLES = [
    ("If all A are B and all B are C, are all A C?", "Yes, by transitivity"),
    ("If P implies Q and P is true, what follows?", "Q is true (modus ponens)"),
    ("What is the contrapositive of 'If P then Q'?", "If not Q then not P"),
    ("If no A are B, can any A be B?", "No, by universal negation"),
    ("If some X are Y and all Y are Z, what do we know?", "Some X are Z"),
    ("Is 'All cats are animals' a universal or existential statement?", "Universal"),
    ("What is a syllogism?", "Deductive reasoning with two premises and a conclusion"),
    ("If A = {1,2,3} and B = {2,3,4}, what is A ∩ B?", "{2, 3}"),
    ("If A = {1,2,3} and B = {2,3,4}, what is A ∪ B?", "{1, 2, 3, 4}"),
    ("What is the negation of 'P and Q'?", "not P or not Q (De Morgan)"),
    ("If today is Monday, what day was 3 days ago?", "Friday"),
    ("If x > 5 and x < 10, can x = 7?", "Yes, 7 is between 5 and 10"),
    ("What is deductive reasoning?", "Drawing specific conclusions from general premises"),
    ("What is inductive reasoning?", "Drawing general conclusions from specific examples"),
    ("If A ⊂ B and B ⊂ C, is A ⊂ C?", "Yes, by transitivity of subsets"),
    ("What is the truth table value of T AND F?", "False"),
    ("What is the truth table value of T OR F?", "True"),
    ("What is the truth table value of NOT T?", "False"),
    ("What is a tautology?", "A statement that is always true, e.g., P or not P"),
    ("What is a contradiction?", "A statement that is always false, e.g., P and not P"),
]

SCIENCE_EXAMPLES = [
    ("What is H₂O?", "Water"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the speed of light approximately?", "3 × 10⁸ m/s"),
    ("What is Newton's first law?", "An object in motion stays in motion unless acted on by a force"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide (CO₂)"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What is DNA?", "Deoxyribonucleic acid — carries genetic information"),
    ("What is photosynthesis?", "Plants convert sunlight + CO₂ + H₂O into glucose + O₂"),
    ("What is the periodic table?", "Table organizing elements by atomic number and properties"),
    ("What is gravity?", "Force of attraction between masses; ~9.8 m/s² on Earth surface"),
    ("What is the boiling point of water at sea level?", "100°C / 212°F"),
    ("What is an atom?", "Smallest unit of an element; protons, neutrons, electrons"),
    ("What is osmosis?", "Movement of water through a semi-permeable membrane"),
    ("What is evolution?", "Change in heritable traits of populations over generations"),
    ("What is E=mc²?", "Mass-energy equivalence: energy equals mass times speed of light squared"),
    ("What organ pumps blood?", "The heart"),
    ("What is the ozone layer?", "Stratospheric layer of O₃ that absorbs UV radiation"),
    ("What is a molecule?", "Two or more atoms bonded together, e.g., H₂O"),
]

ALL_DOMAIN_DATA = {
    "math": MATH_EXAMPLES,
    "code": CODE_EXAMPLES,
    "language": LANGUAGE_EXAMPLES,
    "logic": LOGIC_EXAMPLES,
    "science": SCIENCE_EXAMPLES,
}
DOMAIN_NAMES = list(ALL_DOMAIN_DATA.keys())


def build_training_corpus(tokenizer, n_per_domain: int = 20) -> list:
    """Build tokenized training samples from all 5 domains."""
    samples = []
    for domain, examples in ALL_DOMAIN_DATA.items():
        chosen = examples[:n_per_domain]
        for q, a in chosen:
            text = f"Q: {q}\nA: {a}"
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN]
            samples.append(tokens)
    random.seed(SEED)
    random.shuffle(samples)
    return samples


def get_batch_random(samples: list, batch_size: int, step: int) -> mx.array:
    """Get a random batch for this step."""
    rng = np.random.default_rng(SEED + step)
    indices = rng.choice(len(samples), size=batch_size, replace=True)
    batch_tokens = [samples[i] for i in indices]
    max_len = max(len(t) for t in batch_tokens)
    padded = [t + [0] * (max_len - len(t)) for t in batch_tokens]
    return mx.array(padded, dtype=mx.uint32)


# ──────────────────────────────────────────────────────────────
# PoLAR Linear module
# ──────────────────────────────────────────────────────────────

class PoLARLinear(nn.Module):
    """PoLAR adapter: ΔW = A @ B where A^T A = I_r and B B^T = I_r.

    A ∈ R^{d_in × r}: initialized with QR orthonormal columns → A^T A = I_r
    B ∈ R^{r × d_out}: initialized to zeros

    Forward: x @ A @ B + x @ base  (same flop count as LoRA)
    Retraction (every RETRACT_EVERY steps): polar project A and B to Stiefel
    """

    def __init__(self, base_linear: nn.Module, rank: int, scale: float):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        # Get dimensions from base layer
        if hasattr(base_linear, 'group_size'):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]

        self.d_in = d_in
        self.d_out = d_out

        # Init A with orthonormal columns via QR
        rng = np.random.default_rng(SEED)
        rand_mat = rng.standard_normal((d_in, rank)).astype(np.float32)
        Q, _ = np.linalg.qr(rand_mat)  # Q: (d_in, rank), orthonormal columns
        self.lora_a = mx.array(Q)  # (d_in, r): orthonormal columns

        # Init B to zeros (residual property at t=0)
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b  # (...,d_in)→(...,r)→(...,d_out)
        return base_out + self.scale * lora_out

    def retract_to_stiefel(self) -> tuple[float, float]:
        """Polar-project A and B to Stiefel. Returns (stiefel_dist_A, stiefel_dist_B)."""
        I_r = np.eye(self.rank)

        # Retract A: (d_in, r) → columns orthonormal → A^T A = I_r
        A_np = np.array(self.lora_a.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
            dist_A = float(np.sqrt(np.sum((A_np.T @ A_np - I_r) ** 2))) if np.all(np.isfinite(A_np)) else float('inf')
        else:
            W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
            A_retracted = W @ Vh  # (d_in, r) with A^T A = I_r
            self.lora_a = mx.array(A_retracted.astype(np.float32))
            AtA = A_retracted.T @ A_retracted
            dist_A = float(np.sqrt(np.sum((AtA - I_r) ** 2)))

        # Retract B: (r, d_out) → rows orthonormal → B B^T = I_r
        # Skip if B is near-zero (early training steps before loss diverges from 0)
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(B_np)) or np.sum(B_np ** 2) < 1e-12:
            # B too small to retract meaningfully — skip, report distance from identity
            dist_B = float(np.sqrt(self.rank))  # ||0 - I||_F = sqrt(r)
        else:
            W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
            B_retracted = W2 @ Vh2  # (r, d_out) with B B^T = I_r
            self.lora_b = mx.array(B_retracted.astype(np.float32))
            BtB = B_retracted @ B_retracted.T
            dist_B = float(np.sqrt(np.sum((BtB - I_r) ** 2)))

        return dist_A, dist_B


class LoRALinear(nn.Module):
    """Standard LoRA baseline: ΔW = A @ B, no Stiefel constraint."""

    def __init__(self, base_linear: nn.Module, rank: int, scale: float):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        if hasattr(base_linear, 'group_size'):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]

        rng = np.random.default_rng(SEED)
        A_init = rng.standard_normal((d_in, rank)).astype(np.float32) * (1.0 / math.sqrt(d_in))
        self.lora_a = mx.array(A_init)  # (d_in, r)
        self.lora_b = mx.zeros((rank, d_out))  # (r, d_out)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scale * lora_out


# ──────────────────────────────────────────────────────────────
# Stable rank computation
# ──────────────────────────────────────────────────────────────

def stable_rank_via_svd(A_np: np.ndarray, B_np: np.ndarray) -> float:
    """Compute sr(A @ B) exactly via thin SVD chain.

    For A:(d_in, r) @ B:(r, d_out):
    A = U_A diag(S_A) Vh_A  → A@B = U_A @ M where M = diag(S_A) @ Vh_A @ B
    Since U_A is unitary: sv(A@B) = sv(M).
    """
    if np.sum(A_np ** 2) < 1e-12 or np.sum(B_np ** 2) < 1e-12:
        return 0.0
    if not np.all(np.isfinite(A_np)) or not np.all(np.isfinite(B_np)):
        return 0.0

    _, S_A, Vh_A = np.linalg.svd(A_np.astype(np.float64), full_matrices=False)
    M = np.diag(S_A) @ Vh_A @ B_np.astype(np.float64)  # (r, d_out)
    if not np.all(np.isfinite(M)):
        return 0.0
    _, S_M, _ = np.linalg.svd(M, full_matrices=False)
    frob_sq = float(np.sum(S_M ** 2))
    spec_sq = float(S_M[0] ** 2) if len(S_M) > 0 else 0.0
    return frob_sq / spec_sq if spec_sq > 1e-12 else 0.0


def measure_stable_ranks(lora_modules: list, adapter_cls) -> dict:
    """Measure sr(ΔW) for all LoRA/PoLAR modules. Returns stats."""
    srs = []
    for mod in lora_modules:
        A_np = np.array(mod.lora_a.tolist(), dtype=np.float64)  # (d_in, r)
        B_np = np.array(mod.lora_b.tolist(), dtype=np.float64)  # (r, d_out)
        srs.append(stable_rank_via_svd(A_np, B_np))
    mx.eval()
    if not srs:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(srs)),
        "min": float(np.min(srs)),
        "max": float(np.max(srs)),
        "median": float(np.median(srs)),
    }


# ──────────────────────────────────────────────────────────────
# Adapter injection
# ──────────────────────────────────────────────────────────────

def inject_adapters(model, adapter_cls, rank: int, scale: float) -> list:
    """Replace q_proj in all Gemma 4 layers with adapter. Returns list of modules."""
    modules = []
    for li in range(N_LAYERS):
        layer = model.layers[li]
        original_q = layer.self_attn.q_proj
        adapter = adapter_cls(original_q, rank, scale)
        layer.self_attn.q_proj = adapter
        modules.append(adapter)
    return modules


def retract_all(modules: list) -> tuple[float, float]:
    """Retract all PoLAR modules. Returns (max_dist_A, max_dist_B)."""
    dists_A, dists_B = [], []
    for mod in modules:
        d_A, d_B = mod.retract_to_stiefel()
        dists_A.append(d_A)
        dists_B.append(d_B)
    mx.eval()  # sync after numpy operations modified mx arrays
    return float(max(dists_A)), float(max(dists_B))


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_phase(
    model,
    tokenizer,
    samples: list,
    modules: list,
    n_steps: int,
    do_retract: bool = False,
    phase_name: str = "",
) -> dict:
    """Train for n_steps. Optionally retract PoLAR modules every RETRACT_EVERY."""
    model.freeze()
    for mod in modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {n_params:,}")

    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        B, L, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    stiefel_log = []
    t0 = time.time()

    for step in range(n_steps):
        batch = get_batch_random(samples, BATCH_SIZE, step)
        loss, grads = loss_and_grad(model, batch)

        # Gradient clipping
        from mlx.utils import tree_flatten, tree_map
        grad_list = [(k, v) for k, v in tree_flatten(grads) if isinstance(v, mx.array)]
        if grad_list:
            gnorm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in grad_list))
            if gnorm > GRAD_CLIP:
                scale = GRAD_CLIP / (gnorm + 1e-8)
                grads = tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

        optimizer.update(model, grads)

        # Stiefel retraction for PoLAR
        if do_retract and (step + 1) % RETRACT_EVERY == 0:
            mx.eval(model.parameters())
            max_A, max_B = retract_all(modules)
            stiefel_log.append({
                "step": step + 1,
                "max_dist_A": max_A,
                "max_dist_B": max_B,
            })

        mx.eval(loss, model.parameters())
        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log(f"  [{phase_name}] step {step+1:4d}/{n_steps}: "
                f"loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

    # Final retraction to ensure post-retraction measurements
    if do_retract:
        mx.eval(model.parameters())
        max_A, max_B = retract_all(modules)
        stiefel_log.append({"step": n_steps, "max_dist_A": max_A, "max_dist_B": max_B})
        log(f"  Final retraction: max||UU^T-I||={max_A:.2e}, max||VV^T-I||={max_B:.2e}")

    elapsed = time.time() - t0
    log(f"  Training complete: {elapsed:.1f}s ({elapsed/n_steps:.2f}s/step)")

    return {
        "losses": losses,
        "stiefel_log": stiefel_log,
        "final_loss": losses[-1] if losses else float("nan"),
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────────────────────
# GSM8K evaluation (KC08)
# ──────────────────────────────────────────────────────────────

def load_gsm8k_eval(n: int = N_EVAL_GSM8K) -> list:
    """Load GSM8K test examples for KC08 evaluation."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(SEED + 1)
    indices = list(range(min(n * 2, len(ds))))
    random.shuffle(indices)
    examples = []
    for i in indices:
        item = ds[i]
        match = re.search(r"####\s*([0-9,\-\.]+)", item["answer"])
        if match:
            final = match.group(1).replace(",", "")
            examples.append({"question": item["question"], "answer": final})
        if len(examples) >= n:
            break
    return examples[:n]


def extract_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def eval_gsm8k(model, tokenizer, examples: list) -> float:
    """Evaluate model on GSM8K. Returns accuracy."""
    from mlx_lm import generate as mlx_generate
    correct = 0
    for ex in examples:
        prompt = f"Q: {ex['question']}\nA:"
        try:
            response = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=128, verbose=False,
            )
        except Exception:
            response = ""
        pred = extract_number(response)
        correct += int(pred == ex["answer"])
    return correct / len(examples) if examples else 0.0


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    results = {"is_smoke": IS_SMOKE, "model": MODEL_ID}

    log(f"=== C1.1: PoLAR Joint Stiefel on Gemma 4 E4B ===")
    log(f"Smoke={IS_SMOKE}, steps_r16={STEPS_R16}, steps_r6={STEPS_R6}")

    # Load GSM8K test set once (shared across phases)
    log("\nLoading GSM8K test set for KC08...")
    gsm8k_eval = load_gsm8k_eval(N_EVAL_GSM8K)
    log(f"GSM8K test examples: {len(gsm8k_eval)}")

    # ──────────────────────────────────────────────────────────────
    # Phase 1: PoLAR r=16 — KC07 (sr >= 5) and KC09
    # ──────────────────────────────────────────────────────────────
    log(f"\n=== Phase 1: PoLAR r=16, {STEPS_R16} steps (KC07 + KC09) ===")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("p1-loaded")

    samples = build_training_corpus(tokenizer, n_per_domain=20 if IS_SMOKE else 20)
    log(f"Training corpus: {len(samples)} samples from 5 domains")

    modules_r16 = inject_adapters(model, PoLARLinear, rank=16, scale=SCALE)
    train_r16 = train_phase(
        model, tokenizer, samples, modules_r16,
        n_steps=STEPS_R16, do_retract=True, phase_name="PoLAR-r16"
    )

    # KC07: sr(ΔW) at r=16
    sr_r16 = measure_stable_ranks(modules_r16, PoLARLinear)
    kc07_pass = sr_r16["mean"] >= 5.0
    log(f"KC07 sr(PoLAR r=16): mean={sr_r16['mean']:.2f} min={sr_r16['min']:.2f} "
        f"→ {'PASS' if kc07_pass else 'FAIL'} (threshold 5)")

    # KC09 at r=16 (Stiefel distances from last retraction)
    last_stiefel_r16 = train_r16["stiefel_log"][-1] if train_r16["stiefel_log"] else {}
    kc09_A_r16 = last_stiefel_r16.get("max_dist_A", float("inf"))
    kc09_B_r16 = last_stiefel_r16.get("max_dist_B", float("inf"))
    kc09_r16_pass = kc09_A_r16 < 0.01 and kc09_B_r16 < 0.01
    log(f"KC09 r=16: ||UU^T-I||_max={kc09_A_r16:.2e} ||VV^T-I||_max={kc09_B_r16:.2e} "
        f"→ {'PASS' if kc09_r16_pass else 'FAIL'}")

    results["phase1_r16"] = {
        "train": train_r16,
        "stable_rank": sr_r16,
        "kc07": {"value": sr_r16["mean"], "threshold": 5.0, "pass": kc07_pass},
        "kc09_r16": {"dist_A": kc09_A_r16, "dist_B": kc09_B_r16, "pass": kc09_r16_pass},
    }

    cleanup(model)

    # ──────────────────────────────────────────────────────────────
    # Phase 2: PoLAR r=6 — KC08 (PoLAR part) + KC09 at r=6
    # ──────────────────────────────────────────────────────────────
    log(f"\n=== Phase 2: PoLAR r=6, {STEPS_R6} steps (KC08 + KC09) ===")
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("p2-loaded")

    modules_polar_r6 = inject_adapters(model, PoLARLinear, rank=6, scale=SCALE)
    train_polar_r6 = train_phase(
        model, tokenizer, samples, modules_polar_r6,
        n_steps=STEPS_R6, do_retract=True, phase_name="PoLAR-r6"
    )

    # KC09 at r=6
    last_stiefel_r6 = train_polar_r6["stiefel_log"][-1] if train_polar_r6["stiefel_log"] else {}
    kc09_A = last_stiefel_r6.get("max_dist_A", float("inf"))
    kc09_B = last_stiefel_r6.get("max_dist_B", float("inf"))
    kc09_pass = kc09_A < 0.01 and kc09_B < 0.01
    log(f"KC09 r=6: ||UU^T-I||_max={kc09_A:.2e} ||VV^T-I||_max={kc09_B:.2e} "
        f"→ {'PASS' if kc09_pass else 'FAIL'}")

    # Stable rank for Phase 2
    sr_polar_r6 = measure_stable_ranks(modules_polar_r6, PoLARLinear)
    log(f"sr(PoLAR r=6): mean={sr_polar_r6['mean']:.2f} (expected ~6 from Theorem 2)")

    # KC08 part 1: PoLAR accuracy on GSM8K
    log("Evaluating PoLAR r=6 on GSM8K...")
    polar_r6_acc = eval_gsm8k(model, tokenizer, gsm8k_eval)
    log(f"PoLAR r=6 GSM8K: {polar_r6_acc:.1%}")

    results["phase2_polar_r6"] = {
        "train": train_polar_r6,
        "stable_rank": sr_polar_r6,
        "gsm8k_acc": polar_r6_acc,
        "kc09": {"dist_A": kc09_A, "dist_B": kc09_B, "pass": kc09_pass},
    }

    cleanup(model)

    # ──────────────────────────────────────────────────────────────
    # Phase 3: LoRA r=6 baseline — KC08 (LoRA baseline)
    # ──────────────────────────────────────────────────────────────
    log(f"\n=== Phase 3: LoRA baseline r=6, {STEPS_R6} steps (KC08 baseline) ===")
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("p3-loaded")

    modules_lora_r6 = inject_adapters(model, LoRALinear, rank=6, scale=SCALE)
    train_lora_r6 = train_phase(
        model, tokenizer, samples, modules_lora_r6,
        n_steps=STEPS_R6, do_retract=False, phase_name="LoRA-r6"
    )

    # Stable rank of LoRA (expected low, ~1-3 from rank collapse)
    sr_lora_r6 = measure_stable_ranks(modules_lora_r6, LoRALinear)
    log(f"sr(LoRA r=6): mean={sr_lora_r6['mean']:.2f} (expected 1-3 rank collapse)")

    # KC08 part 2: LoRA accuracy on GSM8K
    log("Evaluating LoRA r=6 on GSM8K...")
    lora_r6_acc = eval_gsm8k(model, tokenizer, gsm8k_eval)
    log(f"LoRA r=6 GSM8K: {lora_r6_acc:.1%}")

    results["phase3_lora_r6"] = {
        "train": train_lora_r6,
        "stable_rank": sr_lora_r6,
        "gsm8k_acc": lora_r6_acc,
    }

    cleanup(model)

    # ──────────────────────────────────────────────────────────────
    # Kill criteria summary
    # ──────────────────────────────────────────────────────────────
    kc08_pass = polar_r6_acc >= lora_r6_acc

    log("\n=== Kill Criteria Summary ===")
    log(f"KC07: sr(PoLAR r=16)={sr_r16['mean']:.2f} >= 5 → {'PASS' if kc07_pass else 'FAIL'}")
    log(f"KC08: PoLAR={polar_r6_acc:.1%} vs LoRA={lora_r6_acc:.1%} → {'PASS' if kc08_pass else 'FAIL'}")
    log(f"KC09: ||UU^T-I||={kc09_A:.2e}, ||VV^T-I||={kc09_B:.2e} < 0.01 → {'PASS' if kc09_pass else 'FAIL'}")
    log(f"KC09-r16: ||UU^T-I||={kc09_A_r16:.2e}, ||VV^T-I||={kc09_B_r16:.2e} → {'PASS' if kc09_r16_pass else 'FAIL'}")

    results["kill_criteria"] = {
        "KC07": {"sr_mean": sr_r16["mean"], "threshold": 5.0, "pass": kc07_pass},
        "KC08": {"polar_acc": polar_r6_acc, "lora_acc": lora_r6_acc, "pass": kc08_pass},
        "KC09": {"dist_A": kc09_A, "dist_B": kc09_B, "threshold": 0.01, "pass": kc09_pass},
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults → {RESULTS_FILE}")

    all_pass = kc07_pass and kc08_pass and kc09_pass
    log(f"\n{'ALL K PASS' if all_pass else 'SOME K FAIL'}: "
        f"KC07={'P' if kc07_pass else 'F'} "
        f"KC08={'P' if kc08_pass else 'F'} "
        f"KC09={'P' if kc09_pass else 'F'}")


if __name__ == "__main__":
    main()
