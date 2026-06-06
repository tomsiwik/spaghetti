#!/usr/bin/env python3
"""Q_wrong measurement: cross-domain interference from M2P adapter.

Epsilon-map blind spot #2: what happens when the GSM8K M2P adapter is
applied to prompts from structurally different domains?

Kill criteria:
  K944: Q_wrong measured for at least 3 domain pairs
        (GSM8K adapter → sort, reverse, count_even)

NO TRAINING in this experiment — pure measurement using v4 M2P weights.

Supports SMOKE_TEST=1 for quick validation (5 examples/domain, ~5 min).

References:
  Ha et al. (arXiv:1609.09106) — HyperNetworks
  Hu et al. (arXiv:2106.09685) — LoRA
  Finding #378/379: Q_right=1.433 for GSM8K M2P v4
"""

import gc
import json
import os
import random
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ------------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
OUTPUT_SCALE = 0.032
N_EXAMPLES = 5 if IS_SMOKE else 50
MAX_GEN_TOKENS = 32
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"

V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"
V4_M2P_PATH = V4_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Model dims for Qwen3-0.6B
N_LAYERS = 28
D_MODEL = 1024
Q_PROJ_OUT = 2048
V_PROJ_OUT = 1024


# ---- Utilities ---------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ---- Domain generation -------------------------------------------------------

WORDS_FOR_SORT = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
    "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya",
    "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla",
    "watermelon", "yam", "zucchini", "avocado", "blueberry",
]

WORDS_FOR_REVERSE = [
    "cat", "dog", "fish", "bird", "frog", "bear", "lion", "wolf",
    "duck", "deer", "fox", "owl", "bat", "cow", "pig", "hen",
    "ant", "bee", "fly", "crab", "moth", "worm", "slug", "newt",
]

SORT_FEW_SHOT = (
    "Sort these words alphabetically: mango, apple, banana\n"
    "Answer: apple, banana, mango\n\n"
    "Sort these words alphabetically: dog, cat, fox, ant\n"
    "Answer: ant, cat, dog, fox\n\n"
)

REVERSE_FEW_SHOT = (
    "Reverse the order of these words: cat dog fish\n"
    "Answer: fish dog cat\n\n"
    "Reverse the order of these words: one two three four\n"
    "Answer: four three two one\n\n"
)

EVEN_FEW_SHOT = (
    "How many even numbers are in this list: [2, 7, 4, 9]\n"
    "Answer: 2\n\n"
    "How many even numbers are in this list: [3, 5, 8, 1, 6]\n"
    "Answer: 2\n\n"
)


def generate_sort_examples(n: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        k = rng.randint(3, 5)
        sample = rng.sample(WORDS_FOR_SORT, k)
        shuffled = sample.copy()
        rng.shuffle(shuffled)
        expected = ", ".join(sorted(sample))
        prompt = SORT_FEW_SHOT + f"Sort these words alphabetically: {', '.join(shuffled)}\nAnswer:"
        examples.append({"prompt": prompt, "answer": expected, "words": sample})
    return examples


def generate_reverse_examples(n: int, seed: int = SEED + 1) -> list:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        k = rng.randint(3, 5)
        sample = rng.sample(WORDS_FOR_REVERSE, k)
        expected = " ".join(reversed(sample))
        prompt = REVERSE_FEW_SHOT + f"Reverse the order of these words: {' '.join(sample)}\nAnswer:"
        examples.append({"prompt": prompt, "answer": expected, "words": sample})
    return examples


def generate_count_even_examples(n: int, seed: int = SEED + 2) -> list:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        k = rng.randint(3, 7)
        nums = [rng.randint(1, 20) for _ in range(k)]
        count = sum(1 for x in nums if x % 2 == 0)
        expected = str(count)
        prompt = EVEN_FEW_SHOT + f"How many even numbers are in this list: {nums}\nAnswer:"
        examples.append({"prompt": prompt, "answer": expected, "nums": nums})
    return examples


# ---- Answer parsing ----------------------------------------------------------

def parse_sort_answer(text: str, expected_words: list) -> str | None:
    """Extract the first word sequence that matches sorted words."""
    text_lower = text.strip().lower()
    # Try comma-separated
    words_lower = [w.lower() for w in expected_words]
    # Look for any permutation of words in first 200 chars
    first_line = text_lower.split("\n")[0].strip()
    # Remove punctuation/spaces and check
    found = re.findall(r'[a-z]+', first_line)
    if found:
        return ", ".join(found)
    return first_line


def parse_reverse_answer(text: str) -> str | None:
    """Extract the first word sequence."""
    first_line = text.strip().split("\n")[0].strip().lower()
    found = re.findall(r'[a-z]+', first_line)
    if found:
        return " ".join(found)
    return first_line


def parse_count_even_answer(text: str) -> str | None:
    """Extract the first integer from the answer."""
    match = re.search(r'\b(\d+)\b', text.strip())
    if match:
        return match.group(1)
    return None


def check_sort_correct(pred_text: str, expected: str) -> bool:
    """Check if predicted sort order matches expected."""
    if pred_text is None:
        return False
    pred_words = re.findall(r'[a-z]+', pred_text.lower())
    exp_words = re.findall(r'[a-z]+', expected.lower())
    return pred_words == exp_words


def check_reverse_correct(pred_text: str, expected: str) -> bool:
    """Check if predicted reverse order matches expected."""
    if pred_text is None:
        return False
    pred_words = re.findall(r'[a-z]+', pred_text.lower())
    exp_words = re.findall(r'[a-z]+', expected.lower())
    return pred_words == exp_words


def check_count_even_correct(pred_text: str, expected: str) -> bool:
    """Check if predicted count matches expected."""
    return pred_text is not None and pred_text.strip() == expected.strip()


DOMAIN_CONFIGS = {
    "sort_words": {
        "generator": generate_sort_examples,
        "parse_fn": lambda text, ex: parse_sort_answer(text, ex["words"]),
        "check_fn": check_sort_correct,
        "label": "Sort Words (alphabetical ordering)",
    },
    "reverse_words": {
        "generator": generate_reverse_examples,
        "parse_fn": lambda text, ex: parse_reverse_answer(text),
        "check_fn": check_reverse_correct,
        "label": "Reverse Words (sequence reversal)",
    },
    "count_even": {
        "generator": generate_count_even_examples,
        "parse_fn": lambda text, ex: parse_count_even_answer(text),
        "check_fn": check_count_even_correct,
        "label": "Count Even Numbers",
    },
}


# ---- LoRA functional forward (identical to v4 — proven design) ---------------

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection: y = linear(x) + scale * (x @ A) @ B"""
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def functional_attention_forward(
    attn, x: mx.array, B_q: mx.array, B_v: mx.array,
    A_q: mx.array, A_v: mx.array, lora_scale: float, mask, cache=None,
) -> mx.array:
    """Functional attention forward with LoRA B as tensor args (v4 proven design)."""
    B_batch, L, D = x.shape

    q = functional_lora_proj(x, attn.q_proj.linear, A_q, B_q, lora_scale)
    k = attn.k_proj(x)
    v = functional_lora_proj(x, attn.v_proj.linear, A_v, B_v, lora_scale)

    queries = attn.q_norm(q.reshape(B_batch, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attn.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B_batch, L, -1)
    return attn.o_proj(output)


def extract_hidden_states_functional(
    model, tokens_arr: mx.array,
    A_q_layers: list, A_v_layers: list,
    B_q_zero: list, B_v_zero: list,
) -> mx.array:
    """Extract per-layer mean-pooled hidden states (base model, no LoRA)."""
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)

    layer_states = []
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn,
            x=normed,
            B_q=B_q_zero[li],
            B_v=B_v_zero[li],
            A_q=A_q_layers[li],
            A_v=A_v_layers[li],
            lora_scale=0.0,
            mask=mask,
            cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))

    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- M2P Network (identical to v4 architecture) ------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: hidden states → LoRA B-matrices. Identical to v4."""

    def __init__(self, n_layers, d_model, d_m2p, rank, q_proj_out, v_proj_out,
                 output_scale=0.032):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.output_scale = output_scale

        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        h = mx.mean(layer_hs, axis=0)
        h = nn.gelu(self.enc_linear1(h))
        z = self.enc_linear2(h)
        B_q_layers, B_v_layers = [], []
        for li in range(self.n_layers):
            B_q_layers.append(
                self.b_heads_q[li](z).reshape(self.rank, -1) * self.output_scale
            )
            B_v_layers.append(
                self.b_heads_v[li](z).reshape(self.rank, -1) * self.output_scale
            )
        return B_q_layers, B_v_layers


# ---- Model loading helpers ---------------------------------------------------

def load_lora_a_matrices() -> dict:
    """Load v2 SFT A-matrices."""
    if not V2_LORA_A_PATH.exists():
        raise FileNotFoundError(f"v2 lora_a not found at {V2_LORA_A_PATH}")
    saved = np.load(str(V2_LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} A-matrices from {V2_LORA_A_PATH}")
    return result


def apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj/v_proj and set A-matrices."""
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


def load_m2p_network() -> M2PNetwork:
    """Load v4 M2P weights."""
    if not V4_M2P_PATH.exists():
        raise FileNotFoundError(f"v4 M2P weights not found at {V4_M2P_PATH}")
    m2p = M2PNetwork(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=Q_PROJ_OUT,
        v_proj_out=V_PROJ_OUT,
        output_scale=OUTPUT_SCALE,
    )
    saved = np.load(str(V4_M2P_PATH))
    weight_list = [(k, mx.array(saved[k])) for k in saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P v4 from {V4_M2P_PATH}")
    return m2p


# ---- Evaluation phase --------------------------------------------------------

def eval_domain(
    domain_name: str,
    examples: list,
    model,
    tokenizer,
    lora_a_dict: dict,
    m2p: M2PNetwork,
    A_q_layers: list,
    A_v_layers: list,
) -> dict:
    """Evaluate base vs M2P-adapted accuracy on a domain.

    Returns dict with base_acc, adapted_acc, q_wrong, and per-example details.
    """
    cfg = DOMAIN_CONFIGS[domain_name]
    log(f"\n  [Domain: {domain_name}] n={len(examples)}")

    B_q_zero = [mx.zeros((LORA_RANK, Q_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]
    B_v_zero = [mx.zeros((LORA_RANK, V_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]

    # ---- Base model evaluation (zero B-matrices → effectively base forward) ----
    log("    Phase A: base model")
    base_correct = 0
    base_details = []
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]

        # Inject zero B-matrices (no adaptation)
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_zero[li]
            layer.self_attn.v_proj.lora_b = B_v_zero[li]
        mx.eval(model.parameters())

        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        pred = cfg["parse_fn"](generated, ex)
        correct = cfg["check_fn"](pred, ex["answer"])
        if correct:
            base_correct += 1
        base_details.append({
            "pred": pred, "gold": ex["answer"],
            "correct": correct, "generated": generated[:80],
        })

        if i == 0:
            log(f"    [BASE-DEBUG] generated[:80]={generated[:80]!r}")
            log(f"    [BASE-DEBUG] pred={pred!r}, gold={ex['answer']!r}, correct={correct}")

    base_acc = base_correct / len(examples)
    log(f"    Base accuracy: {base_acc:.3f} ({base_correct}/{len(examples)})")

    # ---- M2P-adapted evaluation -----------------------------------------------
    log("    Phase B: M2P-adapted (GSM8K adapter, wrong domain)")
    adapted_correct = 0
    adapted_details = []
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Step 1: Extract hidden states (base model forward)
        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        mx.eval(layer_hs)

        # Step 2: M2P generates B-matrices conditioned on WRONG-domain hidden states
        B_q_layers, B_v_layers = m2p(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        # Step 3: Inject B-matrices
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_layers[li]
            layer.self_attn.v_proj.lora_b = B_v_layers[li]
        mx.eval(model.parameters())

        # Step 4: Generate
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        pred = cfg["parse_fn"](generated, ex)
        correct = cfg["check_fn"](pred, ex["answer"])
        if correct:
            adapted_correct += 1
        adapted_details.append({
            "pred": pred, "gold": ex["answer"],
            "correct": correct, "generated": generated[:80],
        })

        del tokens_arr, layer_hs, B_q_layers, B_v_layers

        if i == 0:
            log(f"    [ADAPTED-DEBUG] generated[:80]={generated[:80]!r}")
            log(f"    [ADAPTED-DEBUG] pred={pred!r}, gold={ex['answer']!r}, correct={correct}")

    adapted_acc = adapted_correct / len(examples)
    log(f"    Adapted accuracy: {adapted_acc:.3f} ({adapted_correct}/{len(examples)})")

    # Q_wrong = (adapted - base) / max(base, 0.01)
    q_wrong = (adapted_acc - base_acc) / max(base_acc, 0.01)
    log(f"    Q_wrong = {q_wrong:.4f} (adapted={adapted_acc:.3f}, base={base_acc:.3f})")

    return {
        "domain": domain_name,
        "n_examples": len(examples),
        "base_correct": base_correct,
        "adapted_correct": adapted_correct,
        "base_acc": base_acc,
        "adapted_acc": adapted_acc,
        "q_wrong": q_wrong,
        "base_details": base_details,
        "adapted_details": adapted_details,
    }


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("Q_wrong Measurement: Cross-Domain Interference from M2P Adapter")
    log(f"IS_SMOKE={IS_SMOKE}, N_EXAMPLES={N_EXAMPLES}")
    log("=" * 70)

    # ---- Load model and assets -----------------------------------------------
    log("\n[Phase 0] Loading model and assets")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log(f"  Loaded {MODEL_ID}")

    lora_a_dict = load_lora_a_matrices()
    apply_lora_structure(model, lora_a_dict)

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(N_LAYERS)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(N_LAYERS)]

    m2p = load_m2p_network()
    log_memory("after loading")

    # ---- Generate domain examples --------------------------------------------
    log("\n[Phase 1] Generating domain examples")
    domain_examples = {
        "sort_words": generate_sort_examples(N_EXAMPLES),
        "reverse_words": generate_reverse_examples(N_EXAMPLES),
        "count_even": generate_count_even_examples(N_EXAMPLES),
    }
    for name, exs in domain_examples.items():
        log(f"  {name}: {len(exs)} examples")
        if exs:
            log(f"    Sample prompt (first 100 chars): {exs[0]['prompt'][-100:]!r}")
            log(f"    Expected answer: {exs[0]['answer']!r}")

    # ---- Evaluate each domain -----------------------------------------------
    log("\n[Phase 2] Evaluating Q_wrong for each domain pair")
    domain_results = {}
    for domain_name, examples in domain_examples.items():
        result = eval_domain(
            domain_name=domain_name,
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            lora_a_dict=lora_a_dict,
            m2p=m2p,
            A_q_layers=A_q_layers,
            A_v_layers=A_v_layers,
        )
        domain_results[domain_name] = result
        log_memory(f"after {domain_name}")
        gc.collect()
        mx.clear_cache()

    # ---- K944: at least 3 domain pairs measured ------------------------------
    measured_pairs = [k for k, v in domain_results.items()
                      if "q_wrong" in v and v["n_examples"] > 0]
    k944_pass = len(measured_pairs) >= 3

    log("\n" + "=" * 70)
    log("[Results] Q_wrong Summary")
    log("=" * 70)
    for domain, res in domain_results.items():
        q = res["q_wrong"]
        sign = "+" if q > 0.05 else ("-" if q < -0.05 else "≈0")
        log(f"  {domain:20s}: Q_wrong={q:+.4f} {sign}  "
            f"(base={res['base_acc']:.3f}, adapted={res['adapted_acc']:.3f})")

    log(f"\n  K944 (≥3 pairs measured): {'PASS' if k944_pass else 'KILL'} "
        f"({len(measured_pairs)}/3 measured)")

    total_time = time.time() - t_start
    log(f"\n  Total time: {total_time:.1f}s")

    # ---- Save results --------------------------------------------------------
    results = {
        "experiment": "q_wrong_real_domains",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "n_examples": N_EXAMPLES,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "adapter_source": "m2p_qwen06b_gsm8k_v4",
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
        },
        "domain_results": {
            k: {
                "domain": v["domain"],
                "n_examples": v["n_examples"],
                "base_acc": v["base_acc"],
                "adapted_acc": v["adapted_acc"],
                "q_wrong": v["q_wrong"],
                "base_correct": v["base_correct"],
                "adapted_correct": v["adapted_correct"],
            }
            for k, v in domain_results.items()
        },
        "k944_measured_pairs": measured_pairs,
        "k944_pass": k944_pass,
        "kill_criteria": {
            "K944_q_wrong_3_pairs_measured": "PASS" if k944_pass else "KILL",
        },
        "total_time_s": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
