#!/usr/bin/env python3
"""Per-user M2P adapter PoC: 3 simulated personas with behavioral differentiation.

Kill criteria:
  K940: Cohen's d > 0.3 between concise vs step-by-step persona outputs (response length)
  K941: User adapter composes with domain adapter (<10% quality loss on GSM8K)

Design (MATH.md Theorem 1-3):
  - 3 personas: concise ("#### N"), code ("answer = N; #### N"), step (original GSM8K)
  - Each M2P trained on 50 persona-style examples, warm-started from v4 weights
  - Behavioral measure: response token length (clear signal, large predicted effect d≈3.5)
  - Composition: B_composed = 0.5*B_domain + 0.5*B_step_persona on 50 GSM8K eval

References:
  Ha et al. (arXiv:1609.09106) — HyperNetworks
  SHINE (arXiv:2602.06358) — functional LoRA forward
  exp_m2p_qwen06b_gsm8k_v4 — base M2P architecture (unchanged)

Supports SMOKE_TEST=1 for quick validation (~5 min).
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config -----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"

LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
OUTPUT_SCALE = 0.032

N_TRAIN_PERSONA = 5 if IS_SMOKE else 50
N_TEST = 5 if IS_SMOKE else 50
M2P_STEPS = 10 if IS_SMOKE else 300
LR = 5e-5
LR_WARMUP = 2 if IS_SMOKE else 30
MAX_SEQ_LEN = 128 if IS_SMOKE else 512
# Behavioral eval: enough room for step-by-step (120 tokens) to show style
MAX_GEN_TOKENS_BEHAV = 64 if IS_SMOKE else 200
# Composition eval: same as v4 for fair accuracy comparison
MAX_GEN_TOKENS_COMP = 64 if IS_SMOKE else 384
SEED = 42

PERSONAS = ["concise", "code", "step"]

EXPERIMENT_DIR = Path(__file__).parent
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V4_M2P_PATH = V4_DIR / "m2p_weights.npz"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix (for K941 composition accuracy — same as v4 for fair comparison)
FEW_SHOT_PREFIX = (
    "Solve the math problem step by step and end with '#### <answer>'.\n\n"
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. "
    "Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
)

# Behavioral prompt: no few-shot — matches training format, lets M2P style show clearly
# (Training: "Question: X\nAnswer: style_answer" — inference must match for style signal)
BEHAVIORAL_PROMPT_TMPL = "Question: {question}\nAnswer:"


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


def extract_gsm8k_answer(text: str):
    """Extract final numeric answer from GSM8K #### format."""
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


def cohen_d(group1: list, group2: list) -> float:
    """Cohen's d effect size: |μ1 - μ2| / pooled std. Returns Python float."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    arr1 = np.array(group1, dtype=np.float64)
    arr2 = np.array(group2, dtype=np.float64)
    mean1, mean2 = float(arr1.mean()), float(arr2.mean())
    var1 = float(arr1.var(ddof=1))
    var2 = float(arr2.var(ddof=1))
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(max(pooled_var, 0.0))
    if pooled_std < 1e-9:
        return 0.0 if abs(mean1 - mean2) < 1e-9 else 99.0  # capped sentinel for JSON
    return float(abs(mean1 - mean2) / pooled_std)


# ---- Persona dataset creation ------------------------------------------------

def make_persona_answer(original_answer: str, persona: str) -> str:
    """Transform a GSM8K answer into persona-specific style.

    concise: "#### N" only — minimal, just the final answer
    code:    "answer = N  # computed\n#### N" — code-adjacent style
    step:    original answer unchanged — verbose multi-step reasoning
    """
    number = extract_gsm8k_answer(original_answer)
    if number is None:
        number = "0"

    if persona == "concise":
        return f"#### {number}"
    elif persona == "code":
        return f"answer = {number}  # computed\n#### {number}"
    else:  # step
        return original_answer


def make_persona_dataset(examples: list, persona: str) -> list:
    """Create persona-style training examples from raw GSM8K."""
    return [
        {"question": ex["question"], "answer": make_persona_answer(ex["answer"], persona)}
        for ex in examples
    ]


# ---- A-matrix loading -------------------------------------------------------

def load_lora_a_matrices() -> dict:
    """Load lora_a matrices from v2 SFT phase (same as v4)."""
    if not V2_LORA_A_PATH.exists():
        raise FileNotFoundError(f"v2 lora_a not found: {V2_LORA_A_PATH}")
    saved = np.load(str(V2_LORA_A_PATH))
    result = {}
    for key in saved.files:
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} lora_a matrices from v2")
    return result


# ---- LoRA structure ---------------------------------------------------------

def apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, freeze all (identical to v4)."""
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


# ---- Functional forward (IDENTICAL to v4 — proven design) -------------------

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def functional_attention_forward(attn, x, B_q, B_v, A_q, A_v,
                                  lora_scale, mask, cache=None):
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


def model_forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers,
                              A_q_layers, A_v_layers, lora_scale=LORA_SCALE):
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn, x=normed,
            B_q=B_q_layers[li], B_v=B_v_layers[li],
            A_q=A_q_layers[li], A_v=A_v_layers[li],
            lora_scale=lora_scale, mask=mask, cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = qwen3_model.norm(h)
    if model.args.tie_word_embeddings:
        return qwen3_model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def extract_hidden_states_functional(model, tokens_arr, A_q_layers, A_v_layers,
                                      B_q_zero, B_v_zero):
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    layer_states = []
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn, x=normed,
            B_q=B_q_zero[li], B_v=B_v_zero[li],
            A_q=A_q_layers[li], A_v=A_v_layers[li],
            lora_scale=0.0, mask=mask, cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))
    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- M2P Architecture (IDENTICAL to v4) ------------------------------------

class M2PNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_m2p, rank, q_proj_out, v_proj_out,
                 output_scale=0.032):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_m2p = d_m2p
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
            B_q_layers.append(self.b_heads_q[li](z).reshape(self.rank, -1) * self.output_scale)
            B_v_layers.append(self.b_heads_v[li](z).reshape(self.rank, -1) * self.output_scale)
        return B_q_layers, B_v_layers


def make_m2p(model_dims: dict) -> M2PNetwork:
    return M2PNetwork(
        n_layers=model_dims["n_layers"],
        d_model=model_dims["d_model"],
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=model_dims["q_proj_out"],
        v_proj_out=model_dims["v_proj_out"],
        output_scale=OUTPUT_SCALE,
    )


def load_m2p_from_file(path: Path, model_dims: dict) -> M2PNetwork:
    m2p = make_m2p(model_dims)
    saved = np.load(str(path))
    weight_list = [(k, mx.array(saved[k])) for k in saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    return m2p


def save_m2p(m2p: M2PNetwork, path: Path) -> None:
    params = dict(tree_flatten(m2p.parameters()))
    np.savez(str(path), **{k: np.array(v.astype(mx.float32)) for k, v in params.items()})


# ---- Tokenization -----------------------------------------------------------

def tokenize_examples(tokenizer, examples: list) -> list:
    result = []
    for ex in examples:
        text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Phase 0: Load model dims from v4 ---------------------------------------

def phase_load_dims() -> dict:
    log("\n" + "=" * 70)
    log("[Phase 0] Loading model dims from v4 results")
    log("=" * 70)
    v4_results_path = V4_DIR / "results.json"
    if not v4_results_path.exists():
        raise FileNotFoundError(f"v4 results not found: {v4_results_path}")
    with open(v4_results_path) as f:
        v4 = json.load(f)
    dims = {k: v4["config"][k] for k in [
        "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "q_proj_out", "v_proj_out"
    ]}
    v4_acc = v4["m2p_accuracy"]
    base_acc = v4["base_accuracy"]
    log(f"  model dims: {dims}")
    log(f"  v4 M2P acc: {v4_acc:.4f}, base acc: {base_acc:.4f}")
    return dims, v4_acc, base_acc


# ---- Phase 1: Train one persona M2P -----------------------------------------

def phase_train_persona(persona: str, train_examples: list, model_dims: dict) -> dict:
    """Train one persona M2P from v4 warm start. Save weights to disk."""
    log(f"\n{'=' * 70}")
    log(f"[Train:{persona}] {M2P_STEPS} steps on {len(train_examples)} persona examples")
    log(f"{'=' * 70}")
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    persona_data = make_persona_dataset(train_examples, persona)
    tokenized = tokenize_examples(tokenizer, persona_data)
    log(f"  Tokenized {len(tokenized)} sequences for '{persona}' persona")

    # Sample answer lengths for verification
    sample_answers = [ex["answer"] for ex in persona_data[:5]]
    sample_lens = [len(tokenizer.encode(a)) for a in sample_answers]
    log(f"  Sample answer lengths: {sample_lens} (mean={sum(sample_lens)/len(sample_lens):.1f})")

    m2p = make_m2p(model_dims)
    mx.eval(m2p.parameters())

    # Warm start from v4 weights
    if V4_M2P_PATH.exists():
        v4_saved = np.load(str(V4_M2P_PATH))
        m2p.load_weights([(k, mx.array(v4_saved[k])) for k in v4_saved.files])
        mx.eval(m2p.parameters())
        log(f"  Warm start from v4 M2P weights")
    else:
        log(f"  [WARN] v4 M2P not found — fresh init")

    rng = random.Random(SEED + hash(persona) % 1000)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    def m2p_loss_fn(m2p_net, tokens_arr):
        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        B_q_layers, B_v_layers = m2p_net(layer_hs)
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # K940-prerequisite: grad smoke test
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)
    grad_norms = [float(mx.sum(g ** 2).item())
                  for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm = math.sqrt(sum(grad_norms))
    log(f"  grad_norm at step 0 = {grad_norm:.4f}")
    if grad_norm == 0.0:
        log(f"  FAIL: zero gradients for persona '{persona}'")
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return {"persona": persona, "status": "kill_zero_grad", "grad_norm": 0.0}
    del smoke_tokens, smoke_loss, smoke_grads

    gc.disable()
    losses = []
    for step in range(M2P_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, M2P_STEPS // 5) == 0 or (step + 1) == M2P_STEPS:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  [{persona}] step {step+1}/{M2P_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    persona_path = EXPERIMENT_DIR / f"m2p_{persona}.npz"
    save_m2p(m2p, persona_path)
    log(f"  Saved {persona} M2P → {persona_path}")
    log(f"  Final loss: {final_loss:.4f} | Time: {time.time()-t0:.1f}s")
    log_memory(f"post-train-{persona}")

    cleanup(m2p, model, tokenizer, optimizer)
    return {"persona": persona, "final_loss": float(final_loss), "grad_norm": grad_norm,
            "steps": M2P_STEPS, "status": "done"}


# ---- Phase 2: Behavioral evaluation ----------------------------------------

def phase_behavioral_eval(test_examples: list, model_dims: dict,
                          tokenizer_ref) -> dict:
    """Evaluate all 3 persona M2Ps on same 50 questions. Measure response lengths.

    Returns dict: persona → list of response token lengths.
    """
    log(f"\n{'=' * 70}")
    log(f"[Phase 2] Behavioral evaluation: {len(test_examples)} questions × 3 personas")
    log(f"{'=' * 70}")
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    lengths = {p: [] for p in PERSONAS}
    first_outputs = {}

    for persona in PERSONAS:
        persona_path = EXPERIMENT_DIR / f"m2p_{persona}.npz"
        if not persona_path.exists():
            log(f"  [WARN] {persona} M2P not found, skipping")
            continue

        m2p = load_m2p_from_file(persona_path, model_dims)
        log(f"\n  Evaluating persona: {persona}")

        for i, ex in enumerate(test_examples):
            # No few-shot: matches training format (Question+Answer only)
            # This lets M2P-encoded style dominate rather than few-shot prior
            prompt = BEHAVIORAL_PROMPT_TMPL.format(question=ex["question"])
            prompt_ids = tokenizer.encode(prompt)
            tokens_arr = mx.array(prompt_ids)[None, :]

            layer_hs = extract_hidden_states_functional(
                model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
            )
            mx.eval(layer_hs)

            B_q_layers, B_v_layers = m2p(layer_hs)
            mx.eval(*B_q_layers, *B_v_layers)

            # Inject B into LoRA modules for generation
            for li, layer in enumerate(model.model.layers):
                layer.self_attn.q_proj.lora_b = B_q_layers[li]
                layer.self_attn.v_proj.lora_b = B_v_layers[li]
            mx.eval(model.parameters())

            generated = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=MAX_GEN_TOKENS_BEHAV, verbose=False,
            )

            resp_len = len(tokenizer.encode(generated))
            lengths[persona].append(resp_len)

            if i == 0:
                first_outputs[persona] = generated[:200]
                log(f"    [{persona}] Sample output: {generated[:150]!r}")
                log(f"    [{persona}] Token length: {resp_len}")

            del tokens_arr, layer_hs, B_q_layers, B_v_layers

        cleanup(m2p)
        log(f"  {persona}: mean_len={np.mean(lengths[persona]):.1f} "
            f"std={np.std(lengths[persona]):.1f} n={len(lengths[persona])}")

    cleanup(model, tokenizer)
    log(f"\n  Behavioral eval time: {time.time()-t0:.1f}s")
    log_memory("post-behavioral-eval")

    return {"lengths": lengths, "first_outputs": first_outputs}


# ---- Phase 3: Composition test (K941) --------------------------------------

def phase_composition_test(test_examples: list, model_dims: dict,
                            base_acc: float, v4_acc: float) -> dict:
    """Test: B_composed = 0.5*B_domain + 0.5*B_step on GSM8K accuracy.

    Quality loss = (acc_domain - acc_composed) / acc_domain
    K941 PASS if quality loss < 10%.
    """
    log(f"\n{'=' * 70}")
    log(f"[Phase 3] Composition test: domain(v4) + step persona on {len(test_examples)} questions")
    log(f"{'=' * 70}")
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Check that both M2P files exist
    step_path = EXPERIMENT_DIR / "m2p_step.npz"
    if not V4_M2P_PATH.exists():
        log(f"  [SKIP] v4 M2P not found: {V4_M2P_PATH}")
        return {"k941_status": "skipped", "reason": "v4_m2p_missing"}
    if not step_path.exists():
        log(f"  [SKIP] step M2P not found: {step_path}")
        return {"k941_status": "skipped", "reason": "step_m2p_missing"}

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    m2p_domain = load_m2p_from_file(V4_M2P_PATH, model_dims)
    m2p_step = load_m2p_from_file(step_path, model_dims)
    log(f"  Loaded domain (v4) and step M2Ps")

    # Evaluate composed adapter
    correct_composed = 0
    correct_domain = 0

    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        mx.eval(layer_hs)

        # Generate B from both M2Ps
        B_q_domain, B_v_domain = m2p_domain(layer_hs)
        B_q_step, B_v_step = m2p_step(layer_hs)
        mx.eval(*B_q_domain, *B_v_domain, *B_q_step, *B_v_step)

        # Eval domain alone
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_domain[li]
            layer.self_attn.v_proj.lora_b = B_v_domain[li]
        mx.eval(model.parameters())
        gen_domain = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS_COMP, verbose=False)

        # Eval composed
        B_q_comp = [0.5 * bq_d + 0.5 * bq_s
                    for bq_d, bq_s in zip(B_q_domain, B_q_step)]
        B_v_comp = [0.5 * bv_d + 0.5 * bv_s
                    for bv_d, bv_s in zip(B_v_domain, B_v_step)]
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_comp[li]
            layer.self_attn.v_proj.lora_b = B_v_comp[li]
        mx.eval(model.parameters())
        gen_composed = mlx_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=MAX_GEN_TOKENS_COMP, verbose=False)

        gold = extract_gsm8k_answer(ex["answer"])
        pred_domain = extract_gsm8k_answer(gen_domain)
        pred_composed = extract_gsm8k_answer(gen_composed)

        if pred_domain is not None and gold is not None and pred_domain == gold:
            correct_domain += 1
        if pred_composed is not None and gold is not None and pred_composed == gold:
            correct_composed += 1

        del tokens_arr, layer_hs, B_q_domain, B_v_domain, B_q_step, B_v_step
        del B_q_comp, B_v_comp

        if (i + 1) % max(1, len(test_examples) // 4) == 0 or (i + 1) == len(test_examples):
            log(f"  [{i+1}/{len(test_examples)}] domain_acc={correct_domain/(i+1):.3f} "
                f"composed_acc={correct_composed/(i+1):.3f}")

    n = len(test_examples)
    acc_domain_here = correct_domain / n
    acc_composed = correct_composed / n

    # K941: quality loss < 10%
    # Use v4_acc (measured over n=500) as domain reference (more reliable)
    quality_loss_vs_v4 = (v4_acc - acc_composed) / max(v4_acc, 1e-9)
    quality_loss_local = (acc_domain_here - acc_composed) / max(acc_domain_here, 1e-9)
    k941_pass = bool(quality_loss_vs_v4 < 0.10)

    log(f"\n  Domain acc (v4, n=500): {v4_acc:.4f}")
    log(f"  Domain acc (local, n={n}): {acc_domain_here:.4f}")
    log(f"  Composed acc (local, n={n}): {acc_composed:.4f}")
    log(f"  Quality loss vs v4: {quality_loss_vs_v4:.4f} ({quality_loss_vs_v4*100:.1f}%)")
    log(f"  [K941] {'PASS' if k941_pass else 'FAIL'}: quality_loss < 10%: "
        f"{quality_loss_vs_v4*100:.1f}%")

    log(f"\n  Composition test time: {time.time()-t0:.1f}s")
    log_memory("post-composition")

    cleanup(m2p_domain, m2p_step, model, tokenizer)

    return {
        "acc_domain_v4": v4_acc,
        "acc_domain_local": acc_domain_here,
        "acc_composed": acc_composed,
        "correct_domain_local": correct_domain,
        "correct_composed": correct_composed,
        "n": n,
        "quality_loss_vs_v4": round(quality_loss_vs_v4, 4),
        "quality_loss_local": round(quality_loss_local, 4),
        "k941_pass": k941_pass,
        "k941_status": "PASS" if k941_pass else "FAIL",
    }


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("Per-user M2P adapter PoC — Persona behavioral differentiation")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN_PERSONA={N_TRAIN_PERSONA} | N_TEST={N_TEST} | M2P_STEPS={M2P_STEPS}")
    log(f"PERSONAS: {PERSONAS}")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load model dims from v4
    model_dims, v4_acc, base_acc = phase_load_dims()

    # Load GSM8K data
    log("\n[Data] Loading GSM8K...")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")

    rng = random.Random(SEED)
    all_train = list(ds["train"])
    rng.shuffle(all_train)
    train_pool = all_train[:N_TRAIN_PERSONA]  # 50 examples for all personas

    all_test = list(ds["test"])
    rng2 = random.Random(SEED + 100)
    rng2.shuffle(all_test)
    test_examples = all_test[:N_TEST]  # Same 50 questions for all personas
    log(f"  Train pool: {len(train_pool)}, Test: {len(test_examples)}")

    # Phase 1: Train 3 persona M2Ps
    log("\n[Phase 1] Training 3 persona M2Ps...")
    train_results = {}
    for persona in PERSONAS:
        persona_path = EXPERIMENT_DIR / f"m2p_{persona}.npz"
        if persona_path.exists() and not IS_SMOKE:
            log(f"  [SKIP] {persona} M2P already exists: {persona_path}")
            train_results[persona] = {"persona": persona, "status": "already_exists"}
        else:
            train_results[persona] = phase_train_persona(persona, train_pool, model_dims)

    # Phase 2: Behavioral evaluation
    log("\n[Phase 2] Behavioral evaluation...")
    # Load a tokenizer reference for logging (lightweight)
    from transformers import AutoTokenizer
    tokenizer_ref = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    behav_results = phase_behavioral_eval(test_examples, model_dims, tokenizer_ref)
    lengths = behav_results["lengths"]

    # Compute Cohen's d for all pairs
    d_concise_step = cohen_d(lengths.get("concise", []), lengths.get("step", []))
    d_concise_code = cohen_d(lengths.get("concise", []), lengths.get("code", []))
    d_code_step = cohen_d(lengths.get("code", []), lengths.get("step", []))

    k940_pass = bool(d_concise_step > 0.3)

    log(f"\n[K940] Cohen's d results:")
    log(f"  concise vs step: d = {d_concise_step:.3f}")
    log(f"  concise vs code: d = {d_concise_code:.3f}")
    log(f"  code vs step:    d = {d_code_step:.3f}")
    log(f"  [K940] {'PASS' if k940_pass else 'FAIL'}: d(concise, step) = {d_concise_step:.3f} > 0.3")

    mean_lens = {p: round(float(np.mean(v)), 2) if v else 0.0 for p, v in lengths.items()}
    std_lens = {p: round(float(np.std(v)), 2) if v else 0.0 for p, v in lengths.items()}
    log(f"\n  Mean response lengths: {mean_lens}")
    log(f"  Std response lengths:  {std_lens}")

    # Phase 3: Composition test
    log("\n[Phase 3] Composition test (domain + step)...")
    comp_results = phase_composition_test(test_examples, model_dims, base_acc, v4_acc)

    # Final results
    total_time = time.time() - t_start

    results = {
        "experiment": "m2p_per_user_poc",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "n_train_persona": N_TRAIN_PERSONA,
            "n_test": N_TEST,
            "m2p_steps": M2P_STEPS,
            "lora_rank": LORA_RANK,
            "d_m2p": D_M2P,
            "seed": SEED,
        },
        "train_results": train_results,
        "lengths_mean": mean_lens,
        "lengths_std": std_lens,
        "first_outputs": behav_results.get("first_outputs", {}),
        "d_concise_step": round(d_concise_step, 3),
        "d_concise_code": round(d_concise_code, 3),
        "d_code_step": round(d_code_step, 3),
        "k940_d_concise_step": round(d_concise_step, 3),
        "k940_pass": k940_pass,
        "k940_status": "PASS" if k940_pass else "FAIL",
        "composition": comp_results,
        "k941_pass": comp_results.get("k941_pass", False),
        "k941_status": comp_results.get("k941_status", "unknown"),
        "v4_acc": v4_acc,
        "base_acc": base_acc,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n{'=' * 70}")
    log(f"FINAL RESULTS:")
    log(f"  K940: {results['k940_status']} — Cohen's d(concise, step) = {d_concise_step:.3f}")
    log(f"  K941: {results['k941_status']} — quality loss = {comp_results.get('quality_loss_vs_v4', '?')}")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"  Results saved: {RESULTS_FILE}")
    log(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
