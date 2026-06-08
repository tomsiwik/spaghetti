#!/usr/bin/env python3
"""exp_spark_cdma_delta_spreading — fixed orthogonal rotation of the off-domain LoRA delta-output.

Frozen base gemma-4-e4b-it-4bit + r=6 q_proj code(HumanEval) and math(GSM8K) adapters (F#627).
HumanEval pass@1 (n=50), real unit-test execution, 4 conditions:

  A base    : no adapters.
  B code    : code adapter only (the ceiling).
  C sum     : code + math, both unrotated (interference baseline).
  D spread  : code unrotated + math delta-OUTPUT rotated by fixed orthogonal P (post-B, activation space).

q_proj wrapper applies, per layer:
    y = W h + s·B_code A_code h + R · (s·B_math A_math h)
with R = I (conditions B/C have their own slot enabled/disabled) or R = Pᵀ (condition D, math slot only).
Composition is Σ_i B_i A_i (two independent deltas), never (ΣB)(ΣA). LORA_SCALE=6.0 ≤ 8.

P is a fixed seeded random orthogonal matrix (QR of Gaussian, seed=1337), PᵀP=I verified, never learned.
Wrapper attaches via subclass nn.Module + setattr — never __call__ override on instance (F#831).

KILL K2295 (target, behavioral): KILL if pass@1(D) < pass@1(C) + 8pp OR pass@1(D) < pass@1(B) - 6pp.

NO MOCKS. Real model, real adapters, real benchmark execution. is_smoke=False. mlx-lm == 0.31.2.
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_DIR = EXP_DIR.parent / "exp_composition_residual_analysis"
ADAPTER_CODE = ADAPTER_DIR / "adapter_code.safetensors"
ADAPTER_MATH = ADAPTER_DIR / "adapter_math.safetensors"

LORA_SCALE = 6.0          # <= 8 guard OK (F#627 recipe)
LORA_RANK = 6
N_HUMANEVAL = 50
MAX_NEW_TOKENS = 1024     # thinking-mode headroom
SEED = 42
P_SEED = 1337             # fixed seed for the orthogonal rotation
D_OUT = 2048              # q_proj output dim
N_LAYERS_EXPECTED = 42

# Kill thresholds (pp, fractional)
K_RECOVER = 0.08          # D must clear C + 8pp
K_CEILING_GAP = 0.06      # D must be within 6pp of B


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Fixed orthogonal rotation P (Haar-ish via QR of Gaussian, seeded, NEVER learned)
# ----------------------------------------------------------------------------

def make_orthogonal(d, seed):
    """Return P orthogonal (d x d), PᵀP ≈ I verified.

    The wrapper applies the rotation to row-vector deltas as `dm @ P`, which realizes the
    column-vector operation `Pᵀ δ` since (Pᵀδ)ᵀ = δᵀ P. P and Pᵀ are both orthogonal; we use P
    in the row-multiply so the activation-space operation is exactly Pᵀδ as in MATH.md §3.
    """
    mx.random.seed(seed)
    # QR in float64 on CPU for tight orthogonality, then cast P to float32 for matmuls.
    g = mx.random.normal((d, d), stream=mx.cpu).astype(mx.float64, stream=mx.cpu)
    with mx.stream(mx.cpu):
        q, r = mx.linalg.qr(g)   # QR is CPU-only in MLX
        # Sign-fix columns so Q is deterministic given seed (make diag(R) > 0).
        sign = mx.sign(mx.diagonal(r))
        sign = mx.where(sign == 0, mx.ones_like(sign), sign)
        P64 = q * sign[None, :]
        err = float(mx.max(mx.abs(P64.T @ P64 - mx.eye(d, dtype=mx.float64))).item())
    log(f"  orthogonality ‖PᵀP - I‖_inf = {err:.2e}  (seed={seed}, d={d}, float64 QR)")
    assert err < 1e-8, f"P not orthogonal: err={err}"
    P = P64.astype(mx.float32, stream=mx.cpu)
    mx.eval(P)
    return P


# ----------------------------------------------------------------------------
# Composed q_proj wrapper: base + code delta + R·(math delta)
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class ComposedQProj(nn.Module):
    """y = linear(x) + use_code·s·(x@Ac)@Bc + use_math·s·R((x@Am)@Bm).

    R is identity (None) or the fixed Pᵀ rotation applied to the math delta OUTPUT.
    """
    def __init__(self, base_linear, ac, bc, am, bm, scale,
                 use_code, use_math, rot):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear
        self.ac, self.bc = ac, bc            # code: (in,r),(r,out)
        self.am, self.bm = am, bm            # math: (in,r),(r,out)
        self.scale = scale
        self.use_code = use_code
        self.use_math = use_math
        self.rot = rot                       # None or Pᵀ (out,out)
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        if self.use_code:
            dc = (x @ self.ac) @ self.bc                  # (..., out)  = s? applied below
            y = y + (self.scale * dc).astype(x.dtype)
        if self.use_math:
            dm = (x @ self.am) @ self.bm                  # math delta output (..., out)
            dm = self.scale * dm
            if self.rot is not None:
                dm = dm @ self.rot                        # rotate OUTPUT: row-vector δ @ P realizes Pᵀδ
            y = y + dm.astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_composed(model, code_ad, math_ad, scale, use_code, use_math, rot):
    """Wrap q_proj on every layer with ComposedQProj. Returns count wrapped."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in code_ad or bk not in code_ad:
            continue
        base_linear = layer.self_attn.q_proj
        ac = code_ad[ak].astype(mx.float32)
        bc = code_ad[bk].astype(mx.float32)
        am = math_ad[ak].astype(mx.float32)
        bm = math_ad[bk].astype(mx.float32)
        wrapper = ComposedQProj(base_linear, ac, bc, am, bm, scale,
                                use_code, use_math, rot)
        setattr(layer.self_attn, "q_proj", wrapper)   # canonical: setattr
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} ComposedQProj (use_code={use_code} use_math={use_math} rot={'P^T' if rot is not None else 'I'})")
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    return model


# ----------------------------------------------------------------------------
# Generation (greedy)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


# ----------------------------------------------------------------------------
# HumanEval data + scoring (real unit-test execution)
# ----------------------------------------------------------------------------

def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        probs.append({
            "task_id": it["task_id"], "prompt": it["prompt"],
            "test": it["test"], "entry_point": it["entry_point"],
        })
    log(f"  Loaded {len(probs)} HumanEval problems")
    return probs


def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def humaneval_prompt(p):
    return (
        "Complete this Python function. Return the full function in a "
        "```python code block.\n\n```python\n" + p["prompt"] + "\n```"
    )


def extract_code(text, prompt_code, entry_point):
    text = strip_thinking(text)
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    for blk in blocks:
        if f"def {entry_point}" in blk:
            return blk.strip()
    if blocks:
        return prompt_code + "\n" + blocks[0].strip()
    if f"def {entry_point}" in text:
        return text
    return prompt_code + "\n" + text


def run_humaneval_test(code, test, entry_point, timeout=12):
    full = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def eval_humaneval(model, tokenizer, problems):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        text, ntok = generate(model, tokenizer, prompt)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    HumanEval pass@1 = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------------

def cond_base(problems):
    log("\n=== COND A: base (no adapters) ===")
    model, tok = load(MODEL_ID)
    acc, det = eval_humaneval(model, tok, problems)
    log_mem("A-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"pass@1": acc, "details": det}


def cond_composed(problems, code_ad, math_ad, label, use_code, use_math, rot):
    log(f"\n=== COND {label}: use_code={use_code} use_math={use_math} rot={'P^T' if rot is not None else 'I'} ===")
    model, tok = load(MODEL_ID)
    attach_composed(model, code_ad, math_ad, LORA_SCALE, use_code, use_math, rot)
    gc.collect(); mx.clear_cache()
    acc, det = eval_humaneval(model, tok, problems)
    log_mem(f"{label}-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"pass@1": acc, "details": det}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_cdma_delta_spreading")
    log(f"Base: {MODEL_ID}")
    log(f"Code adapter: {ADAPTER_CODE}")
    log(f"Math adapter: {ADAPTER_MATH}")
    log(f"n_humaneval={N_HUMANEVAL} scale={LORA_SCALE} rank={LORA_RANK} P_seed={P_SEED}")
    log("=" * 72)
    assert ADAPTER_CODE.exists(), f"missing {ADAPTER_CODE}"
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    log_mem("start")

    log("\n=== Build fixed orthogonal P (applied as P^T on the math delta output) ===")
    P = make_orthogonal(D_OUT, P_SEED)

    log("\n=== Load data + adapters ===")
    problems = load_humaneval(N_HUMANEVAL)
    code_ad = mx.load(str(ADAPTER_CODE))
    math_ad = mx.load(str(ADAPTER_MATH))

    # Condition A: base
    A = cond_base(problems)
    # Condition B: code-solo (ceiling)
    B = cond_composed(problems, code_ad, math_ad, "B", use_code=True, use_math=False, rot=None)
    # Condition C: naive sum (code + math, both unrotated)
    C = cond_composed(problems, code_ad, math_ad, "C", use_code=True, use_math=True, rot=None)
    # Condition D: delta-spread (code unrotated + math delta-output rotated by P^T)
    D = cond_composed(problems, code_ad, math_ad, "D", use_code=True, use_math=True, rot=P)

    pa, pb, pc, pd = A["pass@1"], B["pass@1"], C["pass@1"], D["pass@1"]

    # ---- Kill criterion K2295 (target, behavioral) ----
    clause_recover_fail = pd < pc + K_RECOVER          # D fails to recover >=8pp over C
    clause_ceiling_fail = pd < pb - K_CEILING_GAP      # D not within 6pp of ceiling B
    killed = clause_recover_fail or clause_ceiling_fail
    verdict = "KILLED" if killed else "SUPPORTED"
    all_pass = not killed

    interference_gap = pb - pc
    recovered = pd - pc
    recovery_frac = (recovered / interference_gap) if interference_gap > 1e-9 else float("nan")

    results = {
        "experiment_id": "exp_spark_cdma_delta_spreading",
        "config": {
            "base_model": MODEL_ID,
            "adapter_code": str(ADAPTER_CODE),
            "adapter_math": str(ADAPTER_MATH),
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_humaneval": N_HUMANEVAL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "p_seed": P_SEED,
            "d_out": D_OUT,
            "k_recover_pp": K_RECOVER,
            "k_ceiling_gap_pp": K_CEILING_GAP,
            "mlx_lm": "0.31.2",
        },
        "conditions": {
            "A_base": {"pass@1": pa, "details": A["details"]},
            "B_code_solo": {"pass@1": pb, "details": B["details"]},
            "C_naive_sum": {"pass@1": pc, "details": C["details"]},
            "D_delta_spread": {"pass@1": pd, "details": D["details"]},
        },
        "pass_at_1": {"A": pa, "B": pb, "C": pc, "D": pd},
        "interference_gap_B_minus_C": interference_gap,
        "recovered_D_minus_C": recovered,
        "recovery_fraction": recovery_frac,
        "kill_criteria": {
            "2295": {
                "text": "pass@1(D) < pass@1(C)+8pp OR pass@1(D) < pass@1(B)-6pp",
                "type": "target_behavioral",
                "clause_recover_fail": bool(clause_recover_fail),
                "clause_ceiling_fail": bool(clause_ceiling_fail),
                "result": "fail" if killed else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"pass@1  A(base)={pa:.3f}  B(code-solo)={pb:.3f}  C(naive-sum)={pc:.3f}  D(delta-spread)={pd:.3f}")
    log(f"interference gap B-C = {interference_gap:.3f}  recovered D-C = {recovered:.3f}  frac = {recovery_frac:.2f}")
    log(f"K2295: recover_fail={clause_recover_fail} ceiling_fail={clause_ceiling_fail}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
