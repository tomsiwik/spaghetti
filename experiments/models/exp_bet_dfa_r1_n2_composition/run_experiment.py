#!/usr/bin/env python3
"""exp_bet_dfa_r1_n2_composition — disjoint-frame B-projection cuts N=2 behavioral interference.

BET dfa-init R1. Frozen base gemma-4-e4b-it-4bit + r=6 q_proj python(code) and math adapters.
F#827 interference: python alongside math drags MATH (GSM8K) accuracy down (−14pp python→math).

Mechanism (MATH.md): each adapter's delta-output δ_i = s·(h@A_i)@B_i lives in rowspace(B_i) ⊆ R^d.
Build, PER LAYER, a frozen disjoint-block orthonormal frame by joint QR of stacked B-transposes:
    Q,R = QR([B_py^T | B_math^T])  →  Q_py = Q[:,0:6], Q_math = Q[:,6:12],  Q_py^T Q_math = 0.
Project each delta-output onto its own frame: δ'_i = Q_i (Q_i^T δ_i). The two projected outputs are
output-orthogonal by construction (Theorem 1) → zero cross-overlap, every layer.

Composition is Σ_i P_i (B_i A_i), never (ΣB)(ΣA). LORA_SCALE = 6.0 ≤ 8. Frames are frozen, derived
only from the adapters' own B, built lazily at the TRUE delta width discovered from the model.

Metric: GSM8K exact-match accuracy (the math skill F#827 says gets interfered). Conditions:
  A base        : no adapters.
  B math-solo   : math adapter only           (ceiling for the math skill).
  C naive sum   : python + math, both UNprojected (F#827 interference baseline).
  D dfa sum     : python + math, each projected onto its disjoint frame.
Plus solo-preservation probe on CODE (HumanEval): code-solo unprojected vs code-solo projected.

KILL K2313 (frame destroys skill): KILL if projected code-solo drops > 5pp vs unprojected code-solo.
KILL K2314 (interference uncut):  KILL if residual drag acc(B)-acc(D) > 7pp OR recovered < 0.5*gap.
Gate (supported): both kills clear AND gap = acc(B)-acc(C) >= 0.10 (interference actually present).

NO MOCKS. Real model, real adapters, real GSM8K + HumanEval execution. is_smoke=False. mlx-lm 0.31.2.
Wrapper attaches via subclass nn.Module + setattr (never __call__ override on instance — F#831).
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
# Named python/math/medical adapters (F#827 pairing). Same q_proj r=6 LoRA layout as cdma.
ADAPTER_DIR = EXP_DIR.parent.parent.parent / "data" / "adapters"
ADAPTER_PY = ADAPTER_DIR / "python" / "adapters.safetensors"     # "python" = code skill
ADAPTER_MATH = ADAPTER_DIR / "math" / "adapters.safetensors"

LORA_SCALE = 6.0          # <= 8 guard OK (F#627 recipe)
LORA_RANK = 6
N_GSM8K = int(os.environ.get("N_GSM8K", "200"))         # math skill (interfered) — >=200
N_HUMANEVAL = int(os.environ.get("N_HUMANEVAL", "100")) # code skill (solo-preservation probe)
MAX_NEW_TOKENS_MATH = 512
MAX_NEW_TOKENS_CODE = 768
SEED = 42
N_LAYERS_EXPECTED = 42

# Kill thresholds (fractional)
K2313_SOLO_DROP = 0.05    # projected code-solo must not drop > 5pp vs unprojected code-solo
K2314_RESIDUAL = 0.07     # residual drag B-D must be <= 7pp
K2314_RECOVER_FRAC = 0.50 # recovered must be >= 50% of the interference gap
GAP_MIN = 0.10            # interference must actually be present to make a supported claim


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Disjoint-frame B-projection. Per layer: QR([B_i^T | B_j^T]) -> disjoint blocks.
# Frames are FROZEN (derived only from the adapters' own B), built lazily at real d.
# ----------------------------------------------------------------------------

def build_disjoint_frames(b_i, b_j, r):
    """b_i,b_j: (r, d). Return Q_i,Q_j: (d, r) with Q_i^T Q_i=I, Q_i^T Q_j=0 (verified).

    M = [b_i^T | b_j^T] in R^{d x 2r}; QR(M) -> Q in R^{d x 2r} orthonormal columns.
    First r columns span (a superset of) rowspace(b_i) since Gram-Schmidt processes b_i first;
    next r columns are math's residual directions, orthogonal to block i by construction.
    """
    d = b_i.shape[1]
    M = mx.concatenate([b_i.T, b_j.T], axis=1).astype(mx.float64, stream=mx.cpu)  # (d, 2r)
    with mx.stream(mx.cpu):
        Q, R = mx.linalg.qr(M)            # Q: (d, 2r) (QR is CPU-only in MLX)
        Q_i = Q[:, :r]
        Q_j = Q[:, r:2 * r]
        cross = float(mx.max(mx.abs(Q_i.T @ Q_j)).item())
        gram_i = float(mx.max(mx.abs(Q_i.T @ Q_i - mx.eye(r, dtype=mx.float64))).item())
    assert cross < 1e-8, f"frames not disjoint: max|Q_i^T Q_j|={cross}"
    assert gram_i < 1e-8, f"frame not orthonormal: {gram_i}"
    return Q_i.astype(mx.float32, stream=mx.cpu), Q_j.astype(mx.float32, stream=mx.cpu), cross


# ----------------------------------------------------------------------------
# Composed q_proj wrapper: base + P_py·(py delta) + P_math·(math delta)
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class ComposedQProj(nn.Module):
    """y = linear(x) + use_py·P_py(s·(x@a_py)@b_py) + use_math·P_math(s·(x@a_math)@b_math).

    P_i = Q_i Q_i^T applied via two matmuls: δ @ Q_i then @ Q_i^T (kept as the (d,r) frame).
    When project=False, P_i = I (the naive-sum / unprojected condition).
    """
    def __init__(self, base_linear, a_py, b_py, a_math, b_math, scale,
                 use_py, use_math, q_py, q_math):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear
        self.a_py, self.b_py = a_py, b_py
        self.a_math, self.b_math = a_math, b_math
        self.scale = scale
        self.use_py = use_py
        self.use_math = use_math
        self.q_py = q_py                     # (d, r) frozen frame or None (identity)
        self.q_math = q_math                 # (d, r) frozen frame or None (identity)
        self.linear.freeze()

    def _proj(self, delta, q):
        if q is None:
            return delta
        # δ' = (δ @ Q) @ Q^T  ==  Q Q^T δ  (orthogonal projector onto span(Q))
        return (delta @ q) @ q.T

    def __call__(self, x):
        y = self.linear(x)
        if self.use_py:
            dpy = self.scale * ((x @ self.a_py) @ self.b_py)
            dpy = self._proj(dpy, self.q_py)
            y = y + dpy.astype(x.dtype)
        if self.use_math:
            dm = self.scale * ((x @ self.a_math) @ self.b_math)
            dm = self._proj(dm, self.q_math)
            y = y + dm.astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_composed(model, py_ad, math_ad, scale, use_py, use_math, project):
    """Wrap q_proj on every layer. project=True builds frozen disjoint frames per layer."""
    lm = get_lm(model)
    count = 0
    max_cross = 0.0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in py_ad or bk not in py_ad:
            continue
        base_linear = layer.self_attn.q_proj
        a_py = py_ad[ak].astype(mx.float32)
        b_py = py_ad[bk].astype(mx.float32)
        a_math = math_ad[ak].astype(mx.float32)
        b_math = math_ad[bk].astype(mx.float32)

        q_py = q_math = None
        if project:
            q_py, q_math, cross = build_disjoint_frames(b_py, b_math, LORA_RANK)
            max_cross = max(max_cross, cross)

        wrapper = ComposedQProj(base_linear, a_py, b_py, a_math, b_math, scale,
                                use_py, use_math, q_py, q_math)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} ComposedQProj (use_py={use_py} use_math={use_math} "
        f"project={project} max|Q_py^T Q_math|={max_cross:.2e})")
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    return max_cross


# ----------------------------------------------------------------------------
# Generation (greedy, no-thinking harness — matches training, per memory)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt, max_new):
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
# GSM8K (math skill — the one F#827 says gets interfered)
# ----------------------------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    items = [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(len(ds))]
    log(f"  Loaded {len(items)} GSM8K problems")
    return items


def gsm8k_gt(answer):
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def gsm8k_pred(text):
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def eval_gsm8k(model, tokenizer, items):
    correct, details = 0, []
    for it in items:
        prompt = format_chat(
            tokenizer,
            "Solve this math problem step by step. End your answer with '#### <number>'.\n\n"
            + it["question"],
        )
        text, ntok = generate(model, tokenizer, prompt, MAX_NEW_TOKENS_MATH)
        gt = gsm8k_gt(it["answer"])
        pred = gsm8k_pred(text)
        ok = (gt is not None and pred is not None and pred == gt)
        correct += int(ok)
        details.append({"gt": gt, "pred": pred, "ok": ok, "ntok": ntok})
    acc = correct / len(items)
    log(f"    GSM8K acc = {acc:.4f} ({correct}/{len(items)})")
    return acc, details


# ----------------------------------------------------------------------------
# HumanEval (code skill — solo-preservation probe for K2313)
# ----------------------------------------------------------------------------

def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        probs.append({"task_id": it["task_id"], "prompt": it["prompt"],
                      "test": it["test"], "entry_point": it["entry_point"]})
    log(f"  Loaded {len(probs)} HumanEval problems")
    return probs


def humaneval_prompt(p):
    return ("Complete this Python function. Return the full function in a "
            "```python code block.\n\n```python\n" + p["prompt"] + "\n```")


def extract_code(text, prompt_code, entry_point):
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
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
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
        text, ntok = generate(model, tokenizer, prompt, MAX_NEW_TOKENS_CODE)
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

def with_model(fn):
    model, tok = load(MODEL_ID)
    try:
        return fn(model, tok)
    finally:
        del model, tok
        gc.collect(); mx.clear_cache()


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_bet_dfa_r1_n2_composition — DFA disjoint-frame B-projection, N=2")
    log(f"Base: {MODEL_ID}")
    log(f"python(code) adapter: {ADAPTER_PY}")
    log(f"math adapter:         {ADAPTER_MATH}")
    log(f"N_GSM8K={N_GSM8K} N_HUMANEVAL={N_HUMANEVAL} scale={LORA_SCALE} rank={LORA_RANK}")
    log("=" * 72)
    assert ADAPTER_PY.exists(), f"missing {ADAPTER_PY}"
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    log_mem("start")

    py_ad = mx.load(str(ADAPTER_PY))
    math_ad = mx.load(str(ADAPTER_MATH))

    gsm = load_gsm8k(N_GSM8K)
    he = load_humaneval(N_HUMANEVAL)

    # ---- MATH skill (GSM8K): the interference axis (F#827 python->math) ----
    # A: base
    def _a(model, tok):
        log("\n=== COND A: base (GSM8K) ===")
        return eval_gsm8k(model, tok, gsm)
    A_acc, A_det = with_model(_a)

    # B: math-solo (ceiling)
    def _b(model, tok):
        log("\n=== COND B: math-solo (GSM8K, ceiling) ===")
        attach_composed(model, py_ad, math_ad, LORA_SCALE, use_py=False, use_math=True, project=False)
        return eval_gsm8k(model, tok, gsm)
    B_acc, B_det = with_model(_b)

    # C: naive python+math sum (interference baseline)
    def _c(model, tok):
        log("\n=== COND C: naive python+math sum (GSM8K, interference) ===")
        attach_composed(model, py_ad, math_ad, LORA_SCALE, use_py=True, use_math=True, project=False)
        return eval_gsm8k(model, tok, gsm)
    C_acc, C_det = with_model(_c)

    # D: DFA python+math sum (disjoint-frame projection)
    dfa_cross = {"val": 0.0}
    def _d(model, tok):
        log("\n=== COND D: DFA python+math sum (GSM8K, disjoint frames) ===")
        dfa_cross["val"] = attach_composed(model, py_ad, math_ad, LORA_SCALE,
                                           use_py=True, use_math=True, project=True)
        return eval_gsm8k(model, tok, gsm)
    D_acc, D_det = with_model(_d)

    # ---- CODE skill (HumanEval): solo-preservation probe for K2313 ----
    # E: code-solo unprojected
    def _e(model, tok):
        log("\n=== COND E: code-solo unprojected (HumanEval) ===")
        attach_composed(model, py_ad, math_ad, LORA_SCALE, use_py=True, use_math=False, project=False)
        return eval_humaneval(model, tok, he)
    E_acc, E_det = with_model(_e)

    # F: code-solo projected (only python frame applied; math slot off)
    def _f(model, tok):
        log("\n=== COND F: code-solo PROJECTED (HumanEval) ===")
        attach_composed(model, py_ad, math_ad, LORA_SCALE, use_py=True, use_math=False, project=True)
        return eval_humaneval(model, tok, he)
    F_acc, F_det = with_model(_f)

    # ---- Kill criteria ----
    gap = B_acc - C_acc                  # interference drag present in naive sum
    recovered = D_acc - C_acc
    residual_drag = B_acc - D_acc        # how far DFA still sits below ceiling
    recovery_frac = (recovered / gap) if gap > 1e-9 else float("nan")
    solo_drop = E_acc - F_acc            # code-solo projection cost

    k2313_kill = solo_drop > K2313_SOLO_DROP                       # frame destroys skill
    k2314_kill = (residual_drag > K2314_RESIDUAL) or (recovered < K2314_RECOVER_FRAC * gap)
    interference_present = gap >= GAP_MIN

    if not interference_present:
        verdict = "provisional"          # nothing to cut on this pair — not a refutation
        all_pass = False
    elif k2313_kill or k2314_kill:
        verdict = "killed"
        all_pass = False
    else:
        verdict = "supported"
        all_pass = True

    results = {
        "experiment_id": "exp_bet_dfa_r1_n2_composition",
        "config": {
            "base_model": MODEL_ID,
            "adapter_python": str(ADAPTER_PY),
            "adapter_math": str(ADAPTER_MATH),
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_gsm8k": N_GSM8K,
            "n_humaneval": N_HUMANEVAL,
            "max_new_tokens_math": MAX_NEW_TOKENS_MATH,
            "max_new_tokens_code": MAX_NEW_TOKENS_CODE,
            "no_thinking_harness": True,
            "max_disjoint_cross_QpyT_Qmath": dfa_cross["val"],
            "mlx_lm": "0.31.2",
        },
        "math_gsm8k": {
            "A_base": A_acc,
            "B_math_solo": B_acc,
            "C_naive_sum": C_acc,
            "D_dfa_sum": D_acc,
        },
        "code_humaneval": {
            "E_code_solo_unprojected": E_acc,
            "F_code_solo_projected": F_acc,
        },
        "interference_gap_B_minus_C": gap,
        "recovered_D_minus_C": recovered,
        "residual_drag_B_minus_D": residual_drag,
        "recovery_fraction": recovery_frac,
        "solo_drop_E_minus_F": solo_drop,
        "kill_criteria": {
            "2313": {
                "text": "projected code-solo drops > 5pp vs unprojected code-solo",
                "solo_drop": solo_drop, "threshold": K2313_SOLO_DROP,
                "result": "fail" if k2313_kill else "pass",
            },
            "2314": {
                "text": "interference not cut >=50% (residual drag B-D > 7pp OR recovered < 0.5*gap)",
                "residual_drag": residual_drag, "recovered": recovered,
                "recovery_fraction": recovery_frac,
                "thresholds": {"residual_drag": K2314_RESIDUAL, "recover_frac": K2314_RECOVER_FRAC},
                "result": "fail" if k2314_kill else "pass",
            },
        },
        "interference_present": interference_present,
        "gap_min_required": GAP_MIN,
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "details": {"A": A_det, "B": B_det, "C": C_det, "D": D_det, "E": E_det, "F": F_det},
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"GSM8K  A(base)={A_acc:.3f}  B(math-solo)={B_acc:.3f}  "
        f"C(naive)={C_acc:.3f}  D(dfa)={D_acc:.3f}")
    log(f"interference gap B-C={gap:.3f}  recovered D-C={recovered:.3f}  "
        f"frac={recovery_frac:.2f}  residual B-D={residual_drag:.3f}")
    log(f"HumanEval code-solo  E(unproj)={E_acc:.3f}  F(proj)={F_acc:.3f}  drop={solo_drop:.3f}")
    log(f"K2313 (solo destroyed): {'KILL' if k2313_kill else 'pass'}  "
        f"K2314 (uncut): {'KILL' if k2314_kill else 'pass'}  "
        f"interference_present={interference_present}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
