#!/usr/bin/env python3
"""exp_spark_velocity_mask_interference — split a thinking adapter by per-weight LEARNING VELOCITY.

Frozen base gemma-4-e4b-it-4bit. Two adapters on DISJOINT projections:
  - thinking (OpenThoughts-universal): self_attn.v_proj + o_proj, rank 8, scale 1.0, with a saved
    20-step training TRAJECTORY (checkpoints step 50..1000).
  - math (domain): self_attn.q_proj, rank 6, scale 6.0 (endpoint).

Velocity mask (built from the trajectory, NOT the endpoint alone):
  For each projection/layer, effective delta dW_t = A_t @ B_t. The early-velocity CORE is
      M = (|dW_200| >= 0.80*|dW_1000|) AND (sign(dW_200)==sign(dW_1000)).
  dW_core = M ⊙ dW_1000   (the compose-safe core);  dW_late = (1-M) ⊙ dW_1000 (the interference residual).

GSM8K pass@1 (n=50), greedy, real numeric-answer extraction, 4 conditions:
  A math-solo            : q_proj math only (the domain ceiling).
  B math + full-thinking : math + step-1000 thinking (low-rank, the interference candidate).
  C math + early-core    : math + dense dW_core  (the claim).
  D math + late-residual  : math + dense dW_late  (control: damage should live here).

Composition is Σ_i (B_i A_i) on disjoint projections — never (ΣB)(ΣA). math scale 6.0 ≤ 8 guard OK.
Wrappers attach via subclass nn.Module + setattr — never __call__ override on instance (F#831).

KILL 2298 (pre-registered, behavioral): KILL if acc(C)-acc(B) < +6pp OR acc(C) <= acc(D).

NO MOCKS. Real model, real adapters, real GSM8K execution. is_smoke=False. mlx-lm == 0.31.2.
"""

import gc
import json
import os
import re
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

REPO = EXP_DIR.parent.parent.parent          # .../llm
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
THINK_DIR = REPO / "data" / "adapters" / "thinking-openthoughts-universal-v0"
THINK_EARLY = THINK_DIR / "0000200_adapters.safetensors"
THINK_FINAL = THINK_DIR / "0001000_adapters.safetensors"
MATH_AD = REPO / "data" / "adapters" / "math" / "adapters.safetensors"

THINK_SCALE = 1.0         # thinking adapter's own trained scale
MATH_SCALE = 6.0          # math adapter's trained scale (<= 8 guard OK)
THINK_PROJS = ("v_proj", "o_proj")
MATH_PROJ = "q_proj"
VEL_THRESH = 0.80         # >= 80% of final magnitude by step 200
N_GSM8K = 50
MAX_NEW_TOKENS = 1024     # thinking-mode headroom
SEED = 42

K_RECOVER = 0.06          # acc(C)-acc(B) must be >= +6pp
# clause 2: acc(C) > acc(D) (late residual is the worse half)


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    try:
        peak = mx.get_peak_memory() / 1024**3
        log(f"  [mem {label}] peak={peak:.2f}GB")
    except Exception:
        pass


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


# ----------------------------------------------------------------------------
# Velocity mask: build dense masked deltas from the TRAJECTORY (step 200 vs 1000)
# ----------------------------------------------------------------------------

def layer_keys(weights, li, proj):
    ak = f"language_model.model.layers.{li}.self_attn.{proj}.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.{proj}.lora_b"
    return (ak, bk) if ak in weights and bk in weights else (None, None)


def build_velocity_deltas(w_early, w_final):
    """Return {(li,proj): (dW_core, dW_late)} dense bf16 deltas + global core fraction.

    dW = A@B (in,out). M = (|dW_200|>=0.80*|dW_1000|) & (sign match). core=M⊙dW_1000.
    """
    layers = sorted({
        int(re.search(r"layers\.(\d+)\.", k).group(1))
        for k in w_final if "lora_a" in k
    })
    deltas = {}
    tot_core = 0
    tot = 0
    for li in layers:
        for proj in THINK_PROJS:
            ak, bk = layer_keys(w_final, li, proj)
            if ak is None:
                continue
            A2 = w_early[ak].astype(mx.float32)
            B2 = w_early[bk].astype(mx.float32)
            A1 = w_final[ak].astype(mx.float32)
            B1 = w_final[bk].astype(mx.float32)
            dW2 = A2 @ B2          # (in,out)
            dW1 = A1 @ B1
            m1 = mx.abs(dW1)
            m2 = mx.abs(dW2)
            sign_match = (mx.sign(dW1) == mx.sign(dW2))
            core_mask = (m2 >= VEL_THRESH * m1) & sign_match     # bool (in,out)
            core_mask_f = core_mask.astype(mx.float32)
            dW_core = (core_mask_f * dW1).astype(mx.bfloat16)
            dW_late = ((1.0 - core_mask_f) * dW1).astype(mx.bfloat16)
            deltas[(li, proj)] = (dW_core, dW_late)
            tot_core += int(core_mask.sum().item())
            tot += core_mask.size
    mx.eval([v for pair in deltas.values() for v in pair])
    core_frac = tot_core / tot if tot else float("nan")
    log(f"  built velocity deltas: {len(deltas)} (layer,proj) entries, global core_frac={core_frac:.4f}")
    return deltas, core_frac


# ----------------------------------------------------------------------------
# Wrappers (subclass nn.Module + setattr, never __call__ on instance — F#831)
# ----------------------------------------------------------------------------

class LoRAProj(nn.Module):
    """y = linear(x) + scale * (x@A)@B   — low-rank additive delta on one projection."""
    def __init__(self, base_linear, a, b, scale):
        super().__init__()
        self.linear = base_linear
        self.a = a
        self.b = b
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        d = (x @ self.a) @ self.b
        return y + (self.scale * d).astype(x.dtype)


class DenseDeltaProj(nn.Module):
    """y = linear(x) + scale * (x @ dW)   — dense additive delta (masked thinking core/late)."""
    def __init__(self, base_linear, dW, scale):
        super().__init__()
        self.linear = base_linear
        self.dW = dW                 # (in,out) bf16
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        d = x @ self.dW.astype(x.dtype)
        return y + (self.scale * d).astype(x.dtype)


def attach_math(model, math_ad):
    """Wrap q_proj with the low-rank math delta on every layer that has it. Returns count."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak, bk = layer_keys(math_ad, li, MATH_PROJ)
        if ak is None:
            continue
        a = math_ad[ak].astype(mx.float32)
        b = math_ad[bk].astype(mx.float32)
        wrapper = LoRAProj(layer.self_attn.q_proj, a, b, MATH_SCALE)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    return count


def attach_thinking_lowrank(model, think_ad):
    """Wrap v_proj/o_proj with low-rank step-1000 thinking delta (condition B)."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        for proj in THINK_PROJS:
            ak, bk = layer_keys(think_ad, li, proj)
            if ak is None:
                continue
            a = think_ad[ak].astype(mx.float32)
            b = think_ad[bk].astype(mx.float32)
            base = getattr(layer.self_attn, proj)
            setattr(layer.self_attn, proj, LoRAProj(base, a, b, THINK_SCALE))
            count += 1
    return count


def attach_thinking_dense(model, deltas, which):
    """Wrap v_proj/o_proj with dense masked thinking delta. which in {'core','late'}."""
    idx = 0 if which == "core" else 1
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        for proj in THINK_PROJS:
            if (li, proj) not in deltas:
                continue
            dW = deltas[(li, proj)][idx]
            base = getattr(layer.self_attn, proj)
            setattr(layer.self_attn, proj, DenseDeltaProj(base, dW, THINK_SCALE))
            count += 1
    return count


# ----------------------------------------------------------------------------
# Generation (greedy) + GSM8K
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


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        probs.append({"question": it["question"], "gold": gold})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_answer(text):
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace("$", "").replace(",", "")
    nums = _NUM.findall(text)
    if nums:
        return nums[-1].replace("$", "").replace(",", "")
    return None


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def gsm8k_prompt(q):
    return (
        f"{q}\n\nSolve step by step, then give the final numeric answer on its own "
        "line in the form '#### <number>'."
    )


def eval_gsm8k(model, tokenizer, problems):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p["question"]))
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_answer(text)
        ok = num_eq(pred, p["gold"])
        passed += int(ok)
        details.append({"gold": p["gold"], "pred": pred, "ok": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    GSM8K acc = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# Conditions (each loads a fresh model, attaches, evals, frees)
# ----------------------------------------------------------------------------

def run_condition(label, problems, math_ad, think_setup):
    log(f"\n=== COND {label} ===")
    model, tok = load(MODEL_ID)
    nm = attach_math(model, math_ad)
    nt = think_setup(model) if think_setup else 0
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()
    log(f"  attached math={nm} thinking={nt}")
    acc, det = eval_gsm8k(model, tok, problems)
    log_mem(f"{label}-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"acc": acc, "n_math": nm, "n_think": nt, "details": det}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_velocity_mask_interference")
    log(f"Base: {MODEL_ID}")
    log(f"Thinking early={THINK_EARLY.name} final={THINK_FINAL.name}")
    log(f"Math: {MATH_AD}")
    log(f"n_gsm8k={N_GSM8K} think_scale={THINK_SCALE} math_scale={MATH_SCALE} vel_thresh={VEL_THRESH}")
    log("=" * 72)
    for pth in (THINK_EARLY, THINK_FINAL, MATH_AD):
        assert pth.exists(), f"missing {pth}"
    log_mem("start")

    log("\n=== Build velocity-masked thinking deltas from trajectory ===")
    w_early = mx.load(str(THINK_EARLY))
    w_final = mx.load(str(THINK_FINAL))
    deltas, core_frac = build_velocity_deltas(w_early, w_final)

    math_ad = mx.load(str(MATH_AD))

    log("\n=== Load GSM8K ===")
    problems = load_gsm8k(N_GSM8K)

    # A: math-solo (ceiling, no thinking)
    A = run_condition("A_math_solo", problems, math_ad, None)
    # B: math + full step-1000 thinking (low-rank) — interference candidate
    B = run_condition("B_math_full_thinking", problems, math_ad,
                      lambda m: attach_thinking_lowrank(m, w_final))
    # C: math + early-velocity-core (dense) — the claim
    C = run_condition("C_math_early_core", problems, math_ad,
                      lambda m: attach_thinking_dense(m, deltas, "core"))
    # D: math + late-residual (dense) — control
    D = run_condition("D_math_late_residual", problems, math_ad,
                      lambda m: attach_thinking_dense(m, deltas, "late"))

    aa, ab, ac, ad = A["acc"], B["acc"], C["acc"], D["acc"]

    # ---- KILL 2298 (pre-registered, behavioral) ----
    clause_recover_fail = (ac - ab) < K_RECOVER        # core fails to recover >=6pp over full
    clause_late_fail = ac <= ad                        # late residual NOT the worse half
    killed = clause_recover_fail or clause_late_fail
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_velocity_mask_interference",
        "config": {
            "base_model": MODEL_ID,
            "thinking_early": str(THINK_EARLY),
            "thinking_final": str(THINK_FINAL),
            "math_adapter": str(MATH_AD),
            "think_projs": list(THINK_PROJS),
            "math_proj": MATH_PROJ,
            "think_scale": THINK_SCALE,
            "math_scale": MATH_SCALE,
            "vel_thresh": VEL_THRESH,
            "n_gsm8k": N_GSM8K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "global_core_frac": core_frac,
            "k_recover_pp": K_RECOVER,
            "mlx_lm": "0.31.2",
        },
        "conditions": {
            "A_math_solo": {"acc": aa, "details": A["details"]},
            "B_math_full_thinking": {"acc": ab, "details": B["details"]},
            "C_math_early_core": {"acc": ac, "details": C["details"]},
            "D_math_late_residual": {"acc": ad, "details": D["details"]},
        },
        "acc": {"A": aa, "B": ab, "C": ac, "D": ad},
        "interference_gap_A_minus_B": aa - ab,
        "recovered_C_minus_B": ac - ab,
        "late_minus_core_D_minus_C": ad - ac,
        "kill_criteria": {
            "2298": {
                "text": "acc(C)-acc(B) < +6pp OR acc(C) <= acc(D)",
                "type": "target_behavioral",
                "clause_recover_fail": bool(clause_recover_fail),
                "clause_late_fail": bool(clause_late_fail),
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
    log(f"acc  A(math-solo)={aa:.3f}  B(full-think)={ab:.3f}  C(early-core)={ac:.3f}  D(late-res)={ad:.3f}")
    log(f"recovered C-B = {ac-ab:+.3f} (need >=+{K_RECOVER})   core>late: {ac>ad} (C-D={ac-ad:+.3f})")
    log(f"KILL 2298: recover_fail={clause_recover_fail} late_fail={clause_late_fail}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
