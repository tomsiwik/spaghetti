#!/usr/bin/env python3
"""exp_spark_phase_scheduled_compose — decode-step phase-scheduled adapter mixing.

Frozen base gemma-4-e4b-it-4bit + r=6 q_proj math(GSM8K) and code(HumanEval) adapters (F#627),
from exp_composition_residual_analysis/. Task: reason-then-emit-strict-JSON (n=60 constructed
arithmetic word problems; answer must be returned as strict JSON {"answer": <int>}).

Per-layer q_proj output, decode-step weights w_m(t), w_c(t) read live each step:
    y_t = W h_t + s·[ w_m(t)·(h_t A_m) B_m + w_c(t)·(h_t A_c) B_c ]
Composition is Σ_i (B_i A_i), never (ΣB)(ΣA). LORA_SCALE = s = 6.0 ≤ 8.

MAGNITUDE-MATCH INVARIANT (F#863): w_m(t)+w_c(t)=1 ∀t in EVERY composed arm. Static uses 0.5/0.5.
Scheduled uses w_m=1-ε / w_c=ε in REASON phase, swapped in EMIT phase (ε=0.1 leak). Total injected
magnitude is identical between static and scheduled — only the TIMING differs.

PHASE DETECTOR (NOT oracle): a live boolean emitted_open_brace, flipped True the first decode step
whose generated token text contains '{'. Read by the wrapper BEFORE producing the next step's logits.

ARMS (combined = CoT-correct AND JSON-valid; report two sub-scores separately too):
  math_only : w_m=1, w_c=0           (best-single candidate)
  code_only : w_m=0, w_c=1           (best-single candidate)
  static    : w_m=w_c=0.5  ∀t        (magnitude-matched baseline)
  scheduled : phase schedule, sum=1  (magnitude-matched test arm)

KILL 2308 (pre-registered verbatim):
  "scheduled composition combined-score (CoT-correct AND JSON-valid) does NOT exceed static 0.5/0.5
   merge by >=15pp on the constructed phase task, OR static merge fails to show a >=15pp gap below
   best-single-adapter (regime underpowered)"
  -> UNDERPOWER GUARD tested FIRST: gap_underpower = best_single - static must be >= 0.15, else
     killed (regime underpowered), reported as the first result before testing the schedule.
  -> then lift = scheduled - static must be >= 0.15, else killed.

NO MOCKS. Real model, real adapters, real generation + real JSON parsing. is_smoke=False. mlx-lm 0.31.2.
"""

import gc
import json
import os
import random
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_DIR = EXP_DIR.parent / "exp_composition_residual_analysis"
ADAPTER_CODE = ADAPTER_DIR / "adapter_code.safetensors"
ADAPTER_MATH = ADAPTER_DIR / "adapter_math.safetensors"

LORA_SCALE = 6.0
LORA_RANK = 6
N_PROBLEMS = 60
MAX_NEW_TOKENS = 512
SEED = 42
N_LAYERS_EXPECTED = 42
EPS = 0.1                 # schedule leak; weights still sum to 1

GAP_UNDERPOWER_MIN = 0.15  # static must be >=15pp below best_single
LIFT_MIN = 0.15            # scheduled must beat static by >=15pp


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Live decode-step weight holder. The wrapper reads .wm / .wc each forward.
# The decode loop mutates these between steps based on the phase flag.
# ----------------------------------------------------------------------------

class Sched:
    """Holds the current per-step adapter weights (w_m, w_c). MAGNITUDE-MATCH: wm+wc==1."""
    def __init__(self, mode):
        self.mode = mode            # 'math', 'code', 'static', 'scheduled'
        self.emitted_open_brace = False
        self.reset()

    def reset(self):
        self.emitted_open_brace = False
        self.refresh()

    def refresh(self):
        if self.mode == "math":
            self.wm, self.wc = 1.0, 0.0
        elif self.mode == "code":
            self.wm, self.wc = 0.0, 1.0
        elif self.mode == "static":
            self.wm, self.wc = 0.5, 0.5
        elif self.mode == "scheduled":
            if self.emitted_open_brace:           # EMIT phase: code dominates
                self.wm, self.wc = EPS, 1.0 - EPS
            else:                                 # REASON phase: math dominates
                self.wm, self.wc = 1.0 - EPS, EPS
        else:
            raise ValueError(self.mode)
        # magnitude-match assertion (best-single arms legitimately sum to 1 on one adapter)
        assert abs((self.wm + self.wc) - 1.0) < 1e-9, (self.mode, self.wm, self.wc)

    def saw_token_text(self, text):
        """Flip to EMIT phase the first time a generated token contains '{'."""
        if self.mode == "scheduled" and not self.emitted_open_brace and "{" in text:
            self.emitted_open_brace = True
            self.refresh()


# ----------------------------------------------------------------------------
# Composed q_proj wrapper reading live weights from a shared Sched.
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class ComposedQProj(nn.Module):
    """y = linear(x) + s·[ wm·(x@Am)@Bm + wc·(x@Ac)@Bc ], wm/wc read live from sched."""
    def __init__(self, base_linear, am, bm, ac, bc, scale, sched):
        super().__init__()
        self.linear = base_linear
        self.am, self.bm = am, bm
        self.ac, self.bc = ac, bc
        self.scale = scale
        self.sched = sched
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        wm = self.sched.wm
        wc = self.sched.wc
        if wm != 0.0:
            dm = (x @ self.am) @ self.bm
            y = y + (self.scale * wm * dm).astype(x.dtype)
        if wc != 0.0:
            dc = (x @ self.ac) @ self.bc
            y = y + (self.scale * wc * dc).astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_composed(model, math_ad, code_ad, scale, sched):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in code_ad or bk not in code_ad:
            continue
        base_linear = layer.self_attn.q_proj
        am = math_ad[ak].astype(mx.float32)
        bm = math_ad[bk].astype(mx.float32)
        ac = code_ad[ak].astype(mx.float32)
        bc = code_ad[bk].astype(mx.float32)
        wrapper = ComposedQProj(base_linear, am, bm, ac, bc, scale, sched)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} ComposedQProj (mode={sched.mode})")
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    return model


# ----------------------------------------------------------------------------
# Constructed reason-then-emit-strict-JSON task (n=60, deterministic)
# ----------------------------------------------------------------------------

def build_problems(n, seed=SEED):
    """Small arithmetic word problems; ground-truth integer answer, returned as strict JSON."""
    rng = random.Random(seed)
    templates = [
        ("A baker had {a} loaves and baked {b} more, then sold {c}. How many loaves remain?",
         lambda a, b, c: a + b - c),
        ("There are {a} rows of chairs with {b} chairs each, and {c} chairs are removed. How many chairs are left?",
         lambda a, b, c: a * b - c),
        ("A tank holds {a} liters. {b} liters are added, then {c} liters drained. How many liters now?",
         lambda a, b, c: a + b - c),
        ("A class has {a} students. {b} groups of {c} students each leave. How many students remain?",
         lambda a, b, c: a - b * c),
        ("A shop earns ${a} per day for {b} days, then spends ${c}. What is the net profit?",
         lambda a, b, c: a * b - c),
    ]
    probs = []
    for i in range(n):
        tpl, fn = templates[i % len(templates)]
        a = rng.randint(8, 30)
        b = rng.randint(2, 9)
        c = rng.randint(1, 7)
        ans = fn(a, b, c)
        # keep answers positive & sensible
        if ans <= 0:
            a += abs(ans) + 5
            ans = fn(a, b, c)
        q = tpl.format(a=a, b=b, c=c)
        probs.append({"id": i, "question": q, "answer": int(ans)})
    log(f"  Built {len(probs)} reason-then-emit problems")
    return probs


def task_prompt(p):
    return (
        "Solve the problem. Think step by step. Then, on the FINAL line, output ONLY a strict JSON "
        "object of the exact form {\"answer\": <integer>} with the integer result and nothing else.\n\n"
        f"Problem: {p['question']}"
    )


def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


# ----------------------------------------------------------------------------
# Greedy decode with live phase detection from generated tokens.
# ----------------------------------------------------------------------------

def generate_phased(model, tokenizer, prompt, sched, max_new=MAX_NEW_TOKENS):
    sched.reset()                       # REASON phase, math-heavy / static / single
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    cur = tok.item()
    out = [cur]
    # update phase from the just-generated token BEFORE next forward
    sched.saw_token_text(tokenizer.decode([cur]))
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    for _ in range(max_new - 1):
        if cur in (eos, eot):
            break
        logits = model(mx.array([[cur]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        cur = tok.item()
        out.append(cur)
        sched.saw_token_text(tokenizer.decode([cur]))
    del cache
    return tokenizer.decode(out), len(out), sched.emitted_open_brace


# ----------------------------------------------------------------------------
# Scoring: CoT-correct, JSON-valid, combined.
# ----------------------------------------------------------------------------

def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_last_json_obj(text):
    """Return the last {...} substring that parses as JSON, else None."""
    t = strip_thinking(text)
    # also strip code fences if present
    t = t.replace("```json", "").replace("```", "")
    matches = list(re.finditer(r"\{[^{}]*\}", t))
    for m in reversed(matches):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def score(text, gt_answer):
    obj = extract_last_json_obj(text)
    json_valid = obj is not None and "answer" in obj
    cot_correct = False
    if json_valid:
        try:
            cot_correct = int(obj["answer"]) == int(gt_answer)
        except Exception:
            cot_correct = False
    combined = bool(json_valid and cot_correct)
    return bool(cot_correct), bool(json_valid), combined


def eval_arm(model, tokenizer, problems, sched):
    cot_c = json_c = comb_c = brace_c = 0
    details = []
    for p in problems:
        prompt = format_chat(tokenizer, task_prompt(p))
        text, ntok, saw_brace = generate_phased(model, tokenizer, prompt, sched)
        cot, jv, comb = score(text, p["answer"])
        cot_c += int(cot); json_c += int(jv); comb_c += int(comb); brace_c += int(saw_brace)
        details.append({"id": p["id"], "cot": cot, "json": jv, "combined": comb,
                        "saw_brace": bool(saw_brace), "ntok": ntok})
    n = len(problems)
    res = {
        "cot_correct": cot_c / n,
        "json_valid": json_c / n,
        "combined": comb_c / n,
        "brace_rate": brace_c / n,
        "details": details,
    }
    log(f"    [{sched.mode}] cot={res['cot_correct']:.3f} json={res['json_valid']:.3f} "
        f"combined={res['combined']:.3f} brace_rate={res['brace_rate']:.3f}")
    return res


def run_arm(problems, math_ad, code_ad, mode):
    log(f"\n=== ARM {mode} ===")
    model, tok = load(MODEL_ID)
    sched = Sched(mode)
    attach_composed(model, math_ad, code_ad, LORA_SCALE, sched)
    gc.collect(); mx.clear_cache()
    res = eval_arm(model, tok, problems, sched)
    log_mem(f"{mode}-done")
    del model, tok, sched
    gc.collect(); mx.clear_cache()
    return res


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_phase_scheduled_compose")
    log(f"Base: {MODEL_ID}")
    log(f"math adapter: {ADAPTER_MATH}")
    log(f"code adapter: {ADAPTER_CODE}")
    log(f"n={N_PROBLEMS} scale={LORA_SCALE} rank={LORA_RANK} eps={EPS} max_new={MAX_NEW_TOKENS}")
    log("=" * 72)
    assert ADAPTER_CODE.exists(), f"missing {ADAPTER_CODE}"
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    log_mem("start")

    problems = build_problems(N_PROBLEMS)
    math_ad = mx.load(str(ADAPTER_MATH))
    code_ad = mx.load(str(ADAPTER_CODE))

    # best-single candidates
    R_math = run_arm(problems, math_ad, code_ad, "math")
    R_code = run_arm(problems, math_ad, code_ad, "code")
    # magnitude-matched baseline + test arm
    R_static = run_arm(problems, math_ad, code_ad, "static")
    R_sched = run_arm(problems, math_ad, code_ad, "scheduled")

    c_math = R_math["combined"]
    c_code = R_code["combined"]
    c_static = R_static["combined"]
    c_sched = R_sched["combined"]
    best_single = max(c_math, c_code)
    best_single_arm = "math" if c_math >= c_code else "code"

    # --- UNDERPOWER GUARD (tested FIRST) ---
    gap_underpower = best_single - c_static
    underpowered = gap_underpower < GAP_UNDERPOWER_MIN

    # --- timing lift ---
    lift = c_sched - c_static

    # --- KILL 2308 (pre-registered verbatim) ---
    if underpowered:
        verdict = "killed"
        kill_reason = "regime underpowered: static not >=15pp below best_single"
        all_pass = False
    elif lift < LIFT_MIN:
        verdict = "killed"
        kill_reason = "scheduled does not exceed static by >=15pp (timing does not recover gap)"
        all_pass = False
    else:
        verdict = "supported"
        kill_reason = ""
        all_pass = True

    kill_2308_text = (
        "scheduled composition combined-score (CoT-correct AND JSON-valid) does NOT exceed static "
        "0.5/0.5 merge by >=15pp on the constructed phase task, OR static merge fails to show a "
        ">=15pp gap below best-single-adapter (regime underpowered)"
    )

    results = {
        "experiment_id": "exp_spark_phase_scheduled_compose",
        "config": {
            "base_model": MODEL_ID,
            "adapter_math": str(ADAPTER_MATH),
            "adapter_code": str(ADAPTER_CODE),
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_problems": N_PROBLEMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eps_leak": EPS,
            "magnitude_match_invariant": "w_m(t)+w_c(t)=1 for all t in every composed arm",
            "phase_detector": "live emitted_open_brace flag from generated tokens (NOT oracle)",
            "composition": "Sum_i (B_i A_i)",
            "gap_underpower_min": GAP_UNDERPOWER_MIN,
            "lift_min": LIFT_MIN,
            "mlx_lm": "0.31.2",
        },
        "arms": {
            "math_only": {k: R_math[k] for k in ("cot_correct", "json_valid", "combined", "brace_rate")},
            "code_only": {k: R_code[k] for k in ("cot_correct", "json_valid", "combined", "brace_rate")},
            "static_0.5_0.5": {k: R_static[k] for k in ("cot_correct", "json_valid", "combined", "brace_rate")},
            "scheduled": {k: R_sched[k] for k in ("cot_correct", "json_valid", "combined", "brace_rate")},
        },
        "details": {
            "math_only": R_math["details"],
            "code_only": R_code["details"],
            "static_0.5_0.5": R_static["details"],
            "scheduled": R_sched["details"],
        },
        "combined_scores": {"math": c_math, "code": c_code, "static": c_static, "scheduled": c_sched},
        "best_single": best_single,
        "best_single_arm": best_single_arm,
        # underpower guard reported FIRST, before the schedule test
        "underpower_guard": {
            "gap_underpower_best_single_minus_static": gap_underpower,
            "threshold": GAP_UNDERPOWER_MIN,
            "underpowered": bool(underpowered),
        },
        "timing_lift_scheduled_minus_static": lift,
        "timing_lift_threshold": LIFT_MIN,
        "kill_criteria": {
            "2308": {
                "text": kill_2308_text,
                "underpowered": bool(underpowered),
                "lift": lift,
                "result": "fail" if verdict == "killed" else "pass",
                "reason": kill_reason,
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
    log(f"combined  math={c_math:.3f}  code={c_code:.3f}  static={c_static:.3f}  scheduled={c_sched:.3f}")
    log(f"best_single={best_single:.3f} ({best_single_arm})")
    log(f"[GUARD FIRST] gap_underpower = best_single - static = {gap_underpower:.3f} "
        f"(need >={GAP_UNDERPOWER_MIN}) underpowered={underpowered}")
    log(f"timing lift = scheduled - static = {lift:.3f} (need >={LIFT_MIN})")
    log(f"VERDICT: {verdict}  all_pass={all_pass}  {kill_reason}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
