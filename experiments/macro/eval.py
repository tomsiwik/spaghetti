"""Perplexity evaluation for pretrained LLMs.

Two backends:
  - local: load via mlx-lm, compute NTP loss on GPU
  - api:   send texts to Together AI, collect logprobs
"""

import time
from dataclasses import dataclass, field, asdict

from .data import load_code_eval
from .models import get_model_info


@dataclass
class EvalResult:
    model_name: str
    hf_id: str
    tier: str
    param_count: int
    perplexity: dict[str, float] = field(default_factory=dict)
    tokens_per_sec: float = 0.0
    load_time_s: float = 0.0
    eval_time_s: float = 0.0
    peak_memory_gb: float = 0.0
    backend: str = "local"
    arch: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "EvalResult":
        return EvalResult(**{k: v for k, v in d.items() if k in EvalResult.__dataclass_fields__})


def compute_perplexity_local(
    model, tokenizer, texts: list[str], max_length: int = 512, batch_size: int = 1
) -> tuple[float, float]:
    """Compute perplexity on GPU via mlx. Returns (perplexity, tokens_per_sec)."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        all_tokens = []
        for text in batch_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) < 2:
                continue
            all_tokens.append(tokens)

        if not all_tokens:
            continue

        max_len = max(len(t) for t in all_tokens)
        padded = []
        masks = []
        for tokens in all_tokens:
            pad_len = max_len - len(tokens)
            padded.append(tokens + [0] * pad_len)
            masks.append([1.0] * len(tokens) + [0.0] * pad_len)

        input_ids = mx.array(padded)
        mask = mx.array(masks)

        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        target_mask = mask[:, 1:]

        log_probs = nn.losses.cross_entropy(logits, targets, reduction="none")
        masked_loss = log_probs * target_mask

        mx.eval(masked_loss, target_mask)
        total_loss += masked_loss.sum().item()
        total_tokens += target_mask.sum().item()

    elapsed = time.time() - t0
    if total_tokens == 0:
        return float("inf"), 0.0

    avg_loss = total_loss / total_tokens
    ppl = float(np.exp(avg_loss))
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    return ppl, tps


# --- Param count table for API models (can't introspect remotely) ---
_API_PARAM_COUNTS = {
    "MiniMaxAI/MiniMax-M2.5": 228_700_000_000,
    "deepseek-ai/DeepSeek-V3-0324": 671_000_000_000,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46_700_000_000,
}


def _eval_local(
    model_name: str,
    hf_id: str,
    tier: str,
    langs: list[str],
    n_samples: int,
    max_tokens: int,
    max_length: int,
) -> EvalResult:
    """Evaluate a model locally via mlx-lm."""
    from datetime import datetime, timezone
    from .models import count_params, load_model, unload_model

    print(f"\n{'='*60}")
    print(f"Evaluating (local): {model_name} ({hf_id})")
    print(f"{'='*60}")

    print(f"  Loading model...")
    model, tokenizer, load_time = load_model(hf_id)
    n_params = count_params(model)
    print(f"  Loaded in {load_time:.1f}s | {n_params:,} params")

    perplexity = {}
    total_tps = 0.0
    eval_t0 = time.time()

    for lang in langs:
        print(f"  Evaluating {lang}...")
        texts = load_code_eval(lang, n_samples=n_samples, max_tokens=max_tokens)
        ppl, tps = compute_perplexity_local(model, tokenizer, texts, max_length=max_length)
        perplexity[lang] = round(ppl, 2)
        total_tps = tps
        print(f"    {lang}: ppl={ppl:.2f} | {tps:.0f} tok/s")

    eval_time = time.time() - eval_t0

    is_quantized = "4bit" in hf_id or "8bit" in hf_id
    bytes_per_param = 0.5 if is_quantized else 2.0
    peak_mem_gb = (n_params * bytes_per_param) / 1e9

    del model, tokenizer
    unload_model()

    result = EvalResult(
        model_name=model_name,
        hf_id=hf_id,
        tier=tier,
        param_count=n_params,
        perplexity=perplexity,
        tokens_per_sec=round(total_tps, 1),
        load_time_s=round(load_time, 2),
        eval_time_s=round(eval_time, 2),
        peak_memory_gb=round(peak_mem_gb, 2),
        backend="local",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print(f"  Done: {result.perplexity} | {result.tokens_per_sec} tok/s | "
          f"load={result.load_time_s}s eval={result.eval_time_s}s")
    return result


def _eval_api(
    model_name: str,
    together_id: str,
    tier: str,
    arch: str,
    langs: list[str],
    n_samples: int,
    max_tokens: int,
) -> EvalResult:
    """Evaluate a model via Together AI logprobs API."""
    from datetime import datetime, timezone
    from .api import compute_perplexity_api

    print(f"\n{'='*60}")
    print(f"Evaluating (API): {model_name} ({together_id})")
    print(f"{'='*60}")

    n_params = _API_PARAM_COUNTS.get(together_id, 0)

    perplexity = {}
    total_tps = 0.0
    eval_t0 = time.time()

    for lang in langs:
        print(f"  Evaluating {lang} ({n_samples} snippets via API)...")
        texts = load_code_eval(lang, n_samples=n_samples, max_tokens=max_tokens)
        ppl, tps = compute_perplexity_api(together_id, texts, max_tokens_per_text=max_tokens)
        perplexity[lang] = round(ppl, 2)
        total_tps = tps
        print(f"    {lang}: ppl={ppl:.2f} | {tps:.0f} tok/s (API)")

    eval_time = time.time() - eval_t0

    result = EvalResult(
        model_name=model_name,
        hf_id=together_id,
        tier=tier,
        param_count=n_params,
        perplexity=perplexity,
        tokens_per_sec=round(total_tps, 1),
        load_time_s=0.0,
        eval_time_s=round(eval_time, 2),
        peak_memory_gb=0.0,
        backend="api",
        arch=arch,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print(f"  Done: {result.perplexity} | {result.tokens_per_sec} tok/s (API) | "
          f"eval={result.eval_time_s}s")
    return result


def eval_model(
    model_name: str,
    hf_id: str | None = None,
    tier: str | None = None,
    langs: list[str] | None = None,
    n_samples: int = 100,
    max_tokens: int = 512,
    max_length: int = 512,
    backend: str | None = None,
) -> EvalResult:
    """Full eval pipeline. Dispatches to local or API backend based on catalog."""
    if langs is None:
        langs = ["python"]

    # Resolve model info from catalog
    info = {}
    if hf_id is None:
        info = get_model_info(model_name)
    if backend is None:
        backend = info.get("backend", "local")

    if backend == "api":
        together_id = info.get("together_id", hf_id or model_name)
        return _eval_api(
            model_name=model_name,
            together_id=together_id,
            tier=info.get("tier", tier or "custom"),
            arch=info.get("arch", ""),
            langs=langs,
            n_samples=n_samples,
            max_tokens=max_tokens,
        )
    else:
        resolved_hf_id = hf_id or info.get("hf_id", model_name)
        return _eval_local(
            model_name=model_name,
            hf_id=resolved_hf_id,
            tier=info.get("tier", tier or "custom"),
            langs=langs,
            n_samples=n_samples,
            max_tokens=max_tokens,
            max_length=max_length,
        )
