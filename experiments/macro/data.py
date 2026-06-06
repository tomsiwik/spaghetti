"""Code dataset loading from HuggingFace (streaming, cached)."""

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "macro-code-eval"


def _cache_path(lang: str, n_samples: int, max_tokens: int, seed: int) -> Path:
    key = f"{lang}-{n_samples}-{max_tokens}-{seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{lang}_{n_samples}_{h}.json"


def load_code_eval(
    lang: str = "python",
    n_samples: int = 100,
    max_tokens: int = 512,
    seed: int = 42,
) -> list[str]:
    """Load code snippets from codeparrot/codeparrot-clean (streaming).

    Caches extracted text to ~/.cache/macro-code-eval/ to avoid re-downloading.
    Uses deterministic sampling via seed-based skip in streaming.
    """
    cache = _cache_path(lang, n_samples, max_tokens, seed)
    if cache.exists():
        return json.loads(cache.read_text())

    from datasets import load_dataset

    # codeparrot-clean is Python-only; for JS we use another source
    if lang == "python":
        ds = load_dataset(
            "codeparrot/codeparrot-clean", split="train", streaming=True
        )
        text_key = "content"
    elif lang == "javascript":
        ds = load_dataset(
            "code_search_net", "javascript",
            split="train",
            streaming=True,
        )
        text_key = "whole_func_string"
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'python' or 'javascript'.")

    # Deterministic skip: skip `seed * 1000` examples, then take n_samples
    skip = seed * 1000
    snippets = []
    for i, example in enumerate(ds):
        if i < skip:
            continue
        text = example[text_key]
        # Filter: non-empty, reasonable length
        if not text or len(text) < 50:
            continue
        # Truncate to roughly max_tokens chars (approx 4 chars/token)
        truncated = text[: max_tokens * 4]
        snippets.append(truncated)
        if len(snippets) >= n_samples:
            break

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(snippets))
    print(f"Cached {len(snippets)} {lang} snippets to {cache}")
    return snippets


def load_code_train(
    lang: str = "python",
    n_samples: int = 5000,
    max_tokens: int = 512,
    seed: int = 0,
) -> list[str]:
    """Load training code snippets (non-overlapping with eval set).

    Uses seed=0 by default (skip=0) so training data starts from the
    beginning of the stream, while eval uses seed=42 (skip=42000).
    """
    return load_code_eval(lang, n_samples=n_samples, max_tokens=max_tokens, seed=seed)


def load_multilang_eval(
    langs: list[str] | None = None,
    n_samples: int = 100,
    max_tokens: int = 512,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Multi-language eval sets for cross-domain evaluation."""
    if langs is None:
        langs = ["python"]
    return {lang: load_code_eval(lang, n_samples, max_tokens, seed) for lang in langs}
