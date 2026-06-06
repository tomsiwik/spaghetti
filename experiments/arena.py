"""Arena: compare models, build leaderboard, track lineage."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .models import get_model, MODEL_REGISTRY
from .data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from .train import train, train_multidomain, evaluate
from .metrics import RunMetrics, compute_forgetting, compute_learning_speed


TREE_PATH = Path("micro_model_tree.json")


def _count_params(model) -> int:
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def _build_datasets(tokenizer, docs_train, docs_val, block_size):
    train_ds = CharDataset(docs_train, tokenizer, block_size)
    val_ds = CharDataset(docs_val, tokenizer, block_size)
    return train_ds, val_ds


def run_single(
    model_name: str,
    steps: int = 500,
    block_size: int = 32,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 50,
    **model_kwargs,
) -> RunMetrics:
    """Train a single model, return metrics."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)

    kwargs = dict(vocab_size=tokenizer.vocab_size, block_size=block_size, **model_kwargs)
    model = get_model(model_name, **kwargs)
    mx.eval(model.parameters())

    n_params = _count_params(model)
    print(f"\n=== {model_name} ({n_params:,} params) ===")

    train_ds, val_ds = _build_datasets(tokenizer, docs_train, docs_val, block_size)
    result = train(model, train_ds, val_ds, steps=steps, batch_size=batch_size,
                   lr=lr, seed=seed, log_every=log_every)

    return RunMetrics(
        model_name=model_name,
        param_count=n_params,
        final_loss=result["final_loss"],
        val_loss=result["val_loss"],
        learning_speed=compute_learning_speed(result["losses"]),
        tokens_per_sec=result["tokens_per_sec"],
        elapsed_s=result["elapsed_s"],
    )


def run_multidomain(
    model_name: str,
    steps_per_domain: int = 300,
    block_size: int = 32,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 50,
    **model_kwargs,
) -> RunMetrics:
    """Train a model on sequential domains, return metrics with forgetting."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_ds = CharDataset(d_train, tokenizer, block_size)
        val_ds = CharDataset(d_val, tokenizer, block_size)
        domain_datasets[d_name] = (train_ds, val_ds)

    kwargs = dict(vocab_size=tokenizer.vocab_size, block_size=block_size, **model_kwargs)
    model = get_model(model_name, **kwargs)
    mx.eval(model.parameters())

    n_params = _count_params(model)
    print(f"\n=== {model_name} ({n_params:,} params) [multidomain] ===")

    result = train_multidomain(
        model, domain_datasets, steps_per_domain=steps_per_domain,
        batch_size=batch_size, lr=lr, seed=seed, log_every=log_every,
    )

    forgetting = compute_forgetting(result["eval_matrix"], result["domains"])

    # Final val loss = average across domains after last phase
    last_phase = list(result["eval_matrix"].keys())[-1]
    final_vals = result["eval_matrix"][last_phase]
    avg_val = sum(final_vals.values()) / len(final_vals)

    return RunMetrics(
        model_name=model_name,
        param_count=n_params,
        final_loss=avg_val,
        val_loss=avg_val,
        forgetting=forgetting,
    )


def run_comparison(
    model_names: list[str],
    mode: str = "single",
    **kwargs,
) -> list[RunMetrics]:
    """Run all models under identical conditions, return results."""
    results = []
    for name in model_names:
        if mode == "multidomain":
            r = run_multidomain(name, **kwargs)
        else:
            r = run_single(name, **kwargs)
        results.append(r)
    return results


def leaderboard(results: list[RunMetrics]) -> str:
    """Format results as a ranked leaderboard table."""
    # Sort by val_loss (lower is better)
    ranked = sorted(results, key=lambda r: r.val_loss or r.final_loss)
    lines = [
        f"{'Rank':>4} | {'Model':<16} | {'Val Loss':>9} | {'Params':>8} | {'tok/s':>8} | {'Time':>6} | {'Speed':>5}",
        "-" * 75,
    ]
    for i, r in enumerate(ranked, 1):
        vl = f"{r.val_loss:.4f}" if r.val_loss is not None else "   N/A"
        sp = f"{r.learning_speed}" if r.learning_speed else "  N/A"
        lines.append(
            f"{i:>4} | {r.model_name:<16} | {vl:>9} | {r.param_count:>8,} | {r.tokens_per_sec:>8.0f} | {r.elapsed_s:>5.1f}s | {sp:>5}"
        )
    if any(r.forgetting for r in results):
        lines.append("")
        lines.append("Forgetting:")
        for r in ranked:
            if r.forgetting:
                for d, f in r.forgetting.items():
                    lines.append(f"  {r.model_name}/{d}: {f['forgetting']:+.3f} ({f['pct']:+.1f}%)")
    return "\n".join(lines)


def pareto_dominates(a: RunMetrics, b: RunMetrics) -> bool:
    """True if a dominates b on all key metrics (lower is better for all)."""
    a_loss = a.val_loss or a.final_loss
    b_loss = b.val_loss or b.final_loss
    metrics_a = [a_loss, a.param_count]
    metrics_b = [b_loss, b.param_count]
    return all(ma <= mb for ma, mb in zip(metrics_a, metrics_b)) and any(
        ma < mb for ma, mb in zip(metrics_a, metrics_b)
    )


class ModelTree:
    """JSON-backed lineage tree for model variants."""

    def __init__(self, path: Path = TREE_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"models": {}}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def record(self, metrics: RunMetrics):
        """Record a model run in the tree."""
        entry = metrics.to_dict()
        parent = MODEL_REGISTRY.get(metrics.model_name, {}).get("parent")
        entry["parent"] = parent
        name = metrics.model_name
        if name not in self.data["models"]:
            self.data["models"][name] = {"parent": parent, "runs": []}
        self.data["models"][name]["runs"].append(entry)
        self.save()

    def show_tree(self) -> str:
        """Render lineage tree as text."""
        models = self.data.get("models", {})
        if not models:
            return "(empty tree)"

        # Build parent→children map
        children: dict[str | None, list[str]] = {}
        for name, info in models.items():
            p = info.get("parent")
            children.setdefault(p, []).append(name)

        lines = []
        def render(name: str, prefix: str = "", is_last: bool = True):
            connector = "`-- " if is_last else "|-- "
            runs = models[name].get("runs", [])
            best = min((r.get("val_loss") or r.get("final_loss", 999) for r in runs), default=None)
            best_str = f" (best={best:.4f})" if best is not None else ""
            lines.append(f"{prefix}{connector}{name}{best_str}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            kids = children.get(name, [])
            for i, kid in enumerate(kids):
                render(kid, child_prefix, i == len(kids) - 1)

        roots = children.get(None, [])
        for i, root in enumerate(roots):
            render(root, "", i == len(roots) - 1)
        return "\n".join(lines)
