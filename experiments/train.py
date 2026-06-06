"""Training loop: shared across all arena models."""

import time
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .data import CharDataset


def ntp_loss(model, inputs, targets):
    """Next-token prediction loss + model aux_loss."""
    logits = model(inputs)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="mean",
    )
    return loss + model.aux_loss()


def train(
    model,
    dataset: CharDataset,
    val_dataset: CharDataset | None = None,
    steps: int = 500,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 50,
) -> dict:
    """Train a model, return metrics dict."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()
    total_tokens = 0

    for step in range(1, steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)
        total_tokens += inputs.size

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  step {step:4d}/{steps} | loss {loss_val:.4f} | {tps:.0f} tok/s")

    elapsed = time.time() - t0
    val_loss = evaluate(model, val_dataset, batch_size) if val_dataset else None

    return {
        "final_loss": losses[-1],
        "val_loss": val_loss,
        "losses": losses,
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "total_tokens": total_tokens,
    }


def evaluate(model, dataset: CharDataset, batch_size: int = 32, n_batches: int = 10) -> float:
    """Evaluate average loss over n_batches."""
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        total += loss.item()
    return total / n_batches


def train_multidomain(
    model,
    domain_datasets: dict[str, tuple[CharDataset, CharDataset]],
    steps_per_domain: int = 300,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 50,
) -> dict:
    """Train sequentially on each domain, eval on all after each phase."""
    domains = list(domain_datasets.keys())
    eval_matrix = {}  # {phase: {domain: val_loss}}
    all_losses = {}

    for i, domain in enumerate(domains):
        print(f"\n--- Domain {i+1}/{len(domains)}: {domain} ---")
        train_ds, val_ds = domain_datasets[domain]

        if i > 0:
            model.on_domain_switch(domain)

        result = train(model, train_ds, val_ds, steps=steps_per_domain,
                       batch_size=batch_size, lr=lr, seed=seed + i, log_every=log_every)
        all_losses[domain] = result["losses"]

        # Eval on all domains
        phase_eval = {}
        for d_name, (_, d_val) in domain_datasets.items():
            phase_eval[d_name] = evaluate(model, d_val, batch_size)
        eval_matrix[f"after_{domain}"] = phase_eval
        print(f"  eval: {' | '.join(f'{d}={v:.3f}' for d, v in phase_eval.items())}")

    return {
        "eval_matrix": eval_matrix,
        "losses": all_losses,
        "domains": domains,
    }
