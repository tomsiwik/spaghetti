"""Training infrastructure for macro capsule groups."""

import math
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .surgery import freeze_except_capsules, freeze_except_router


def _warmup_cosine_schedule(lr: float, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0. Returns callable(step) -> lr."""
    warmup = optim.linear_schedule(1e-7, lr, steps=warmup_steps)
    cosine = optim.cosine_decay(lr, decay_steps=total_steps - warmup_steps)
    return optim.join_schedules([warmup, cosine], [warmup_steps])


def tokenize_texts(tokenizer, texts: list[str],
                   max_length: int = 512, min_length: int = 10) -> list[list[int]]:
    """Tokenize and filter texts to usable lengths."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= min_length:
            all_tokens.append(tokens[:max_length])
    return all_tokens


def make_batch(all_tokens: list[list[int]], batch_size: int,
               rng: random.Random, max_length: int = 512):
    """Sample a batch, pad to uniform length. Returns (input_ids, mask)."""
    batch = [rng.choice(all_tokens) for _ in range(batch_size)]
    max_len = min(max(len(t) for t in batch), max_length)
    padded = []
    masks = []
    for tokens in batch:
        t = tokens[:max_len]
        pad_len = max_len - len(t)
        padded.append(t + [0] * pad_len)
        masks.append([1.0] * len(t) + [0.0] * pad_len)
    return mx.array(padded), mx.array(masks)


def ntp_loss(model, input_ids, mask):
    """Next-token prediction loss with padding mask."""
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    target_mask = mask[:, 1:]
    loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (loss * target_mask).sum() / (target_mask.sum() + 1e-8)


def train_capsule_groups(model, tokenizer, texts: list[str],
                         steps: int = 500, lr: float = 1e-4,
                         warmup_frac: float = 0.1,
                         batch_size: int = 4, max_length: int = 512,
                         seed: int = 42, log_every: int = 50) -> list[float]:
    """Train capsule groups on domain-specific texts. Freezes everything else.

    Uses warmup+cosine LR schedule (Exp 19-20: cuts dead capsules from 47% to 20%).
    """
    print(f"  Tokenizing {len(texts)} texts...")
    all_tokens = tokenize_texts(tokenizer, texts, max_length)
    print(f"  {len(all_tokens)} sequences after filtering")

    freeze_except_capsules(model)

    trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")

    warmup_steps = max(1, int(steps * warmup_frac))
    schedule = _warmup_cosine_schedule(lr, warmup_steps, steps)
    optimizer = optim.Adam(learning_rate=schedule)
    print(f"  LR schedule: warmup {warmup_steps} steps → cosine decay")

    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    rng = random.Random(seed)
    t0 = time.time()
    losses = []

    for step in range(1, steps + 1):
        input_ids, mask = make_batch(all_tokens, batch_size, rng, max_length)
        loss, grads = loss_and_grad(model, input_ids, mask)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            current_lr = schedule(step)
            print(f"    step {step:4d}/{steps} | loss {loss_val:.4f} | lr {current_lr:.2e} | {elapsed:.1f}s")

    model.unfreeze()
    return losses


def calibrate_router(model, tokenizer, texts_a: list[str], texts_b: list[str],
                     steps: int = 200, lr: float = 1e-4,
                     warmup_frac: float = 0.1,
                     batch_size: int = 4, max_length: int = 512,
                     seed: int = 42, log_every: int = 50) -> list[float]:
    """Calibrate router on mixed-domain data. Freezes everything except routers.

    Uses warmup+cosine LR schedule.
    """
    print(f"  Tokenizing domain A ({len(texts_a)}) and B ({len(texts_b)})...")
    tokens_a = tokenize_texts(tokenizer, texts_a, max_length)
    tokens_b = tokenize_texts(tokenizer, texts_b, max_length)

    freeze_except_router(model)

    trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params (router only): {trainable:,}")

    warmup_steps = max(1, int(steps * warmup_frac))
    schedule = _warmup_cosine_schedule(lr, warmup_steps, steps)
    optimizer = optim.Adam(learning_rate=schedule)

    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    rng = random.Random(seed)
    t0 = time.time()
    losses = []

    for step in range(1, steps + 1):
        source = tokens_a if step % 2 == 1 else tokens_b
        input_ids, mask = make_batch(source, batch_size, rng, max_length)
        loss, grads = loss_and_grad(model, input_ids, mask)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    router cal step {step:4d}/{steps} | loss {loss_val:.4f} | {elapsed:.1f}s")

    model.unfreeze()
    return losses
