"""CharDataset: names.txt tokenizer, train/val split, domain splits."""

import os
import urllib.request
import random

import mlx.core as mx


DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
DATA_PATH = "input.txt"


def load_names(path: str = DATA_PATH) -> list[str]:
    """Download (if needed) and return list of name strings."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    return [line.strip() for line in open(path) if line.strip()]


class CharTokenizer:
    """Char-level tokenizer with BOS token."""

    def __init__(self, docs: list[str]):
        self.chars = sorted(set("".join(docs)))
        self.bos = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self._c2i = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self._c2i[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] if i != self.bos else "" for i in ids)


class CharDataset:
    """Pack names into fixed-length sequences for next-token prediction."""

    def __init__(self, docs: list[str], tokenizer: CharTokenizer, block_size: int = 32):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Each doc becomes: [BOS] + chars + [BOS]
        self.sequences = []
        for doc in docs:
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def get_batch(self, batch_size: int, rng: random.Random | None = None) -> tuple[mx.array, mx.array]:
        """Return (inputs, targets) each of shape (B, T)."""
        rng = rng or random
        seqs = rng.choices(self.sequences, k=batch_size)
        inputs, targets = [], []
        for seq in seqs:
            n = min(self.block_size, len(seq) - 1)
            # Pad to block_size
            inp = seq[:n] + [0] * (self.block_size - n)
            tgt = seq[1:n + 1] + [0] * (self.block_size - n)
            inputs.append(inp)
            targets.append(tgt)
        return mx.array(inputs), mx.array(targets)


def domain_split(docs: list[str], method: str = "binary") -> dict[str, list[str]]:
    """Split docs into domains by first character.

    Methods:
      'binary': a-m vs n-z (2 domains)
      'quintary': a-e, f-j, k-o, p-t, u-z (5 domains)
    """
    if method == "binary":
        d0 = [d for d in docs if d[0].lower() <= "m"]
        d1 = [d for d in docs if d[0].lower() > "m"]
        return {"a_m": d0, "n_z": d1}
    if method == "quintary":
        ranges = [("a", "e"), ("f", "j"), ("k", "o"), ("p", "t"), ("u", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    if method == "quaternary":
        ranges = [("a", "f"), ("g", "m"), ("n", "s"), ("t", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    if method == "ternary":
        ranges = [("a", "h"), ("i", "p"), ("q", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    if method == "senary":
        ranges = [("a", "d"), ("e", "h"), ("i", "l"), ("m", "p"), ("q", "t"), ("u", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    if method == "septenary":
        ranges = [("a", "d"), ("e", "g"), ("h", "k"), ("l", "n"), ("o", "r"), ("s", "u"), ("v", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    if method == "octonary":
        ranges = [("a", "c"), ("d", "f"), ("g", "i"), ("j", "l"),
                  ("m", "o"), ("p", "r"), ("s", "u"), ("v", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    raise ValueError(f"Unknown split method: {method}")


def train_val_split(docs: list[str], val_frac: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 train/val split."""
    rng = random.Random(seed)
    docs = list(docs)
    rng.shuffle(docs)
    n_val = int(len(docs) * val_frac)
    return docs[n_val:], docs[:n_val]
