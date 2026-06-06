import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import matplotlib.pyplot as plt
import seaborn as sns


def _iter_lora_leaves(loradict: Dict[str, Any]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    leaves: List[Tuple[str, Dict[str, torch.Tensor]]] = []

    def rec(node: Any, path: str):
        if isinstance(node, dict):
            if "A" in node and "B" in node:
                leaves.append((path, node))
                return
            for k, v in node.items():
                rec(v, f"{path}.{k}" if path else str(k))

    rec(loradict, "")
    return leaves


def _sanitize_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().float().cpu()
    if t.ndim == 1:
        return t.unsqueeze(0)  # 1 x N
    if t.ndim == 2:
        return t
    return t.reshape(-1, t.shape[-1])


def _select_batch(t: torch.Tensor, batch_index: int) -> torch.Tensor:
    if t.ndim >= 3:
        if batch_index >= t.shape[0]:
            raise IndexError(f"batch_index={batch_index} out of range for shape[0]={t.shape[0]}")
        return t[batch_index]
    return t


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _crop2d(t2d: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return t2d[: min(h, t2d.shape[0]), : min(w, t2d.shape[1])]


def _labels(n: int, max_labels: int) -> Tuple[bool, bool]:
    """
    Returns (show_ticks, rotate_x) policy.
    We show ticks only if n <= max_labels to keep images readable.
    """
    if n <= max_labels:
        return True, (n > 32)
    return False, False


def _sns_heatmap(ax, mat2d: torch.Tensor, title: str, xlab: Optional[List[int]] = None, ylab: Optional[List[int]] = None):
    # If labels provided, seaborn will place them; else keep them off for readability.
    show_x = xlab is not None
    show_y = ylab is not None
    sns.heatmap(
        mat2d,
        ax=ax,
        cbar=True,
        xticklabels=xlab if show_x else False,
        yticklabels=ylab if show_y else False,
    )
    ax.set_title(title)


@torch.no_grad()
def visualize_loradict_to_files(
    loradict: Dict[str, Any],
    out_dir: str,
    layer: Optional[int] = None,
    only: Optional[str] = None,
    batch_index: int = 0,
    crop_ratio: int = 16,           # take ~1/crop_ratio of each dimension
    min_in: int = 1,
    min_out: int = 1,
    min_r: int = 4,
    max_tick_labels: int = 64,      # show numeric tick labels only up to this many
    dpi: int = 300,
):
    """
    Save heatmaps for A, B, C (if present), and ΔW = A@B for each LoRA leaf.

    Cropping uses ratios of original shapes (not fixed lengths):
      A:  (in, r)   -> (ceil(in/crop_ratio),  ceil(r/crop_ratio))
      B:  (r, out)  -> (ceil(r/crop_ratio),   ceil(out/crop_ratio))
      ΔW: (in, out) -> (ceil(in/crop_ratio),  ceil(out/crop_ratio))

    Layout:
      Row 0: [blank] [B spans col 1-2]
      Row 1: [A]     [C]              [ΔW]   (ΔW bottom-right)

    Tick labels:
      - ΔW rows are 0..(A_crop_rows-1)  (same as A rows)
      - ΔW cols are 0..(B_crop_cols-1)  (same as B cols)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Restrict to a single layer if requested
    if layer is not None:
        if layer not in loradict:
            raise KeyError(f"layer={layer} not found in loradict keys={list(loradict.keys())[:10]}...")
        loradict_view = {layer: loradict[layer]}
    else:
        loradict_view = loradict

    leaves = _iter_lora_leaves(loradict_view)
    if only is not None:
        leaves = [(p, d) for (p, d) in leaves if only in p]

    if not leaves:
        raise ValueError("No LoRA leaves found to visualize (check `layer` / `only`).")

    saved_paths = []

    for path, leaf in leaves:
        print(f"Visualizing: {path}")

        A = _select_batch(leaf["A"], batch_index)
        B = _select_batch(leaf["B"], batch_index)

        A2 = _as_2d(A)  # (in, r)
        B2 = _as_2d(B)  # (r, out)

        in_dim, r_dim = A2.shape
        r_dim_b, out_dim = B2.shape
        if r_dim_b != r_dim:
            # Still allow, but warn loudly because ΔW will be invalid otherwise.
            raise ValueError(f"Rank mismatch for {path}: A is (in={in_dim}, r={r_dim}) but B is (r={r_dim_b}, out={out_dim})")

        # Ratio-based crop sizes
        crop_in = max(min_in, _ceil_div(in_dim, crop_ratio))
        crop_out = max(min_out, _ceil_div(out_dim, crop_ratio))
        crop_r = max(min_r, _ceil_div(r_dim, crop_ratio))

        # Crop A and B, then compute ΔW from cropped blocks
        A2c = _crop2d(A2, crop_in, crop_r)      # (crop_in, crop_r)
        B2c = _crop2d(B2, crop_r, crop_out)     # (crop_r, crop_out)
        dWc = A2c @ B2c                         # (crop_in, crop_out)

        # Decide whether to show tick labels
        show_yA, _ = _labels(A2c.shape[0], max_tick_labels)
        show_xA, _ = _labels(A2c.shape[1], max_tick_labels)
        show_yB, _ = _labels(B2c.shape[0], max_tick_labels)
        show_xB, _ = _labels(B2c.shape[1], max_tick_labels)
        show_yDW, _ = _labels(dWc.shape[0], max_tick_labels)
        show_xDW, _ = _labels(dWc.shape[1], max_tick_labels)

        # Labels that enforce "same rows as A" and "same cols as B"
        A_rows = list(range(A2c.shape[0])) if show_yA else None
        A_cols = list(range(A2c.shape[1])) if show_xA else None
        B_rows = list(range(B2c.shape[0])) if show_yB else None
        B_cols = list(range(B2c.shape[1])) if show_xB else None

        DW_rows = list(range(A2c.shape[0])) if show_yDW else None     # match A rows
        DW_cols = list(range(B2c.shape[1])) if show_xDW else None     # match B cols

        # --- Layout using GridSpec (2 rows x 3 cols) ---
        fig = plt.figure(figsize=(34, 22))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.3], width_ratios=[1.0, 1.0, 1.0])

        axBlank = fig.add_subplot(gs[0, 0])
        axB     = fig.add_subplot(gs[0, 1:3])
        axA     = fig.add_subplot(gs[1, 0])
        axDW    = fig.add_subplot(gs[1, 1:3])

        axBlank.axis("off")
        axBlank.set_title(path)

        _sns_heatmap(
            axA,
            A2c,
            f"A crop (in×r) {tuple(A2.shape)} -> {tuple(A2c.shape)}  (≈1/{crop_ratio})",
            xlab=A_cols,
            ylab=A_rows,
        )
        _sns_heatmap(
            axB,
            B2c,
            f"B crop (r×out) {tuple(B2.shape)} -> {tuple(B2c.shape)}  (≈1/{crop_ratio})",
            xlab=B_cols,
            ylab=B_rows,
        )

        _sns_heatmap(
            axDW,
            dWc,
            f"ΔW = A@B crop {tuple(dWc.shape)}  (rows=A, cols=B)",
            xlab=DW_cols,
            ylab=DW_rows,
        )

        fig.tight_layout()

        fname = f"{_sanitize_filename(path)}__b{batch_index}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {fpath}")
        plt.close(fig)

        saved_paths.append(fpath)

    return saved_paths
