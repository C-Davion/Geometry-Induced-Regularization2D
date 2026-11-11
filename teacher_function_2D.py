'''
teacher_function_2d.py

Random search for "interesting" teacher functions R^2 -> R of the form

        (x, y)  ->  u = (x, y)
                ->  rho(u)  in R^(seq_len * embed_dim)        (single-head attention feature map)
                ->  max( min(A.rho, A1.rho), min(B.rho, B1.rho) )

For a number of trials, this script
  1. samples a 2D attention feature map rho (seq_len=8, embed_dim=16 by default)
     and the four branch vectors A, A1, B, B1 ~ U(-3, 3)^(seq_len*embed_dim),
  2. evaluates the function on a regular (x, y) grid,
  3. scores it (2D analogue of the 1D "num_segments + 0.25 * kink_strength"),
  4. plots the candidates that pass the thresholds and asks, on the command line,
     whether to save them,
  5. appends the saved ones to a JSONL file (one function per line) from which
     they can be rebuilt exactly (see rebuild_teacher / make_target_fn).

----------------------------------------------------------------------------
The feature map (what the forward computes)
----------------------------------------------------------------------------
Tokens (S = seq_len, D = embed_dim, d = 2 inputs):
    h_s(u) = E_s^T (m_s * u) + B_s              E_s in R^{d x D}, B_s in R^D
  m_s = 1 for scheme="mixed" (every token sees x and y), m_s = e_kappa(s) for
  scheme="separate" (token s only sees coordinate kappa(s) = floor(s*d/S)).
Single-head self-attention over the S tokens:
    q_s = W_Q h_s + b_Q,  k_t = W_K h_t + b_K,  v_t = W_V h_t + b_V
    alpha_st = softmax_t( q_s . k_t / sqrt(D) ),   h'_s = W_O ( sum_t alpha_st v_t ) + b_O
Flatten (row-major, (s, j) -> s*D + j):
    rho(u) = (h'_1; ...; h'_S) in R^(S*D)

----------------------------------------------------------------------------
The score (2D analogue of the 1D one)
----------------------------------------------------------------------------
At every grid point, the active affine branch gets a label in {0: A, 1: A1,
2: B, 3: B1} (same rule as in the 1D script).
  * num_regions     : number of connected regions of constant label (4-connectivity),
                      ignoring specks smaller than region_frac * (number of grid points).
                      <- the 1D "num_segments".
  * boundary_length : total length of the label boundaries (L1-type estimate).
  * kink_strength   : line integral of the gradient jump along the boundaries,
                          sum over boundary edges of || grad y(after) - grad y(before) || * edge length
                      <- the 1D sum of |slope jump| over switches. Grid-resolution independent.
  * score = num_regions + kink_weight * kink_strength          (kink_weight = 0.25)
Candidates need num_regions >= min_regions and score >= min_score (defaults
mirror the 1D script; they are NOT calibrated for 2D - the end-of-run summary
prints the score distribution to help you choose).

Note: with b_scale=0 (default) and the default zero attention biases, rho is an
odd function (rho(-u) = -rho(u)), so all four branches vanish at the origin and
every region boundary passes through (0, 0). Use --b-scale > 0 to break that.
'''

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage
from relu_network import LinearFeatureAttention2D


   


# ============================================================
# The teacher function: max(min(.,.), min(.,.)) of four linear forms in rho
# ============================================================

def f(A, A1, B, B1, X):
    Ax = X @ A
    A1x = X @ A1
    Bx = X @ B
    B1x = X @ B1

    mA = torch.minimum(Ax, A1x)
    mB = torch.minimum(Bx, B1x)

    y = torch.maximum(mA, mB)

    return y, Ax, A1x, Bx, B1x


# ============================================================
# Score (pure numpy, no torch)
# ============================================================

def compute_labels(Ax, A1x, Bx, B1x):
    """Label of the active affine branch at every point: 0=A, 1=A1, 2=B, 3=B1."""
    use_A = np.minimum(Ax, A1x) >= np.minimum(Bx, B1x)
    labels = np.empty(Ax.shape, dtype=int)
    labels[use_A & (Ax <= A1x)] = 0
    labels[use_A & (Ax > A1x)] = 1
    labels[~use_A & (Bx <= B1x)] = 2
    labels[~use_A & (Bx > B1x)] = 3
    return labels


def score_function(y, Ax, A1x, Bx, B1x, xs, ys,
                   region_frac=5e-4, kink_weight=0.25, margin=3):
    """
    2D analogue of the 1D score. Arrays are indexed [iy, ix]; xs (nx,), ys (ny,).

    Returns a dict with: labels, num_regions, boundary_length, kink_strength, score.
    """
    labels = compute_labels(Ax, A1x, Bx, B1x)
    ny, nx = labels.shape
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]

    # --- number of connected regions of constant label (specks ignored) ---
    min_pix = max(4, int(region_frac * labels.size))
    num_regions = 0
    for k in range(4):
        comp, n = ndimage.label(labels == k)               # 4-connectivity
        if n:
            sizes = np.bincount(comp.ravel())[1:]
            num_regions += int((sizes >= min_pix).sum())

    # --- boundaries: edges between neighbouring grid points with different labels ---
    diff_h = labels[:, :-1] != labels[:, 1:]               # (ny, nx-1): (i, j) | (i, j+1)
    diff_v = labels[:-1, :] != labels[1:, :]               # (ny-1, nx): (i, j) / (i+1, j)
    boundary_length = float(dy * diff_h.sum() + dx * diff_v.sum())

    # --- kink strength: gradient jump across the boundaries ---
    gy, gx = np.gradient(y, ys, xs)
    m = margin                                              # ignore edges close to the border (as in 1D)
    kink = 0.0
    if ny > 2 * m + 3 and nx > 2 * m + 3:
        # horizontal crossings (i, j) | (i, j+1), j in [m, nx-m-1]; compare gradients at j-1 and j+2
        mh = diff_h[m:ny - m, m:nx - m]
        jh = np.hypot(gx[m:ny - m, m + 2:nx - m + 2] - gx[m:ny - m, m - 1:nx - m - 1],
                      gy[m:ny - m, m + 2:nx - m + 2] - gy[m:ny - m, m - 1:nx - m - 1])
        # vertical crossings (i, j) / (i+1, j), i in [m, ny-m-1]; compare gradients at i-1 and i+2
        mv = diff_v[m:ny - m, m:nx - m]
        jv = np.hypot(gx[m + 2:ny - m + 2, m:nx - m] - gx[m - 1:ny - m - 1, m:nx - m],
                      gy[m + 2:ny - m + 2, m:nx - m] - gy[m - 1:ny - m - 1, m:nx - m])
        kink = float(dy * jh[mh].sum() + dx * jv[mv].sum())

    return {
        "labels": labels,
        "num_regions": num_regions,
        "boundary_length": boundary_length,
        "kink_strength": kink,
        "score": num_regions + kink_weight * kink,
    }


# ============================================================
# Plot
# ============================================================

LABEL_NAMES = ["A", "A1", "B", "B1"]
LABEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def plot_candidate(xs, ys, y, stats, title=""):
    """Figure 1: 3D surface of f(x, y). Figure 2: active branch regions."""

    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    extent = [
        xs[0] - dx / 2, xs[-1] + dx / 2,
        ys[0] - dy / 2, ys[-1] + dy / 2
    ]

    # --------------------------------------------------
    # Figure 1: 3D plot of R² -> R mapping
    # --------------------------------------------------
    X, Y = np.meshgrid(xs, ys)

    fig_3d = plt.figure(figsize=(10, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    surf = ax_3d.plot_surface(
        X, Y, y,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
        alpha=0.95
    )

    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.65, pad=0.1,
                    label="teacher(x,y)")

    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("f(x, y)")
    ax_3d.set_title("teacher function — 3D mapping")

    ax_3d.view_init(elev=30, azim=-135)
    fig_3d.tight_layout()

    # --------------------------------------------------
    # Figure 2: Active branch classification
    # --------------------------------------------------
    fig_labels, ax_l = plt.subplots(figsize=(7, 6))

    labels = stats["labels"]

    ax_l.imshow(
        labels,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=ListedColormap(LABEL_COLORS),
        vmin=-0.5,
        vmax=3.5
    )

    present = [k for k in range(4) if (labels == k).any()]

    ax_l.legend(
        handles=[
            Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k])
            for k in present
        ],
        loc="upper right",
        fontsize=8
    )

    ax_l.set_xlabel("x")
    ax_l.set_ylabel("y")
    ax_l.set_title("Active branch")

    fig_labels.suptitle(
        f"{title}  "
        f"score={stats['score']:.1f}  "
        f"regions={stats['num_regions']}  "
        f"boundary length={stats['boundary_length']:.1f}  "
        f"kink={stats['kink_strength']:.1f}"
    )

    fig_labels.tight_layout()

    return fig_3d, fig_labels

# ============================================================
# Saving / loading (JSONL, one function per line)
# ============================================================

def make_record(rho, A, A1, B, B1, stats, x_range, grid):
    """
    Everything needed to rebuild the function. rho's own parameters are nested
    under "rho" so that its "B" cannot collide with the branch vector "B".
    """
    return {
        "score": float(stats["score"]),
        "num_regions": int(stats["num_regions"]),
        "boundary_length": float(stats["boundary_length"]),
        "kink_strength": float(stats["kink_strength"]),
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_dim": rho.input_dim,
        "seq_len": rho.seq_len,
        "embed_dim": rho.embed_dim,
        "x_range": list(x_range),
        "grid": grid,
        "rho": rho.get_parameters(),            # scheme, E (S,d,D), B (S,D), attention{...}
        "A": A.detach().cpu().tolist(),         # each (seq_len * embed_dim,)
        "A1": A1.detach().cpu().tolist(),
        "B": B.detach().cpu().tolist(),
        "B1": B1.detach().cpu().tolist(),
    }


def append_record(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def iter_records(path):
    """Yield every saved record (dict) from a JSONL file."""
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def rebuild_teacher(record):
    """(rho, A, A1, B, B1) rebuilt from a saved record (float32 tensors, CPU)."""
    rho = LinearFeatureAttention2D.from_parameters(record["rho"])
    A, A1, B, B1 = (torch.tensor(record[k], dtype=torch.float32) for k in ("A", "A1", "B", "B1"))
    return rho, A, A1, B, B1


def make_target_fn(record):
    """
    target_fn(u): (N, 2) -> (N, 1), the function the training pipeline expects
    (same convention as make_target_fn_attention: branch vectors as columns).
    """
    rho, A, A1, B, B1 = rebuild_teacher(record)
    A, A1, B, B1 = (v.reshape(-1, 1) for v in (A, A1, B, B1))

    def target_fn(u):
        with torch.no_grad():
            X = rho(u).reshape(u.shape[0], -1)       # (N, seq_len * embed_dim)
            return f(A, A1, B, B1, X)[0]             # (N, 1)

    return target_fn


# ============================================================
# Command line
# ============================================================

def ask_save():
    """Returns 'save', 'skip' or 'quit'."""
    while True:
        try:
            ans = input("Save this function? [y]es / [n]o / [q]uit the search: ").strip().lower()
        except EOFError:
            return "quit"
        if ans in ("y", "yes"):
            return "save"
        if ans in ("", "n", "no"):
            return "skip"
        if ans in ("q", "quit"):
            return "quit"
        print("Please answer y, n or q.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Random search for interesting 2D teacher functions.")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--scheme", choices=["mixed", "separate"], default="mixed")
    p.add_argument("--init-scale", type=float, default=3.0, help="std of E and B ")
    p.add_argument("--branch-range", type=float, default=3.0, help="A, A1, B, B1 ~ U(-r, r)")
    p.add_argument("--x-range", type=float, nargs=2, default=(-1.0, 1.0), metavar=("LOW", "HIGH"))
    p.add_argument("--grid", type=int, default=200, help="grid points per axis")
    p.add_argument("--region-frac", type=float, default=5e-4,
                   help="regions smaller than this fraction of the grid are ignored")
    p.add_argument("--kink-weight", type=float, default=0.25)
    p.add_argument("--min-regions", type=int, default=6)
    p.add_argument("--min-score", type=float, default=30.0)
    p.add_argument("--out", type=str, default="interesting_functions_attention_2d.jsonl")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    lo, hi = args.x_range
    xs = np.linspace(lo, hi, args.grid)
    ys = np.linspace(lo, hi, args.grid)
    GX, GY = np.meshgrid(xs, ys)                                   # (ny, nx); x varies along columns
    U = torch.tensor(np.stack([GX.ravel(), GY.ravel()], axis=1),
                     dtype=torch.float32, device=device)           # (ny*nx, 2)

    all_scores, n_shown, n_saved = [], 0, 0
    try:
        for trial in range(1, args.trials + 1):
            rho = LinearFeatureAttention2D(
                seq_len=args.seq_len, embed_dim=args.embed_dim, input_dim=2,
                init_scale=args.init_scale, scheme=args.scheme,
            ).to(device)
            out_dim = rho.output_dim
            r = args.branch_range
            A, A1, B, B1 = (torch.empty(out_dim, device=device).uniform_(-r, r) for _ in range(4))

            with torch.no_grad():
                feats = rho(U).reshape(U.shape[0], -1)             # (N, seq_len*embed_dim): flatten (S, D)
                assert feats.shape[1] == out_dim
                y, Ax, A1x, Bx, B1x = f(A, A1, B, B1, feats)

            shape = (len(ys), len(xs))
            arrs = [t.reshape(shape).cpu().numpy().astype(np.float64) for t in (y, Ax, A1x, Bx, B1x)]
            stats = score_function(*arrs, xs, ys, region_frac=args.region_frac,
                                   kink_weight=args.kink_weight)
            all_scores.append(stats["score"])

            if trial % 100 == 0:
                print(f"trial {trial}/{args.trials}   best score so far: {max(all_scores):.1f}")

            if stats["num_regions"] < args.min_regions or stats["score"] < args.min_score:
                continue

            n_shown += 1
            fig_3d, fig_labels = plot_candidate(xs, ys, arrs[0], stats, title=f"trial {trial}")

            plt.show(block=False)
            plt.pause(0.1)

            choice = ask_save()

            plt.close(fig_3d)
            plt.close(fig_labels)

            if choice == "save":
                append_record(args.out, make_record(rho, A, A1, B, B1, stats, (lo, hi), args.grid))
                n_saved += 1
                print(f"Saved to {args.out} ({n_saved} so far).")
            elif choice == "quit":
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")

    if all_scores:
        q = np.percentile(all_scores, [50, 90, 99])
        print(f"\n{len(all_scores)} trials, {n_shown} candidates shown, {n_saved} saved.")
        print(f"score quantiles: median {q[0]:.1f} | 90% {q[1]:.1f} | 99% {q[2]:.1f} | "
              f"max {max(all_scores):.1f}   (thresholds: min_regions={args.min_regions}, "
              f"min_score={args.min_score})")


if __name__ == "__main__":
    main()