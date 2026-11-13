'''
diagnosis_2d.py

Diagnose one trained run.

    run = load_run("path/to/run_folder")                    # reads config.json, builds the network, loads X_train
    list_parameters(run)                                    # names / shapes of the parameters
    res = diagnose_jacobian(run, "W1")                      # Jacobian w.r.t. W1 only
    res = diagnose_jacobian(run, ["rho.attention.in_proj_weight", "b1"])
    res = diagnose_jacobian(run, "attention")               # every parameter of the attention module

load_run
    - reads the dataset path from <run_folder>/config.json (key "dataset_path";
      DEFAULT_DATASET_PATH from build_network_2d if absent) and prints it,
    - builds the network from <run_folder>/model_weights.pt (build_network_2d),
    - loads X_train (and Y_train) from the dataset file.

diagnose_jacobian
    Jacobian of theta -> [f(theta, x_1); ...; f(theta, x_N)] (N = number of
    training points, rows ordered (sample, output)) with respect ONLY to the
    selected parameters, all the others being held fixed. It prints
      - the selected parameters and the shape of the Jacobian,
      - its numerical rank, the tolerance used, the singular values around
        the cutoff and at both ends, and the stable rank,
      - the rank of each parameter's block,
      - a readable corner of the matrix (rows = samples with their (x, y),
        columns = a few entries of each selected parameter),
    and returns everything in a dict.

Parameter names are those of net.named_parameters() / the state dict:
    W1, b1, W2, b2, rho.E, rho.B,
    rho.attention.in_proj_weight, rho.attention.in_proj_bias,
    rho.attention.out_proj.weight, rho.attention.out_proj.bias
and are matched case-insensitively, with or without the leading "rho.", or by
suffix ("in_proj_weight"), or as a group ("rho", "attention").
'''

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch.func import functional_call, jacrev

from buildnetwork import build_network_2d, read_dataset_path, DEFAULT_DATASET_PATH
from compute_ranks import _default_dtype, _resolve_dataset_path
import numpy as np
import pandas as pd
# ============================================================
# Loading a run
# ============================================================

@dataclass
class Run:
    run_dir: Path
    config: dict
    dataset_path: Optional[str]
    net: Any
    X_train: Any = None
    Y_train: Any = None


def load_run(run_dir, default_dataset_path=DEFAULT_DATASET_PATH, dataset_key="dataset_path",
             weights_file="model_weights.pt", device="cpu") -> Run:
    """
    Read the dataset path from <run_dir>/config.json (default if absent) and
    print it, build the network from <run_dir>/<weights_file>, and load
    X_train / Y_train from the dataset file. If the dataset file cannot be
    found, X_train stays None (a warning is printed): the network is still
    built, and diagnose_jacobian then needs an explicit X.
    """
    run_dir = Path(run_dir)
    dataset_path, config = read_dataset_path(run_dir, default_dataset_path, dataset_key)

    net = build_network_2d(run_dir / weights_file, scheme=config.get("scheme"), device=device)

    X_train = Y_train = None
    resolved = _resolve_dataset_path(dataset_path, run_dir) if dataset_path is not None else None
    if resolved is not None and Path(resolved).exists():
        data = torch.load(resolved, map_location="cpu", weights_only=True)
        X_train, Y_train = data["X_train"], data.get("Y_train")
        print(f"Loaded {len(X_train)} training points from the dataset.")
    else:
        print("Warning: dataset file not found, X_train not loaded "
              "(pass X=... to diagnose_jacobian).")

    return Run(run_dir=run_dir, config=config,
               dataset_path=None if resolved is None else str(resolved),
               net=net, X_train=X_train, Y_train=Y_train)


def list_parameters(run_or_net):
    """Print the name, shape and number of entries of every parameter."""
    net = getattr(run_or_net, "net", run_or_net)
    total = 0
    print(f"{'name':<34}{'shape':<18}{'numel':>8}")
    for name, p in net.named_parameters():
        n = int(np.prod(tuple(p.shape)))
        total += n
        print(f"{name:<34}{str(tuple(p.shape)):<18}{n:>8}")
    print(f"{'total':<52}{total:>8}")

# ============================================================
# Parameter names
# ============================================================

def resolve_param_names(net, names: Union[str, Sequence[str]]):
    """
    Map user-given names to the exact names of net.named_parameters().

    A name is matched, in this order, (1) exactly (case-insensitive),
    (2) as the end of a dotted name ("E" -> "rho.E", "in_proj_weight",
    "out_proj.weight"), (3) as the start of dotted names, i.e. a group
    ("rho", "attention" -> every parameter of that module).
    Raises ValueError, listing the available names, if nothing matches.
    """
    if isinstance(names, str):
        names = [names]
    available = [n for n, _ in net.named_parameters()]

    resolved = []
    for name in names:
        key = name.lower()
        matches = [n for n in available if n.lower() == key]
        if not matches:
            matches = [n for n in available if n.lower().endswith("." + key)]
        if not matches:
            prefixes = (key + ".", "rho." + key + ".")
            matches = [n for n in available if n.lower().startswith(prefixes)]
        if not matches:
            raise ValueError(f"Unknown parameter name {name!r}. Available: {available}")
        for m in matches:
            if m not in resolved:
                resolved.append(m)
    return resolved

# ============================================================
# Jacobian w.r.t. selected parameters
# ============================================================

def jacobian_blocks_wrt(net, X, names, dtype=torch.float64, device="cpu"):
    """
    Jacobian of theta -> f(theta, X).flatten() w.r.t. the parameters in `names`
    only (all the other parameters are fixed), computed in `dtype` on a deep
    copy of `net`.

    Returns (blocks, shapes): blocks[name] is a numpy array of shape
    (N * output_dim, numel(name)); shapes[name] is the parameter's shape.
    """
    with _default_dtype(dtype):
        model = copy.deepcopy(net).to(device=device, dtype=dtype)
        Xc = X.to(device=device, dtype=dtype)

        all_params = dict(model.named_parameters())
        selected = {n: all_params[n] for n in names}
        fixed = {n: p.detach() for n, p in all_params.items() if n not in selected}

        def f(sel: dict) -> torch.Tensor:
            return functional_call(model, {**fixed, **sel}, Xc).flatten()     # (N * output_dim,)

        jac = jacrev(f)(selected)

    blocks = {n: jac[n].flatten(start_dim=1).detach().cpu().numpy() for n in names}
    shapes = {n: tuple(all_params[n].shape) for n in names}
    return blocks, shapes


def summarize_rank(J, rtol=None):
    """
    Numerical rank of J from its singular values: number of s > rtol * s_max,
    with the default rtol = max(m, n) * eps(dtype) (same as torch / numpy).
    Returns a dict (rank, singular_values, rtol, tol, stable_rank).
    """
    s = np.linalg.svd(J, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return {"rank": 0, "singular_values": s, "rtol": rtol, "tol": 0.0, "stable_rank": 0.0}
    rel = max(J.shape) * np.finfo(s.dtype).eps if rtol is None else rtol
    tol = rel * s[0]
    return {"rank": int((s > tol).sum()), "singular_values": s, "rtol": rel, "tol": tol,
            "stable_rank": float((s ** 2).sum() / s[0] ** 2)}

# ============================================================
# Printing
# ============================================================

def _fmt(v):
    return f"{v:.3e}"


def _print_rank_report(res, shape):
    s, r = res["singular_values"], res["rank"]
    print(f"\nRank: {r}   (matrix {shape[0]} x {shape[1]}, maximum possible {min(shape)})")
    if s.size == 0 or s[0] == 0:
        print("  the Jacobian is identically zero")
        return
    print(f"  tolerance: {res['rtol']:.3e} x sigma_max = {res['tol']:.3e}   "
          f"(singular values <= tolerance count as zero)")
    print(f"  stable rank ||J||_F^2 / ||J||_2^2 = {res['stable_rank']:.3f}")

    def show(label, lo, hi):
        lo, hi = max(lo, 0), min(hi, len(s))
        if lo < hi:
            print(f"  {label:<18}" + "  ".join(f"s{i + 1}={_fmt(s[i])}" for i in range(lo, hi)))

    show("largest:", 0, 3)
    if 0 < r < len(s):
        show("around the cutoff:", r - 2, r + 2)
        print(f"  gap s{r}/s{r + 1} = {s[r - 1] / s[r]:.3e}")
    show("smallest:", len(s) - 3, len(s))


def _corner_table(blocks, shapes, X, max_rows, cols_per_block):
    """DataFrame: first rows of the Jacobian with the sample coordinates, a few columns per parameter."""
    X_np = X.detach().cpu().numpy()
    n_samples, d = X_np.shape
    n_rows = next(iter(blocks.values())).shape[0]
    out_dim = n_rows // n_samples
    n_show = n_rows if max_rows is None else min(max_rows, n_rows)

    coord_names = ["x", "y"] if d == 2 else [f"x{k}" for k in range(d)]
    sample = np.arange(n_show) // out_dim
    data = {c: X_np[sample, k] for k, c in enumerate(coord_names)}

    for name, B in blocks.items():
        n_cols = B.shape[1] if cols_per_block is None else min(cols_per_block, B.shape[1])
        for j in range(n_cols):
            idx = ",".join(str(i) for i in np.unravel_index(j, shapes[name]))
            data[f"{name}[{idx}]"] = B[:n_show, j]

    index = [str(i) if out_dim == 1 else f"{i // out_dim}.{i % out_dim}" for i in range(n_show)]
    return pd.DataFrame(data, index=index)

# ============================================================
# Main entry point
# ============================================================

def diagnose_jacobian(run, names, X=None, dtype=torch.float64, device="cpu", rtol=None,
                      max_rows=8, cols_per_block=4, verbose=True):
    """
    Jacobian of the network w.r.t. only the parameter(s) `names` (see the
    module docstring for the accepted names), printed in a readable form,
    with its rank.

    run            : a Run from load_run (or any object with .net and .X_train)
    names          : "W1", "rho.E", "attention", ... or a list of such names
    X              : points where the Jacobian is evaluated (default run.X_train)
    dtype          : torch.float64 (default) or torch.float32
    rtol           : relative tolerance for the rank (default max(m, n) * eps)
    max_rows,
    cols_per_block : size of the printed corner of the matrix (None = everything)

    Returns a dict: names, J (numpy, rows ordered (sample, output)), blocks,
    shapes, rank, block_ranks, singular_values, rtol, tol, stable_rank.
    """
    X = run.X_train if X is None else X
    if X is None:
        raise ValueError("No training inputs available: pass X=... (a float tensor of shape (N, 2)).")
    net = run.net

    resolved = resolve_param_names(net, names)
    blocks, shapes = jacobian_blocks_wrt(net, X, resolved, dtype=dtype, device=device)
    J = np.concatenate([blocks[n] for n in resolved], axis=1)

    total = summarize_rank(J, rtol=rtol)
    block_ranks = {n: summarize_rank(blocks[n], rtol=rtol)["rank"] for n in resolved}

    if verbose:
        dtype_name = str(dtype).replace("torch.", "")
        print(f"\nJacobian of f(theta, X) w.r.t. {', '.join(resolved)}   "
              f"[{dtype_name}, {len(X)} points, other parameters fixed]")
        for n in resolved:
            numel = blocks[n].shape[1]
            print(f"  {n:<34}shape {str(shapes[n]):<14}{numel:>7} columns   block rank {block_ranks[n]}")
        _print_rank_report(total, J.shape)
        print(f"\nTop-left corner (rows: sample 'i' or 'i.output' with its coordinates; "
              f"columns: first entries of each parameter):")
        print(_corner_table(blocks, shapes, X, max_rows, cols_per_block)
              .to_string(float_format=lambda v: f"{v:.4g}"))

    return {"names": resolved, "J": J, "blocks": blocks, "shapes": shapes,
            "rank": total["rank"], "block_ranks": block_ranks,
            "singular_values": total["singular_values"], "rtol": total["rtol"],
            "tol": total["tol"], "stable_rank": total["stable_rank"]}