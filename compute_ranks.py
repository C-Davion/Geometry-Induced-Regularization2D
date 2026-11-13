'''
jacobian_ranks_2d.py

For every row of a DataFrame (one column with the run folder, one with the
path to model_weights.pt):

  1. build the trained network with build_network_2d,
  2. load that run's training inputs X_train from the dataset .pt whose path
     is stored in the run's config.json,
  3. compute, from ONE Jacobian of theta -> f(theta, X_train) (float64 by
     default), the numerical rank of
        - the full Jacobian                           -> rank_full
        - the partial Jacobian w.r.t. rho's parameters  -> rank_rho
        - the partial Jacobian w.r.t. every parameter
          but rho's (W1, b1, W2, b2)                  -> rank_rest
  4. add them (plus n_train and rank_error) to the DataFrame.

The Jacobian helpers are the same as in the earlier partial-rank code
(_jacobian_blocks, _rank_of, _ranks_from_blocks, _default_dtype,
_METHOD_SPECS); they are repeated here so the script is self-contained.

Methods (as in compute_rank): "float64" (default, default tolerance),
"float32", "float32_rtol" (rtol = 2^-37), "stable_rank". For any method other
than "float64" the column names get a "_<method>" suffix, so several methods
can be added to the same DataFrame.

Usage (in a session):
    df = add_jacobian_ranks(df, folder_col="run_folder", weights_col="weights_path")
Usage (command line):
    python jacobian_ranks_2d.py --df runs.pkl --out runs_with_ranks.pkl
'''

import argparse
import copy
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, jacrev
from buildnetwork import build_network_2d, _load_config, DEFAULT_DATASET_PATH
RHO_PREFIX = "rho."      # named_parameters(): "rho.E", "rho.B", "rho.attention.*"

_METHOD_SPECS = {
    # method:        (dtype,          rtol,       stable)
    "float32":       (torch.float32,  None,       False),
    "float32_rtol":  (torch.float32,  2.0 ** -37, False),
    "float64":       (torch.float64,  None,       False),
    "stable_rank":   (torch.float64,  None,       True),
}

# ============================================================
# Jacobian ranks (same code as the earlier partial-rank module)
# ============================================================

@contextmanager
def _default_dtype(dtype):
    """
    Temporarily change torch's default dtype: LinearFeatureAttention.forward
    builds torch.ones(seq_len) in the default dtype, which would clash with a
    float64 input. (Not needed if that forward uses dtype=x.dtype.)
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _jacobian_blocks(model, X):
    """
    Stacked Jacobian of theta -> f(theta, X), one block per parameter:
    {param_name: np.ndarray of shape (N * output_dim, numel)}.
    (Dtype is whatever model / X are.)
    """
    params = dict(model.named_parameters())

    def f_batch(p: dict) -> torch.Tensor:
        return functional_call(model, p, X).flatten()      # (N * output_dim,)

    jac_dict = jacrev(f_batch)(params)
    return {
        name: j.flatten(start_dim=1).detach().cpu().numpy()
        for name, j in jac_dict.items()
    }


def _rank_of(mats, rtol=None, stable=False):
    """
    Rank of the horizontal concatenation of the given 2D arrays (0 if none).
      rtol=None   : default tolerance max(m, n) * eps(dtype) * sigma_max
                    (same as torch.linalg.matrix_rank's default)
      rtol=r      : singular values <= r * sigma_max count as zero
      stable=True : stable rank ||M||_F^2 / ||M||_2^2 (a float) instead
    """
    if not mats:
        return 0.0 if stable else 0
    M = np.concatenate(mats, axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0.0 if stable else 0
    if stable:
        return float((s ** 2).sum() / s[0] ** 2)
    if rtol is None:
        rtol = max(M.shape) * np.finfo(s.dtype).eps
    return int((s > rtol * s[0]).sum())


def _ranks_from_blocks(blocks, rho_prefix=RHO_PREFIX, rtol=None, stable=False):
    """
    (rank(J_rho), rank(J_rest), rank(J_full)) from per-parameter blocks:
    J_rho = blocks of parameters named "rho.*", J_rest = all the others.
    """
    rho = [b for name, b in blocks.items() if name.startswith(rho_prefix)]
    rest = [b for name, b in blocks.items() if not name.startswith(rho_prefix)]
    return (_rank_of(rho, rtol, stable),
            _rank_of(rest, rtol, stable),
            _rank_of(rho + rest, rtol, stable))


def jacobian_ranks(net, X, method="float64", device="cpu"):
    """
    Ranks of the full / rho-only / everything-but-rho Jacobian of `net` on X,
    all from the same Jacobian. Works on a deep copy: `net` is not modified.

    Returns {"rank_full": ..., "rank_rho": ..., "rank_rest": ...}
    (floats for "stable_rank", ints otherwise).
    """
    if method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {method!r}")
    dtype, rtol, stable = _METHOD_SPECS[method]

    with _default_dtype(dtype):
        model = copy.deepcopy(net).to(device=device, dtype=dtype)
        Xc = X.to(device=device, dtype=dtype)
        blocks = _jacobian_blocks(model, Xc)

    rank_rho, rank_rest, rank_full = _ranks_from_blocks(blocks, rtol=rtol, stable=stable)
    return {"rank_full": rank_full, "rank_rho": rank_rho, "rank_rest": rank_rest}

# ============================================================
# Training inputs
# ============================================================

def _resolve_dataset_path(dataset_path, run_dir):
    """The path as given; if it does not exist and is relative, also try it relative to the run folder."""
    p = Path(dataset_path)
    if p.exists() or p.is_absolute():
        return p
    q = Path(run_dir) / p
    return q if q.exists() else p


def _load_X_train(dataset_path, run_dir, cache):
    """X_train from the dataset .pt (cached: many runs share the same dataset)."""
    p = _resolve_dataset_path(dataset_path, run_dir)
    key = str(p)
    if key not in cache:
        data = torch.load(p, map_location="cpu", weights_only=True)
        cache[key] = data["X_train"]
    return cache[key]

# ============================================================
# DataFrame driver
# ============================================================

def add_jacobian_ranks(df, folder_col="run_folder", base_dir_col='base_dir', weights_col="weights_path",
                       method="float64", device="cpu", dataset_key="dataset_path",
                       default_dataset_path=DEFAULT_DATASET_PATH, verbose_every=25):
    """
    Add Jacobian-rank columns to `df` (in place; `df` is also returned):

        rank_full, rank_rho, rank_rest   (suffix "_<method>" if method != "float64")
        n_train                          number of training points the ranks were computed on
        rank_error                       None if the row went fine, else the error message

    A row that fails (missing file, bad checkpoint, ...) gets missing values
    and its error message in rank_error; the loop goes on.

    folder_col  : column with the run folder (config.json lives there)
    weights_col : column with the path to model_weights.pt
    dataset_key : key of config.json holding the dataset path
                  (default_dataset_path is used if the key is absent)
    device      : "cpu" by default (a float64 Jacobian of this size is cheap on CPU)
    """
    if method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {method!r}")
    suffix = "" if method == "float64" else f"_{method}"
    stable = _METHOD_SPECS[method][2]

    cache, results, n_rows, t0 = {}, [], len(df), time.time()
    for k, (_, row) in enumerate(df.iterrows(), start=1):
        base_dir= Path(row[base_dir_col])
        run_dir= base_dir / row[folder_col]
        try:
            config = _load_config(run_dir)
            dataset_path = config.get(dataset_key, default_dataset_path)
            X_train = _load_X_train(dataset_path, run_dir, cache)

            net = build_network_2d(base_dir / row[weights_col], scheme=config.get("scheme"), device=device)
            res = jacobian_ranks(net, X_train, method=method, device=device)
            res.update(n_train=int(X_train.shape[0]), rank_error=None)
        except Exception as e:                                   # keep going on a bad row
            print(f"[{k}/{n_rows}] {run_dir.name}: FAILED ({type(e).__name__}: {e})")
            res = {"rank_full": None, "rank_rho": None, "rank_rest": None,
                   "n_train": None, "rank_error": f"{type(e).__name__}: {e}"}
        results.append(res)

        if verbose_every and k % verbose_every == 0:
            print(f"[{k}/{n_rows}] {time.time() - t0:.0f}s elapsed")

    for name in ("rank_full", "rank_rho", "rank_rest"):
        values = [r[name] for r in results]
        df[name + suffix] = (pd.Series(values, index=df.index, dtype="float64") if stable
                             else pd.Series(values, index=df.index, dtype="Int64"))
    df["n_train"] = pd.Series([r["n_train"] for r in results], index=df.index, dtype="Int64")
    df["rank_error"] = pd.Series([r["rank_error"] for r in results], index=df.index, dtype="object")

    n_failed = sum(r["rank_error"] is not None for r in results)
    print(f"Done: {n_rows - n_failed}/{n_rows} rows ok, {n_failed} failed, "
          f"{time.time() - t0:.0f}s.")
    return df

# ============================================================
# Command line
# ============================================================

def _read_df(path):
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_pickle(path)


def _write_df(df, path):
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Add Jacobian-rank columns to a DataFrame of runs.")
    p.add_argument("--df", required=True, help="DataFrame file (.pkl or .csv)")
    p.add_argument("--out", default=None, help="output file (default: <df>_with_ranks.<ext>)")
    p.add_argument("--folder-col", default="run_folder")
    p.add_argument("--weights-col", default="weights_path")
    p.add_argument("--method", choices=list(_METHOD_SPECS), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset-key", default="dataset_path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = _read_df(args.df)
    df = add_jacobian_ranks(df, folder_col=args.folder_col, weights_col=args.weights_col,
                            method=args.method, device=args.device, dataset_key=args.dataset_key)
    src = Path(args.df)
    out = args.out or str(src.with_name(f"{src.stem}_with_ranks{src.suffix}"))
    _write_df(df, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()