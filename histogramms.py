'''
rank_histograms_2d.py

Histograms of the Jacobian ranks of a DataFrame of runs, one histogram per
filter, all in the same figure (stacked by default).

The DataFrame must have the columns
    run, scale, scheme, dataset, weight_path, is_random_sampling, hidden_dim, base_dir
(`run` is the run folder name inside `base_dir`; `dataset` is not used here:
the dataset path is read from each run's config.json by jacobian_ranks_2d).

    fig, ax, groups = plot_rank_histograms(
        df, "rank_full",
        filters=[(True, 20, "mixed", 3.0),        # (is_random_sampling, hidden_dim, scheme, scale)
                 (False, 20, "mixed", 3.0),
                 (None, 20, "separate", None)],   # None = no condition on that variable
    )

The three rank columns (rank_full, rank_rho, rank_rest) are added to `df`
in place by jacobian_ranks_2d.add_jacobian_ranks. Only the runs selected by
the filters and not computed yet are processed, so calling the function again
with other filters only computes what is missing (compute_all=True does every
row at once, recompute=True redoes the selected rows).
'''

from pathlib import Path
import matplotlib.pyplot as plt
from compute_ranks import add_jacobian_ranks, _read_df, _METHOD_SPECS
# Order of the values inside each filter tuple.
FILTER_COLUMNS = ("is_random_sampling", "hidden_dim", "scheme", "scale")
_SHORT_NAMES = {"is_random_sampling": "random", "hidden_dim": "hidden", "scheme": "scheme", "scale": "scale"}

RANK_COLUMNS = ("rank_full", "rank_rho", "rank_rest")
RANK_LABELS = {
    "rank_full": "rank of the Jacobian (all parameters)",
    "rank_rho": "rank of the partial Jacobian (rho parameters only)",
    "rank_rest": "rank of the partial Jacobian (all parameters but rho's)",
}
import numpy as np
import pandas as pd
# ============================================================
# Filters
# ============================================================

def _match(series, value):
    """Boolean mask: series == value (np.isclose for floats); value None = no condition."""
    if value is None:
        return pd.Series(True, index=series.index)
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if is_number and pd.api.types.is_float_dtype(series):
        return pd.Series(np.isclose(series.to_numpy(dtype=float), float(value)), index=series.index)
    return series == value


def select_runs(df, filt):
    """Boolean mask of the rows matching filt = (is_random_sampling, hidden_dim, scheme, scale); None = any."""
    if len(filt) != len(FILTER_COLUMNS):
        raise ValueError(f"each filter must be a tuple of {len(FILTER_COLUMNS)} values "
                         f"{FILTER_COLUMNS}, got {filt!r}")
    mask = pd.Series(True, index=df.index)
    for column, value in zip(FILTER_COLUMNS, filt):
        mask &= _match(df[column], value)
    return mask


def _filter_label(filt):
    parts = [f"{_SHORT_NAMES[c]}={v}" for c, v in zip(FILTER_COLUMNS, filt) if v is not None]
    return ", ".join(parts) if parts else "all runs"

# ============================================================
# Rank columns (computed by jacobian_ranks_2d)
# ============================================================

def _resolve_weights(weight_path, run_folder):
    """The path as given; if it does not exist and is relative, also try it inside the run folder."""
    p = Path(weight_path)
    if p.exists() or p.is_absolute():
        return p
    q = Path(run_folder) / p
    return q if q.exists() else p


def ensure_rank_columns(df, mask=None, method="float64", device="cpu", recompute=False):
    """
    Make sure df has the rank columns (rank_full, rank_rho, rank_rest, plus
    n_train and rank_error; "_<method>" suffix on the ranks if method != "float64")
    and fill them, in place, for the rows of `mask` (default: all rows) where
    they are still missing (or for every row of `mask` if recompute=True).

    Returns the names of the three rank columns.
    """
    if method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {method!r}")
    suffix = "" if method == "float64" else f"_{method}"
    stable = _METHOD_SPECS[method][2]
    cols = [c + suffix for c in RANK_COLUMNS]

    for c in cols:
        if c not in df.columns:
            df[c] = (pd.Series(np.nan, index=df.index, dtype="float64") if stable
                     else pd.Series(pd.NA, index=df.index, dtype="Int64"))
    if "n_train" not in df.columns:
        df["n_train"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "rank_error" not in df.columns:
        df["rank_error"] = pd.Series(None, index=df.index, dtype="object")

    needed = pd.Series(True, index=df.index) if mask is None else mask
    todo = needed & (df[cols[0]].isna() | recompute)
    if not todo.any():
        return cols

    rows = df.loc[todo]
    folders = [Path(b) / r for b, r in zip(rows["base_dir"], rows["run"])]
    sub = pd.DataFrame(index=rows.index)
    sub["run_folder"] = [str(f) for f in folders]
    sub["weights_path"] = [str(_resolve_weights(w, f)) for w, f in zip(rows["weight_path"], folders)]

    print(f"Computing Jacobian ranks ({method}) for {len(sub)} run(s)...")
    add_jacobian_ranks(sub, folder_col="run_folder", weights_col="weights_path",
                       method=method, device=device)

    for c in cols + ["n_train", "rank_error"]:
        df.loc[sub.index, c] = sub[c]
    return cols

# ============================================================
# Histograms
# ============================================================

def _bin_edges(arrays, bins):
    """Common bin edges. bins=None: unit-width bins centred on the integers for integer ranks."""
    values = np.concatenate([a for a in arrays if len(a)])
    if bins is None:
        if np.allclose(values, np.round(values)) and values.max() - values.min() <= 80:
            return np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
        return np.histogram_bin_edges(values, bins=30)
    if isinstance(bins, (int, np.integer)):
        return np.histogram_bin_edges(values, bins=int(bins))
    return np.asarray(bins, dtype=float)


def plot_rank_histograms(df, rank_col, filters, method="float64", device="cpu", bins=None,
                         stacked=True, recompute=False, compute_all=False, figsize=(9, 5.5)):
    """
    One histogram of `rank_col` per filter, in the same figure, one colour per filter.

    df        : runs DataFrame (see the module docstring); rank columns are added in place.
    rank_col  : "rank_full" (full Jacobian), "rank_rho" (rho parameters only)
                or "rank_rest" (all parameters but rho's).
    filters   : list of tuples (is_random_sampling, hidden_dim, scheme, scale);
                None in a position = no condition on that variable.
    stacked   : True -> stacked bars (one call to hist); False -> overlaid, semi-transparent.
    bins      : None (unit bins for integer ranks), an int, or explicit edges; common to all filters.

    The title gives the number of runs behind each histogram ("n" if all have a
    rank, "ok/n" if some failed); the legend repeats it per filter.

    Returns (fig, ax, groups): groups[i] is the sub-DataFrame matched by filters[i].
    """
    if rank_col not in RANK_COLUMNS:
        raise ValueError(f"rank_col must be one of {RANK_COLUMNS}, got {rank_col!r}")
    if not filters:
        raise ValueError("filters must be a non-empty list of tuples")

    masks = [select_runs(df, f) for f in filters]
    needed = pd.Series(np.logical_or.reduce([m.to_numpy() for m in masks]), index=df.index)
    cols = ensure_rank_columns(df, mask=None if compute_all else needed,
                               method=method, device=device, recompute=recompute)
    col = cols[RANK_COLUMNS.index(rank_col)]

    n_overlap = int((np.sum([m.to_numpy() for m in masks], axis=0) > 1).sum())
    if n_overlap:
        print(f"Warning: {n_overlap} run(s) match several filters and appear in several histograms.")

    arrays, n_match, n_ok = [], [], []
    for m in masks:
        values = df.loc[m, col].dropna().to_numpy(dtype=float)
        arrays.append(values)
        n_match.append(int(m.sum()))
        n_ok.append(len(values))
    for f, nm, no in zip(filters, n_match, n_ok):
        if nm == 0:
            print(f"Warning: no run matches the filter {f!r}.")
        elif no < nm:
            print(f"Warning: {nm - no}/{nm} run(s) matching {f!r} have no rank (failed rows).")

    if not any(n_ok):
        print("Nothing to plot: no selected run has a rank.")
        return None, None, [df.loc[m].copy() for m in masks]

    counts = [str(no) if no == nm else f"{no}/{nm}" for nm, no in zip(n_match, n_ok)]
    labels = [f"{_filter_label(f)}  (n={c})" for f, c in zip(filters, counts)]
    edges = _bin_edges(arrays, bins)
    cmap = plt.get_cmap("tab10" if len(filters) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(filters))]

    fig, ax = plt.subplots(figsize=figsize)
    if stacked:
        ax.hist(arrays, bins=edges, stacked=True, color=colors, label=labels,
                edgecolor="black", linewidth=0.5)
    else:
        for a, c, lab in zip(arrays, colors, labels):
            ax.hist(a, bins=edges, color=c, label=lab, alpha=0.55, edgecolor="black", linewidth=0.5)

    title = f"{RANK_LABELS[rank_col]}\nnumber of runs per histogram: " + " | ".join(counts)
    if stacked:
        title += f"   (total {sum(n_ok)})"
    ax.set_title(title)
    ax.set_xlabel(RANK_LABELS[rank_col] + ("" if method == "float64" else f" [{method}]"))
    ax.set_ylabel("number of runs")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig, ax, [df.loc[m].copy() for m in masks]

if __name__=="__main__":
    df=_read_df("randomsampling_h10_1112.csv")
    fig, ax, groups = plot_rank_histograms(
    df, "rank_full",                       # ou "rank_rho", "rank_rest"
    filters=[(True,  10, "uniform_bias",    1.0),   # (is_random_sampling, hidden_dim, scheme, scale)
             (True, 10, "uniform_bias",    4.5),
            ])
    plt.show()