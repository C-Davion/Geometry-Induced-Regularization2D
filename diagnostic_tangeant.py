"""
Jacobian of  theta -> model(theta, X)  (R^p -> R^n)  at the parameters of model_weights.pt,
plus a sanity check on a randomly drawn sample x of X.

CLI example:
    python jacobian_study.py --folder path/to/run --keys full --device cuda --show

Conventions
-----------
* `keys` are keys of model.state_dict() that are *trainable parameters* (buffers such as
  BatchNorm running stats are rejected). The string "full" means all parameters.
* The columns of the Jacobian follow the order of `keys`, each tensor being flattened in
  row-major order (use `param_layout` to get the column range of each key).
* The model is assumed to return ONE scalar per data point, so J has shape (n, p).

The functions load_model_and_data / compute_jacobian / run_check / make_plot are reused
by batch_study.py.
"""
import argparse
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union


import torch
from torch.func import functional_call, grad, jacrev, vmap

from buildnetwork import build_network_2d, DEFAULT_DATASET_PATH
from helper import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from compute_ranks import jacobian

Keys = Union[str, Sequence[str]]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def resolve_keys(model: torch.nn.Module, keys: Keys = "full") -> List[str]:
    """Turn the user's `keys` argument into a validated list of parameter names."""
    named = dict(model.named_parameters())
    if keys is None or keys == "full" or list(keys) == ["full"]:
        return list(named.keys())
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    for k in keys:
        if k not in named:
            if k in model.state_dict():
                raise ValueError(f"'{k}' is in the state_dict but is a buffer, not a trainable parameter.")
            raise KeyError(f"'{k}' not found. Available parameters: {list(named.keys())}")
    return keys


def param_layout(model: torch.nn.Module, keys: Keys = "full") -> Dict[str, Tuple[int, int]]:
    """Column range [start, stop) occupied by each key in the Jacobian / flattened gradient."""
    named = dict(model.named_parameters())
    layout, offset = {}, 0
    for k in resolve_keys(model, keys):
        layout[k] = (offset, offset + named[k].numel())
        offset += named[k].numel()
    return layout


def _flat(tensors: Dict[str, torch.Tensor], keys: List[str], lead_dims: int = 0) -> torch.Tensor:
    """Concatenate tensors[k] (flattening all dims after the first `lead_dims`)."""
    return torch.cat([tensors[k].reshape(*tensors[k].shape[:lead_dims], -1) for k in keys], dim=lead_dims)


def load_model_and_data(folder: str, dataset_folder: str, device: str = "cpu", float32: bool = False):
    """Load the model from <folder>/model_weights.pt and the dataset; cast to float64 by default."""
    model = build_network_2d(os.path.join(folder, "model_weights.pt")).to(device)
    (X_train,_,_,_),_=load_dataset(dataset_folder)
    X = torch.as_tensor(X_train).to(device)
    if not float32:
        model = model.double()
        if X.is_floating_point():
            X = X.double()
    model.eval()
    return model, X


# ----------------------------------------------------------------------------------------
# 1) The Jacobian
# ----------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------
# 2) Debug / accuracy tools
# ----------------------------------------------------------------------------------------
def gradient_at_sample(model: torch.nn.Module, x: torch.Tensor, keys: Keys = "full") -> torch.Tensor:
    """
    Gradient of theta -> model(theta, x) (x of shape (1, ...)) computed with classic autograd,
    i.e. through a code path independent from compute_jacobian. Shape (p_selected,).
    """
    model.eval()
    keys = resolve_keys(model, keys)
    named = dict(model.named_parameters())
    out = model(x)
    if out.numel() != 1:
        raise ValueError(f"Expected a scalar output, got shape {tuple(out.shape)}.")
    g = torch.autograd.grad(out.reshape(()), [named[k] for k in keys])
    return torch.cat([gi.reshape(-1) for gi in g]).detach()


@torch.no_grad()
def model_along_direction(
    model: torch.nn.Module,
    x: torch.Tensor,
    direction: torch.Tensor,
    ts: Sequence[float],
    keys: Keys = "full",
) -> np.ndarray:
    """t -> model(theta + t * direction, x) for each t in ts (direction lives on `keys` only)."""
    model.eval()
    keys = resolve_keys(model, keys)
    named = {k: p.detach() for k, p in model.named_parameters()}
    layout = param_layout(model, keys)
    vals = []
    for t in ts:
        pert = {k: named[k] + float(t) * direction[a:b].view_as(named[k]) for k, (a, b) in layout.items()}
        vals.append(functional_call(model, pert, (x,)).reshape(-1)[0].item())
    return np.array(vals)


def run_check(
    model: torch.nn.Module,
    X: torch.Tensor,
    keys: Keys,
    idx: int,
    J: Optional[torch.Tensor] = None,
    t_max: Optional[float] = None,
    n_points: int = 201,
) -> Tuple[dict, Optional[dict]]:
    """
    Check on sample X[idx]:
      * gradient vs. row idx of J (if J is given),
      * ||grad||^2 vs. a central finite difference of t -> f(theta + t*grad, x) at t = 0,
      * data for the plot.
    Returns (metrics dict, plot-data dict or None if the gradient is exactly zero).
    """
    keys = resolve_keys(model, keys)
    x = X[idx:idx + 1]
    g = gradient_at_sample(model, x, keys)
    g_norm2 = (g @ g).item()

    metrics = {"sample_idx": idx, "grad_norm2": g_norm2,
               "finite_diff": float("nan"), "rel_diff": float("nan")}
    if J is not None:
        row = J[idx]
        metrics["row_max_abs_err"] = (g - row).abs().max().item()
        metrics["row_rel_err"] = ((g - row).norm() / (row.norm() + 1e-30)).item()

    if g_norm2 == 0:  # e.g. dead ReLUs: nothing to differentiate along
        return metrics, None

    g_norm = g_norm2 ** 0.5
    theta_sel = torch.cat([dict(model.named_parameters())[k].detach().reshape(-1) for k in keys])
    if t_max is None:
        t_max = 0.001 * theta_sel.norm().item() / g_norm  # parameter displacement = 5% of ||theta||
    ts = np.linspace(-t_max, t_max, n_points)

    f_t = model_along_direction(model, x, g, ts, keys)           # f(theta + t*grad, x)
    f0 = model_along_direction(model, x, g, [0.0], keys)[0]      # f(theta, x)
    tangent = f0 + ts * g_norm2                                  # f(theta, x) + t ||grad||^2

    h = 1e-5 / g_norm
    fd = (model_along_direction(model, x, g, [h], keys)[0]
          - model_along_direction(model, x, g, [-h], keys)[0]) / (2 * h)
    metrics["finite_diff"] = float(fd)
    metrics["rel_diff"] = abs(fd - g_norm2) / g_norm2

    plot_data = {"ts": ts, "f_t": f_t, "tangent": tangent, "f0": f0, "idx": idx, "keys": keys}
    return metrics, plot_data


def make_plot(plot_data: dict, title_suffix: str = ""):
    """Figure with t -> f(theta + t*grad, x) and its tangent t -> f(theta, x) + t||grad||^2."""
    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(plot_data["ts"], plot_data["f_t"],
             label=r"$t\mapsto f(\theta + t\,\nabla_\theta f(\theta,x),\,x)$")
    plt.plot(plot_data["ts"], plot_data["tangent"], "--",
             label=r"$t\mapsto f(\theta,x) + t\,\|\nabla_\theta f(\theta,x)\|^2$")
    #plt.scatter([0], [plot_data["f0"]], color="k", zorder=3)
    plt.xlabel("t"); plt.ylabel("model output")
    plt.title(f"Sample #{plot_data['idx']}{title_suffix}", fontsize=9)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------
# CLI (single folder)
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder containing model_weights.pt")
    ap.add_argument("--dataset_folder",required=False,help="folder containing the dataset.pt",default=DEFAULT_DATASET_PATH)
    ap.add_argument("--keys", nargs="+", default=["full"], help="state_dict keys, or 'full'")
    ap.add_argument("--jac_method", default="batched", choices=["batched", "per_sample"])
    ap.add_argument("--chunk_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_max", type=float, default=None)
    ap.add_argument("--n_points", type=int, default=201)
    ap.add_argument("--float32", action="store_true", help="keep float32 (default: cast to float64)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="jacobian_check.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model, X = load_model_and_data(args.folder, args.dataset_folder, args.device, args.float32)
    keys = resolve_keys(model, args.keys)
    n = X.shape[0]
    print(f"n = {n} data points | selected keys: {keys}")

    #J = compute_jacobian(model, X, keys, method=args.method, chunk_size=args.chunk_size)
    J=jacobian(model,X,keys,jac_method=args.jac_method,chunk_size=args.chunk_size,copy_model=False)['J']
    print(f"Jacobian shape: {tuple(J.shape)}  (n x p_selected)")
    print("column layout:", param_layout(model, keys))

    idx = random.Random(args.seed).randrange(n)
    metrics, plot_data = run_check(model, X, keys, idx, J, args.t_max, args.n_points)
    print(f"\nSample index {idx}")
    print(f"max |grad - J[idx]| = {metrics['row_max_abs_err']:.3e}   "
          f"(relative L2 error {metrics['row_rel_err']:.3e})")
    print(f"||grad||^2 = {metrics['grad_norm2']:.6e} | finite-difference d/dt at 0 = "
          f"{metrics['finite_diff']:.6e} | rel. diff = {metrics['rel_diff']:.2e}")

    if plot_data is None:
        print("Gradient is exactly zero at this sample: nothing to plot.")
        return
    fig = make_plot(plot_data, f", keys = {'full' if args.keys == ['full'] else keys}")
    fig.savefig(args.out, dpi=150)
    print(f"Figure saved to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()