'''
build_network_2d.py

Rebuild a trained network from the model_weights.pt written by the training
script, for the 2D case:

    u = (x, y)  ->  rho(u)  ->  W2 ReLU(W1 rho(u) + b1) + b2

where rho is the 2D single-head attention feature map (E of shape
(seq_len, input_dim, embed_dim)) and the rest is the usual
OneHiddenLayerReLUNet.

Same idea as expstats_attention.build_network, but E is now 3D, so
seq_len, input_dim and embed_dim are all read off the saved rho.E tensor.

One thing the checkpoint does NOT contain: the token scheme ("mixed" /
"separate") of the 2D feature map. Its mask is a non-persistent buffer, so
it is not in the state_dict, yet it changes the function for "separate"
runs (the masked entries of E are never trained, but they would be used by
a "mixed" rebuild). It is therefore taken from the `scheme` argument or,
if None, from the "scheme" key of the config.json next to the weights.

Once the network is built, plot_network_2d(net, run_dir) prints the dataset
path stored in the run's config.json and draws the network output as a 3D
surface, one colour per distinct activation pattern.
'''

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Adjust to wherever your 2D class lives and to its actual name.
# Assumed constructor: (seq_len, embed_dim, input_dim, scheme=...), as in
# teacher_function_2d.LinearFeatureAttention.
from relu_network import OneHiddenLayerReLUNet, LinearFeatureAttention2D
from helper import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

_SCHEMES = ("mixed", "separate")

# Used when the run's config.json has no dataset path. Edit to your own path.
DEFAULT_DATASET_PATH = "generated_datasets\datasets\score_11.36_function_0000_.pt"


def _resolve_scheme(scheme: Optional[str], weights_path: Path) -> str:
    """`scheme` if given, else config.json["scheme"] next to the weights, else "mixed" (with a warning)."""
    if scheme is None:
        config_file = weights_path.parent / "config.json"
        if config_file.exists():
            with open(config_file) as fh:
                scheme = json.load(fh).get("scheme")
        if scheme is None:
            print(f"Warning: no scheme given and none found in {config_file}; assuming 'mixed'. "
                  "Pass scheme= explicitly if this run used 'separate'.")
            scheme = "mixed"
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
    return scheme


def build_network_2d(weights_path, scheme: Optional[str] = None, device=DEVICE,
                     rho_cls=LinearFeatureAttention2D) -> OneHiddenLayerReLUNet:
    """
    Load model_weights.pt and return the trained network, in eval mode, on `device`.

    The returned object is an nn.Module, hence callable: net(X) with X a
    float32 tensor of shape (N, input_dim) on the same device gives (N, output_dim).
    (See as_callable for a version that takes numpy arrays of any leading shape.)

    Shapes are inferred from the checkpoint itself:
        W1: (hidden_dim, feature_dim)        rho.E: (seq_len, input_dim, embed_dim)
    """
    weights_path = Path(weights_path)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    W1, b1 = state_dict["W1"], state_dict["b1"]
    W2, b2 = state_dict["W2"], state_dict["b2"]

    E_shape = tuple(state_dict["rho.E"].shape)
    if len(E_shape) != 3:
        raise ValueError(
            f"rho.E has shape {E_shape}, expected (seq_len, input_dim, embed_dim): "
            "this checkpoint was not trained with the 2D feature map."
        )
    seq_len, input_dim, embed_dim = E_shape

    feature_dim = W1.shape[1]
    if seq_len * embed_dim != feature_dim:
        raise ValueError(
            f"rho.E shape {E_shape} gives seq_len*embed_dim = {seq_len * embed_dim}, "
            f"but W1 expects feature_dim = {feature_dim}."
        )

    scheme = _resolve_scheme(scheme, weights_path)

    rho = rho_cls(seq_len=seq_len, embed_dim=embed_dim, input_dim=input_dim,
                  scheme=scheme).to(device, dtype=DTYPE)
    net = OneHiddenLayerReLUNet(W1, b1, W2, b2, rho=rho).to(device, dtype=DTYPE)

    # W1/b1/W2/b2 are already set by the constructor call; this also fills in
    # rho's weights (E, B, attention.*).
    net.load_state_dict(state_dict)
    net.eval()
    return net


def as_callable(net):
    """
    Wrap a network as a plain function on arrays.

        fn = as_callable(build_network_2d(path))
        z = fn(u)       # u: array-like (..., 2)  ->  numpy array (..., output_dim)

    Handles dtype / device, no_grad, and any leading shape (e.g. an
    (ny, nx, 2) grid of points gives an (ny, nx, output_dim) array).
    """
    device = next(net.parameters()).device

    def fn(u):
        u = torch.as_tensor(u, dtype=DTYPE, device=device)
        lead = tuple(u.shape[:-1])
        with torch.no_grad():
            out = net(u.reshape(-1, u.shape[-1]))
        return out.reshape(*lead, -1).cpu().numpy()

    return fn


# ============================================================
# Config / dataset path
# ============================================================

def _load_config(run_dir) -> dict:
    config_file = Path(run_dir) / "config.json"
    if not config_file.exists():
        print(f"Warning: {config_file} not found, using defaults.")
        return {}
    with open(config_file) as fh:
        return json.load(fh)


def read_dataset_path(run_dir, default=DEFAULT_DATASET_PATH, key="dataset_path"):
    """
    Read the dataset path from <run_dir>/config.json[key] (falling back to
    `default` if the key is absent), print it, and say if the file exists.

    Returns (dataset_path, config).
    """
    config = _load_config(run_dir)
    dataset_path, source = config.get(key), f"config.json['{key}']"
    if dataset_path is None:
        dataset_path, source = default, "default"

    print(f"Dataset path ({source}): {dataset_path}")
    if dataset_path is not None and not Path(dataset_path).exists():
        print("  -> file not found")
    return dataset_path, config


def _resolve_x_range(x_range, config, dataset_path):
    """x_range argument, else config['x_range'], else the 'x_range' stored in the dataset file."""
    if x_range is not None:
        return tuple(x_range)
    if "x_range" in config:
        return tuple(config["x_range"])
    if dataset_path is not None and Path(dataset_path).exists():
        data = torch.load(dataset_path, map_location="cpu", weights_only=True)
        if "x_range" in data:
            return tuple(data["x_range"])
    raise ValueError("x_range not found: pass x_range=(low, high), or add it to config.json.")


# ============================================================
# 3D surface coloured by activation pattern
# ============================================================

def _distinct_colors(n):
    """(n, 3) RGB colours; golden-ratio hue steps, so consecutive ids are well separated."""
    i = np.arange(n)
    hue = (i * 0.6180339887) % 1.0
    sat = 0.60 + 0.30 * (i % 2)
    val = 0.95 - 0.20 * ((i // 2) % 2)
    return hsv_to_rgb(np.stack([hue, sat, val], axis=1))


def plot_pattern_surface(net, x_range, grid=120, output_index=0, ax=None, scatter_training=False,dataset_path=DEFAULT_DATASET_PATH,
                         elev=30, azim=-60, title=None):
    """
    3D surface of the network output over [lo, hi] x [lo, hi] (grid x grid
    points). Every distinct activation pattern (which hidden units are on)
    gets its own colour, flat-shaded so the colour identifies the pattern.

    A facet takes the pattern of its first corner vertex (matplotlib's
    convention for `facecolors`), i.e. it can be off by one grid cell.

    Returns (fig, ax, pattern_ids) with pattern_ids of shape (grid, grid):
    the integer id of the pattern at each grid point.
    """
    device = next(net.parameters()).device
    lo, hi = x_range
    xs = np.linspace(lo, hi, grid)
    GX, GY = np.meshgrid(xs, xs)                                   # (ny, nx), x along columns
    U = torch.as_tensor(np.stack([GX.ravel(), GY.ravel()], axis=1), dtype=DTYPE, device=device)

    with torch.no_grad():
        Z = net(U)[:, output_index]
        pattern = net.activation_pattern(U)                        # (N, hidden_dim), 0/1

    Z = Z.cpu().numpy().reshape(GX.shape)
    pattern = (pattern > 0).cpu().numpy().astype(np.uint8)

    _, ids = np.unique(pattern, axis=0, return_inverse=True)
    ids = np.asarray(ids).reshape(GX.shape)
    n_patterns = int(ids.max()) + 1

    facecolors = np.concatenate(
        [_distinct_colors(n_patterns)[ids], np.ones(ids.shape + (1,))], axis=-1
    )                                                              # (ny, nx, 4)

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(GX, GY, Z, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    if scatter_training:
        offset=0.05
        datas,_=load_dataset(dataset_path)
        X_train, Y_train,_,_=datas
        X_train_np = X_train.cpu().numpy()
        Y_train_np = Y_train.cpu().numpy().ravel()
        ax.scatter(X_train_np[:, 0],X_train_np[:, 1],Y_train_np+offset,color="black",s=5,alpha=0.55,label="Training set")
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("network output")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or f"Network output: {n_patterns} distinct activation patterns "
                          "(one colour each)")
    return fig, ax, ids


def plot_network_2d(net, run_dir, grid=120, x_range=None, output_index=0,
                    default_dataset_path=DEFAULT_DATASET_PATH, scatter_training=False, dataset_key="dataset_path",
                    **plot_kwargs):
    """
    Once the network is built (build_network_2d):
      1. read the dataset path from <run_dir>/config.json (default if absent) and print it,
      2. plot the network output surface coloured by activation pattern.

    x_range: (low, high) used on both axes; default config['x_range'], else the
    value stored in the dataset file. Extra keyword arguments go to
    plot_pattern_surface (ax, elev, azim, title).

    Returns (fig, ax, pattern_ids, dataset_path).
    """
    run_dir = Path(run_dir)
    dataset_path, config = read_dataset_path(run_dir, default_dataset_path, dataset_key)
    x_range = _resolve_x_range(x_range, config, dataset_path)

    fig, ax, ids = plot_pattern_surface(net, x_range, grid=grid,
                                        output_index=output_index, scatter_training=scatter_training,dataset_path=dataset_path, **plot_kwargs)
    if "title" not in plot_kwargs:
        ax.set_title(f"{run_dir.name}\n" + ax.get_title())
    return fig, ax, ids, dataset_path


if __name__=="__main__":
    run_dir=r"reg_experiments_attention_init2D\run_11.36_h10_1111_2"
    weight_path=run_dir + r"\model_weights.pt"
    net=build_network_2d(weights_path=weight_path)
    fig, ax,_,_=plot_network_2d(net,run_dir,scatter_training=True)
    plt.show()