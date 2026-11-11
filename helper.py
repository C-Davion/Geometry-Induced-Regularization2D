'''
Some function used to diagnosis the training or to compute the local dimension with the method : float64,float32... passed as argument,  
put them here to not clog the train file.
'''
import json
from copy import deepcopy
#from pathlib import Path
#import glob
from typing import Callable, Optional, Tuple
#from datetime import datetime
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev
#import matplotlib.pyplot as plt
#import time
from relu_network import OneHiddenLayerReLUNet, QuadraticFeatureMap
import math
#import copy
from contextlib import contextmanager
#torch.set_default_dtype(torch.float64)
def _jsonify_param(value):
    """Turn a target-function parameter into a JSON-serializable value."""
    if torch.is_tensor(value):
        return value.tolist()
    return value

# ── Iterages over the target functions generates by ploteacher, used in train2.py whe rhox(x)=(x,x**2) ────────────────────────────

def iter_target_functions(jsonl_path):
    """
    Iterate over a JSONL file whose lines have the form

    {
        "score": ...,
        "A": [...],
        "A1": [...],
        "B": [...],
        "B1": [...]
    }
    """
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            target_params = {
                "A": torch.tensor(obj["A"], dtype=torch.float32).reshape(-1, 1),
                "A1": torch.tensor(obj["A1"], dtype=torch.float32).reshape(-1, 1),
                "B": torch.tensor(obj["B"], dtype=torch.float32).reshape(-1, 1),
                "B1": torch.tensor(obj["B1"], dtype=torch.float32).reshape(-1, 1),
            }

            yield obj["score"], target_params
            break #do just one


# ── Structural-quantity helpers (batch versions) ────────────────────────────
@contextmanager
def _float64_copy(net, X):
    """
    Yield (net64, X64): a float64 deep copy of `net` (the original is left
    untouched, so training can continue in float32) and X in float64.

    torch's default dtype is switched to float64.
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield deepcopy(net).double(), X.to(torch.float64)
    finally:
        torch.set_default_dtype(old)


def stacked_jacobian_rank(net: OneHiddenLayerReLUNet, X: torch.Tensor) -> int:
    """
    Rank of the Jacobian of θ -> [f(θ,x1); ...; f(θ,xN)] w.r.t. all
    parameters θ, evaluated at the current weights, over the full batch X.
    """
    params = dict(net.named_parameters())

    def f_batch(p: dict) -> torch.Tensor:
        out = functional_call(net, p, X)   # (N, output_dim)
        return out.flatten()               # (N*output_dim,)

    jac_dict = jacrev(f_batch)(params)
    J = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1)
    return int(torch.linalg.matrix_rank(J))

def stacked_jacobian_rank_float64(net: OneHiddenLayerReLUNet, X: torch.Tensor) -> int:
    """
    Rank of the Jacobian of θ -> [f(θ,x1); ...; f(θ,xN)] w.r.t. all
    parameters θ, evaluated at the current weights, over the full batch X.
    Computed in float64 (float32 underestimates the rank).
    """
    with _float64_copy(net, X) as (net64, X64):
        params = dict(net64.named_parameters())

        def f_batch(p: dict) -> torch.Tensor:
            out = functional_call(net64, p, X64)   # (N, output_dim)
            return out.flatten()                   # (N*output_dim,)

        jac_dict = jacrev(f_batch)(params)
        J = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1)
        return int(torch.linalg.matrix_rank(J))


def pattern_stats(net: OneHiddenLayerReLUNet, X: torch.Tensor) -> Tuple[int, int]:
    """Return (rank of pattern matrix, number of distinct pattern rows)."""
    with torch.no_grad():
        pat = net.activation_pattern(X)          # (N, hidden_dim)
    rank = int(torch.linalg.matrix_rank(pat.float()))
    n_distinct = len(set(tuple(row.tolist()) for row in pat))
    return rank, n_distinct

def pattern_stats_float64(net: OneHiddenLayerReLUNet, X: torch.Tensor) -> Tuple[int, int]:
    """Return (rank of pattern matrix, number of distinct pattern rows)."""
    with _float64_copy(net, X) as (net64, X64):
        with torch.no_grad():
            pat = net64.activation_pattern(X64)      # (N, hidden_dim)
    rank = int(torch.linalg.matrix_rank(pat.float()))
    n_distinct = len(set(tuple(row.tolist()) for row in pat))
    return rank, n_distinct


# ── Used for debugging convergence,most of it was used when rho(x)=(x,x**2) ────────────────────────────

def adam_step_norm(optimizer):
    """
    Norm of the update Adam would actually apply on its NEXT .step() call,
    using its current momentum (m) and variance (v) estimates — as opposed
    to the raw instantaneous gradient norm.
    """
    total_sq = 0.0
    for group in optimizer.param_groups:
        lr, eps = group['lr'], group['eps']
        beta1, beta2 = group['betas']
        for p in group['params']:
            state = optimizer.state.get(p)
            if not state or 'exp_avg' not in state:
                continue
            t = state['step']
            t = t.item() if torch.is_tensor(t) else t
            if t == 0:
                continue
            m_hat = state['exp_avg'] / (1 - beta1 ** t)
            v_hat = state['exp_avg_sq'] / (1 - beta2 ** t)
            update = lr * m_hat / (v_hat.sqrt() + eps)
            total_sq += update.pow(2).sum().item()
    return total_sq ** 0.5

def hidden_unit_roots(net, x_range):
    """
    For each hidden unit i, with z_i(x) = a_i*x^2 + b_i*x + c_i
    (a_i = W1[i,1], b_i = W1[i,0], c_i = b1[i]),
    return real root(s) in x where z_i(x) = 0 — the neuron's ReLU boundary.
    Only roots inside x_range are kept, since only those affect training data.
    """
    low, high = x_range
    with torch.no_grad():
        a = net.W1[:, 1].detach().cpu()
        b = net.W1[:, 0].detach().cpu()
        c = net.b1.detach().cpu()

    roots_per_unit = []
    for i in range(net.hidden_dim):
        ai, bi, ci = a[i].item(), b[i].item(), c[i].item()
        roots = []
        if abs(ai) < 1e-10:
            if abs(bi) > 1e-10:
                roots = [-ci / bi]
        else:
            disc = bi**2 - 4 * ai * ci
            if disc >= 0:
                s = disc ** 0.5
                roots = sorted([(-bi + s) / (2 * ai), (-bi - s) / (2 * ai)])
        roots_per_unit.append([r for r in roots if low <= r <= high])
    return roots_per_unit
def find_kinks(target_fn, x_range=(-3.0, 3.0), n_points=20000, rel_thresh=0.05):
    """
    Numerically detect kink locations of a scalar target_fn(x) over x_range,
    by looking for jumps in the discrete first derivative (slope). Works for
    any callable target_fn: (N,1) -> (N,1), regardless of how it's built
    internally (min/max composition, splines, etc.) — no access to its
    internal parameters is needed.
    """
    x = torch.linspace(x_range[0], x_range[1], n_points, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        y = target_fn(x).squeeze(-1).double()

    dx = (x[1, 0] - x[0, 0]).item()
    slopes = (y[1:] - y[:-1]) / dx                      # discrete first derivative
    slope_jumps = (slopes[1:] - slopes[:-1]).abs()       # change in slope, point to point

    # a true kink is one grid point with a much bigger slope-jump than its
    # neighbours (which just reflect smooth within-piece curvature)
    thresh = rel_thresh * slope_jumps.max().item()
    idx = torch.nonzero(slope_jumps > thresh).flatten()

    # merge adjacent grid indices that belong to the same kink
    kink_locations = []
    if len(idx) > 0:
        groups = [[idx[0].item()]]
        for i in idx[1:].tolist():
            (groups[-1].append(i) if i - groups[-1][-1] <= 2 else groups.append([i]))
        for g in groups:
            center = int(sum(g) / len(g)) + 1
            kink_locations.append(round(x[center, 0].item(), 4))

    return kink_locations


# ── Muon optimizer and related functions (not used in the end) ────────────────────────────


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize a 2D matrix's update via a quintic Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.double()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                assert g.ndim == 2, "Muon is only defined for 2D matrix parameters"
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                g = g * max(g.size(0), g.size(1)) ** 0.5   # scale to match typical update size

                p.add_(g, alpha=-lr)
def maybe_warm_restart(opt_muon, scheduler_weight, opt_bias, scheduler_bias,
                        total_epochs, epoch_reached, last_restart_epoch,
                        patience_for_restart=5000, restart_factor=0.1):
    epochs_since_improvement = total_epochs - epoch_reached
    epochs_since_restart = total_epochs - last_restart_epoch

    if epochs_since_improvement > patience_for_restart and epochs_since_restart > patience_for_restart:
        for g in opt_muon.param_groups:
            g['lr'] = opt_muon.defaults['lr'] * restart_factor
        for g in opt_bias.param_groups:
            g['lr'] = opt_bias.defaults['lr'] * restart_factor   # reset from its OWN base, not multiply current

        scheduler_weight.cooldown_counter = 0
        scheduler_weight.num_bad_epochs = 0
        scheduler_bias.cooldown_counter = 0
        scheduler_bias.num_bad_epochs = 0
        return True, total_epochs
    return False, last_restart_epoch


#For the initialisation file (train_attention_init)
def init_weight_(W, scheme):
    """In-place init of a 2-D weight (out, in)."""
    fan_out, fan_in = W.shape
    with torch.no_grad():
        if   scheme == 'he_uniform':     nn.init.kaiming_uniform_(W, nonlinearity='relu')
        elif scheme == 'he_normal':      nn.init.kaiming_normal_(W, nonlinearity='relu')
        elif scheme == 'torch_default':  nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        elif scheme == 'xavier_uniform': nn.init.xavier_uniform_(W)
        elif scheme == 'xavier_normal':  nn.init.xavier_normal_(W)
        elif scheme == 'lecun_normal':   nn.init.normal_(W, 0.0, 1 / math.sqrt(fan_in))
        elif scheme == 'lecun_uniform':  nn.init.uniform_(W, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        elif scheme == 'orthogonal':     nn.init.orthogonal_(W, gain=math.sqrt(2))
        elif scheme == 'trunc_0.02':     nn.init.trunc_normal_(W, 0.0, 0.02, -0.04, 0.04)
        else: raise ValueError(f"Unknown weight init: {scheme}")
    return W

def init_bias(n, fan_in, scheme, device):
    if   scheme == 'zeros':         return torch.zeros(n, device=device)
    elif scheme == 'uniform':       return torch.rand(n, device=device) * 2 - 1          # U(-1, 1), your current one
    elif scheme == 'torch_default': return (torch.rand(n, device=device) * 2 - 1) / math.sqrt(fan_in)
    elif scheme == 'normal':        return torch.randn(n, device=device)
    else: raise ValueError(f"Unknown bias init: {scheme}")

def init_rho_(rho, attn_scheme, E_dist, B_dist, scale):
    """Init the feature-attention module (E, B, and the attention projections)."""
    with torch.no_grad():
        # Token embeddings E: acts like a frequency scale on the scalar input
        if   E_dist == 'normal':  rho.E.normal_(0.0, scale)
        elif E_dist == 'uniform': rho.E.uniform_(-math.sqrt(3) * scale, math.sqrt(3) * scale)  # same std as normal
        else: raise ValueError(f"Unknown E_dist: {E_dist}")

        if   B_dist == 'normal':  rho.B.normal_(0.0, scale)
        elif B_dist == 'uniform': rho.B.uniform_(-math.sqrt(3) * scale, math.sqrt(3) * scale)  # same std as normal
        elif B_dist == 'zeros' : rho.B.zeros_
        else: raise ValueError(f"Unknown B_dist: {B_dist}")
        

        # Attention: q, k, v are stacked in in_proj_weight (3*d, d); init each block separately
        mha = rho.attention
        for blk in mha.in_proj_weight.chunk(3, dim=0):   # chunks are views -> in-place works
            init_weight_(blk, attn_scheme)
        init_weight_(mha.out_proj.weight, attn_scheme)
        mha.in_proj_bias.zero_()
        mha.out_proj.bias.zero_()
#### For parallel script, resovling folder name race (train attention_SGD_para)

def make_run_dir(out_root, prefix, run_id):
    """Atomic, race-free: run_id makes names unique across parallel tasks;
    the _vN suffix only kicks in if you re-run the same run_id on the same day."""
    base = Path(out_root) / f"{prefix}{run_id}"
    out_dir, n = base, 0
    while True:
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            n += 1
            out_dir = base.with_name(f"{base.name}_v{n}")

def load_dataset(dataset_path):
    """
    Load a saved .pt dataset.

    Returns:
        X_train, Y_train, X_val, Y_val
    """
    data = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=True
    )

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    

    return (X_train, Y_train, X_val, Y_val),data