"""
train_attention_init.py
--------
Like the train_attention.py, but this time we focus on varying the initialisation scheme rather then the init_scale. 
To keep the file readable, we remove the section/branch with the muon optimizer and keep only adam.

However, we implement way to initilise the network differently.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev
import matplotlib.pyplot as plt
import time
from relu_network import OneHiddenLayerReLUNet, LinearFeatureAttention2D
from helper import _jsonify_param,stacked_jacobian_rank,pattern_stats,init_weight_,init_bias,init_rho_, load_dataset



# ── Main training routine ────────────────────────────────────────────────────

def train_network(
    datas: Tuple,
    input_dim: int = 1,
    seq_len: int =8,
    embed_dim: int=32,
    hidden_dim: int = 20,
    output_dim: int = 1,
    n_samples: int = 200,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    noise_std: float = 0.0,
    lr: float = 1e-2,
    epochs: int = 2000,
    log_every: int = 10,
    seed: int = 0,
    timestamp: str = None,
    score: Optional[float] = None,
    out_root: str = "reg_experiments_attention_SGD",
    dataset_path: str= "generated_datasets\datasets\score_11.36_function_0000_.pt",
    rho: LinearFeatureAttention2D = None,
    grad_tol: float = 1.5e-5,
):
    """
    Train a OneHiddenLayerReLUNet on a fixed single batch to approximate
    target_fn, tracking loss and structural quantities, then save all
    artefacts to out_root/folder_template.format(**config).

    Parameters
    ----------
    target_fn       : callable (N, input_dim) -> (N, output_dim); the function
                       to learn. Imported from a separate module by the caller.
    input_dim, hidden_dim, output_dim : network architecture
    n_samples       : number of points in the single training batch
    x_range         : range from which X is sampled
    noise_std       : optional Gaussian noise added to labels
    lr              : Adam learning rate
    epochs          : number of full-batch gradient steps
    log_every       : how often (in epochs) to log the 3 structural quantities
    seed            : RNG seed (data + weight init)
    target_params   : optional dict of the target function's own fixed
                       parameters (independent of the network's trainable
                       parameters), e.g. {"A": A, "A1": A1, "B": B, "B1": B1}.
                       Tensor values are stringified (e.g. [1,2,-1] -> "1_2_-1")
                       and made available to folder_template under their key
                       names; raw values are saved in config.json.
    out_root        : parent directory for the experiment folder
    rho             : optional feature-transformation module (see
                       OneHiddenLayerReLUNet); defaults to identity

    Returns
    -------
    net, initial_net, out_dir
    """
    start=time.perf_counter()
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── 1. Build the (single) training batch ───────────────────────────────
 
    X_train,Y_train,X_val,Y_val=datas
    X_train=X_train.to(device,dtype=torch.float32)
    X_val=X_val.to(device,dtype=torch.float32)
    Y_train=Y_train.to(device,dtype=torch.float32)
    Y_val=Y_val.to(device,dtype=torch.float32)
    # print(torch.allclose(Y_train.cpu(), X_train.cpu(), atol=1e-2))
    # print((Y_train - X_train).abs().max().item())

     # ── 2. Initialise the network ───────────────────────────────────────────
    init_scale = 1.0    #Do not touch here, to change it, go to train_attention.py but init_type will be kaiming_uniform
    init_type  = 'he_uniform' # he_uniform | he_normal | torch_default | xavier_uniform | xavier_normal
                            # lecun_normal | lecun_uniform | orthogonal | trunc_0.02
    bias_type  = 'uniform'    # zeros | uniform | torch_default | normal
    rho_type   = 'xavier_uniform'  # same names as init_type; xavier_uniform = PyTorch's MHA default
    E_dist     = 'normal'     # normal | uniform
    B_dist     ='normal'
    rho = LinearFeatureAttention2D(seq_len, embed_dim,input_dim, init_scale).to(device, dtype=torch.float32)
    init_rho_(rho, rho_type, E_dist, B_dist,  init_scale)

    feature_dim = rho.output_dim if rho is not None else input_dim
    feature_parameter_count = rho.get_parameter_count()

    W1 = init_weight_(torch.empty(hidden_dim, feature_dim, device=device), init_type) * init_scale
    b1 = init_bias(hidden_dim, feature_dim, bias_type, device) * init_scale
    W2 = init_weight_(torch.empty(output_dim, hidden_dim, device=device), init_type) * init_scale
    b2 = torch.zeros(output_dim, device=device)
    
    net = OneHiddenLayerReLUNet(W1, b1, W2, b2, rho=rho).to(device,dtype=torch.float32)

    initial_net = deepcopy(net)   # frozen snapshot, for the "before" plot
    print("X sum:", X_train.sum().item())              # should be identical every run
    print("W1 sum:", net.W1.sum().item()) 
    

    # ── 3. Full-batch training loop ─────────────────────────────────────────
    losses = []
    val_losses=[]
    log_epochs, jac_ranks, pattern_ranks, n_distinct_list = [], [], [], []
    pattern_hashes=[]
    min_grad_norm=float("inf")
    min_normalized_grad_norm=float("inf")
    jac_rank_at_best_state=10
    p_rank_at_best_state=10
    n_dist_at_best_state=10
    best_state=None    
    critical_point_reached=False
    epoch_reached=None
    def gradient_norm():
        net.zero_grad()

        pred = net(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()

        grads = torch.cat([
            p.grad.flatten()
            for p in net.parameters()
            if p.grad is not None
        ])

        P = grads.numel()                     # number of trainable scalars actually receiving grads
        norm = torch.sqrt((grads ** 2).sum()).item()
        normalized_norm = norm / (P ** 0.5)   # ||grad|| / sqrt(P): per-parameter-averaged magnitude
        max_component = grads.max().item()
        min_component = grads.min().item()

        net.zero_grad()   # leave state clean — this is a pure read, not a training step

        return norm, normalized_norm, max_component, min_component
    def closure():
            optimizer.zero_grad()
            pred=net(X_train)
            loss=loss_fn(pred,Y_train)
            loss.backward()
            return loss
    def validation():
        net.eval()
        with torch.no_grad():
            val_out=net(X_val)
            val_loss=loss_fn(val_out,Y_val)
            return val_loss
    
    total_epochs = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000,threshold=1e-5, threshold_mode='abs', min_lr=1e-8,)
    loss_fn = nn.MSELoss()
    optimizer_config=f"Adam with {scheduler.__class__.__name__}"

    for _ in range(epochs):
        net.train()
        loss = closure()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        val_loss = validation()
        with torch.no_grad():
            pat = net.activation_pattern(X_train)
            pat_hash = hash(tuple(pat.flatten().tolist()))     # add this
            pattern_hashes.append(pat_hash)                     # and this

        if total_epochs % log_every == 0:
            jr = stacked_jacobian_rank(net, X_train)
            p_rank, n_dist = pattern_stats(net, X_train)
            log_epochs.append(total_epochs)
            jac_ranks.append(jr)
            pattern_ranks.append(output_dim * p_rank)
            losses.append(loss.item())
            val_losses.append(val_loss.item())
            n_distinct_list.append(output_dim * (feature_dim + 1) * n_dist+feature_parameter_count)

        grad_norm, normalized_grad_norm,max_component,min_component = gradient_norm()
        if normalized_grad_norm < min_normalized_grad_norm:
            min_normalized_grad_norm = normalized_grad_norm
            min_grad_norm = grad_norm
            max_coord_at_min=max_component
            min_coord_at_min=min_component
            epoch_reached = total_epochs
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            loss_at_best_state=loss.item()
            val_loss_at_best_state=val_loss.item()
            jr = stacked_jacobian_rank(net, X_train)
            p_rank, n_dist = pattern_stats(net, X_train)
            jac_rank_at_best_state=jr
            p_rank_at_best_state=p_rank
            n_dist_at_best_state=n_dist
        if normalized_grad_norm<= grad_tol:
            print("reached critical point, stopping training")
            critical_point_reached = True
            break
        total_epochs += 1
    if best_state is not None:
        net.load_state_dict(best_state)   
    with torch.no_grad():
        true_mse = loss_fn(net(X_train), Y_train).item()
    print("true MSE:", true_mse)
    end=time.perf_counter()
    # tail = pattern_hashes[-200:]
    # print("distinct patterns in tail:", len(set(tail)))
    # print("tail sequence:", tail[-20:], "...")
   
    # ── 4. Resolve output folder name ───────────────────────────────────────

    config = dict(
    input_dim=input_dim,
    seq_len=seq_len,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_samples=n_samples,
    x_range=x_range,
    noise_std=noise_std,
    lr=lr,
    epochs=epochs,
    log_every=log_every,
    seed=seed,
    timestamp=timestamp,
    total_epochs=total_epochs,
    grad_tol=grad_tol,
    initalisation_scale=init_scale,
    initialisation_type=f"weight:{init_type},bias:{bias_type},Attention:{rho_type},embedding:{E_dist}",
    datatset_path=dataset_path
    )
    max_jacobian_rank = max(jac_ranks) if jac_ranks else 0

    monthday = datetime.now().strftime("%m%d")

    prefix = f"run_{score:.2f}_h{hidden_dim}_{monthday}_"

    existing = sorted(Path(out_root).glob(prefix + "*"))
    run_idx = len(existing)

    folder_name = f"{prefix}{run_idx}"

    out_dir = Path(out_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=False)
    # ── 5. Plot: loss (left) and 3 quantities (right), side by side ────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(log_epochs,losses, color="tab:red",label="training loss")
    axes[0].plot(log_epochs,val_losses,color="tab:blue",label="validation_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].plot(log_epochs, jac_ranks, label=rf"$\mathrm{{rank}}(\mathbf{{D}}f(\theta,X))$", marker="o", markersize=3)
    axes[1].plot(log_epochs,pattern_ranks, label=rf"$n_2\mathrm{{rank}}D$", marker="s", markersize=3)
    axes[1].plot(log_epochs, n_distinct_list, label=rf"$n_2(n_0+1)|\mathcal{{A}}(\theta,X)|+n_f$", marker="^", markersize=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].set_title("Structural quantities during training")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "loss_and_quantities.png", dpi=150)
    plt.close(fig)

    # ── 6. Plot: the model output with the dataset───────────
    x_min, x_max = x_range
    y_min, y_max = x_range
    resolution=500 #resolution of the output grid to plot the network output
    nx = ny = resolution
    xs = torch.linspace(x_min, x_max, nx)
    ys = torch.linspace(y_min, y_max, ny)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    
    X_grid = torch.stack([XX.reshape(-1), YY.reshape(-1)],dim=1).to(device)
    with torch.no_grad():
        Z = net(X_grid).reshape(ny, nx).cpu().numpy()
    
    XX = XX.numpy()
    YY = YY.numpy()

    # Convert datasets to numpy
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy().ravel()

    X_val_np = X_val.cpu().numpy()
    Y_val_np = Y_val.cpu().numpy().ravel()
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection="3d")
    surf = ax2.plot_surface(XX,YY,Z,cmap="viridis",alpha=0.65,edgecolor="none",antialiased=True)
    
    fig2.colorbar(surf,ax=ax2,shrink=0.6,pad=0.1,label="f(x, y)")
    
    # Training samples
    ax2.scatter(X_train_np[:, 0],X_train_np[:, 1],Y_train_np,color="red",s=5,alpha=0.55,label="Training set")
    
    # Validation samples
    ax2.scatter(X_val_np[:, 0],X_val_np[:, 1],Y_val_np,color="blue",s=10,alpha=0.8,label="Validation set")
    
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x, y)")

    ax2.set_title(
        f"Model output"
        f"Training: {len(X_train)} | Validation: {len(X_val)}"
    )

    ax2.legend()
    ax2.view_init(elev=30, azim=-135)

    fig2.tight_layout()
    fig2.savefig(out_dir / "model output ", dpi=150)
    plt.close(fig2)

    # ── 7. Save weights and config ──────────────────────────────────────────\
    torch.save(net.state_dict(), out_dir / "model_weights.pt")
    torch.save(initial_net.state_dict(), out_dir / "initial_weights.pt")

    config_to_save = dict(config)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(
            dict(
            losses=losses,
            val_losses=val_losses,
            log_epochs=log_epochs,
            jacobian_rank=jac_ranks,
            pattern_rank=pattern_ranks,
            n_distinct_patterns=n_distinct_list,
            max_jacobian_rank=max_jacobian_rank,
            final_gradient_norm=grad_norm,
            critical_point_reached=critical_point_reached,
            total_epochs=total_epochs,
            min_grad_norm=min_grad_norm,
            min_normalized_grad_norm=min_normalized_grad_norm,
            epoch_reached=epoch_reached,
            max_component_at_min_grad_norm=max_coord_at_min,
            min_component_at_min_grad_norm=min_coord_at_min,
            last_val_loss=val_losses[-1],
            min_val_loss=min(val_losses),
            loss_at_min_grad_norm=loss_at_best_state,
            val_loss_at_min_grad_norm=val_loss_at_best_state,
            jac_rank_at_min_grad_norm=jac_rank_at_best_state,
            p_rank_at_min_grad_norm=p_rank_at_best_state,
            ndistinct_pattern_at_min_grad_norm=n_dist_at_best_state,
            optimizer_config=optimizer_config,
            time_taken=f"Training took {end-start} second",
            rho_formula=rho.formula,
            ),
            f,
        indent=2,
        )

    print(f"Saved all artefacts to: {out_dir}")
    return net, initial_net, out_dir



# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    datasets=["generated_datasets\datasets\score_11.36_function_0000_.pt"]
    for hidden_dim in [10]:
        for epoch in [1000]:
            for dataset_path in datasets:
                datas,hyperparams=load_dataset(dataset_path)
                for i in range(10):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        net, initial_net, out_dir = train_network(
                            datas=datas,
                            input_dim=hyperparams['input_dim'],
                            seq_len=hyperparams['seq_len'],
                            embed_dim=hyperparams['embed_dim'],
                            hidden_dim=hidden_dim,
                            output_dim=1,
                            n_samples=hyperparams['num_train'],
                            x_range=hyperparams['x_range'],
                            noise_std=0.0,
                            lr=1e-3,
                            timestamp=timestamp,
                            epochs=epoch,
                            log_every=500,
                            seed=None,
                            out_root="reg_experiments_attention_init2D",
                            dataset_path=dataset_path,
                            score=11.36, #TODO change it so that is strips from the datasets path.
                            rho=LinearFeatureAttention2D,
                        )
                        print(f"run {i,hidden_dim,epoch} ended")
    print("all run ended")

