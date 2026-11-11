
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from teacher_function_2D import make_target_fn



# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "interesting_functions_attention_2d.jsonl"

OUTPUT_DIR = "generated_datasets"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")

NUMBER_OF_DATA_POINT = 400

TRAIN_RATIO = 0.8

RANDOM_SEED = 42

DEVICE = "cpu"


os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


# ============================================================
# Load JSONL records
# ============================================================

def iter_records(path):
    """Yield every saved record from a JSONL file."""
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(record, n_points, train_ratio=0.8, seed=42):
    """
    Generate training and validation datasets for one teacher function.

    Returns:
        X_train: (N_train, 2)
        Y_train: (N_train, 1)
        X_val:   (N_val, 2)
        Y_val:   (N_val, 1)
    """

    # Reconstruct teacher function
    target_fn = make_target_fn(record)

    # Domain
    x_min, x_max = record["x_range"]
    y_min, y_max = record["x_range"]

    # Sample points uniformly in the domain
    rng = np.random.default_rng(seed)

    X = np.column_stack([
        rng.uniform(x_min, x_max, n_points),
        rng.uniform(y_min, y_max, n_points)
    ])

    X = torch.tensor(X, dtype=torch.float32)

    # Evaluate teacher function
    with torch.no_grad():
        Y = target_fn(X).cpu()

    # Random train/validation split
    permutation = torch.randperm(
        n_points,
        generator=torch.Generator().manual_seed(seed)
    )

    n_train = int(train_ratio * n_points)

    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    return X_train, Y_train, X_val, Y_val


# ============================================================
# Target function visualization
# ============================================================

def plot_dataset(record, X_train, Y_train, X_val, Y_val, save_path):
    """
    Plot the target function as a 3D surface and overlay
    training and validation samples.
    """

    target_fn = make_target_fn(record)

    x_min, x_max = record["x_range"]
    y_min, y_max = record["x_range"]

    # Resolution of visualization grid
    resolution = record.get("grid", 100)

    if isinstance(resolution, (list, tuple)):
        nx, ny = resolution
    else:
        nx = ny = resolution

    # Construct evaluation grid
    xs = torch.linspace(x_min, x_max, nx)
    ys = torch.linspace(y_min, y_max, ny)

    XX, YY = torch.meshgrid(xs, ys, indexing="xy")

    X_grid = torch.stack(
        [XX.reshape(-1), YY.reshape(-1)],
        dim=1
    )

    # Evaluate target function on the visualization grid
    with torch.no_grad():
        Z = target_fn(X_grid).reshape(ny, nx).cpu().numpy()

    XX = XX.numpy()
    YY = YY.numpy()

    # Convert datasets to numpy
    X_train_np = X_train.numpy()
    Y_train_np = Y_train.numpy().ravel()

    X_val_np = X_val.numpy()
    Y_val_np = Y_val.numpy().ravel()

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Target surface
    surf = ax.plot_surface(
        XX,
        YY,
        Z,
        cmap="viridis",
        alpha=0.65,
        edgecolor="none",
        antialiased=True
    )

    fig.colorbar(
        surf,
        ax=ax,
        shrink=0.6,
        pad=0.1,
        label="f(x, y)"
    )

    # Training samples
    ax.scatter(
        X_train_np[:, 0],
        X_train_np[:, 1],
        Y_train_np,
        color="red",
        s=5,
        alpha=0.55,
        label="Training set"
    )

    # Validation samples
    ax.scatter(
        X_val_np[:, 0],
        X_val_np[:, 1],
        Y_val_np,
        color="blue",
        s=10,
        alpha=0.8,
        label="Validation set"
    )

    score = record["score"]

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    ax.set_title(
        f"Target function | Score = {score:.2f}\n"
        f"Training: {len(X_train)} | Validation: {len(X_val)}"
    )

    ax.legend()
    ax.view_init(elev=30, azim=-135)

    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    #plt.close(fig)


# ============================================================
# Main processing loop
# ============================================================

def process_all_records():

    records = iter_records(INPUT_FILE)

    for i, record in enumerate(records):

        score = record["score"]

        print(
            f"\nProcessing function {i + 1} "
            f"(score = {score:.2f})"
        )

        # ----------------------------------------------------
        # Generate dataset
        # ----------------------------------------------------

        X_train, Y_train, X_val, Y_val = generate_dataset(
            record,
            n_points=NUMBER_OF_DATA_POINT,
            train_ratio=TRAIN_RATIO,
            seed=RANDOM_SEED + i
        )

        # ----------------------------------------------------
        # Save dataset
        # ----------------------------------------------------

        filename = f"score_{score:.2f}_function_{i:04d}_.pt"

        dataset_path = os.path.join(
            DATASET_DIR,
            filename
        )

        torch.save(
            {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_val": X_val,
                "Y_val": Y_val,
                "score": score,
                "num_train": len(X_train),
                "num_val": len(X_val),
                "input_dim": record["input_dim"],
                "seq_len":record["seq_len"],
                "embed_dim": record["embed_dim"],
                "x_range": record["x_range"],
                "grid": record["grid"],
            },
            dataset_path
        )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        figure_filename = f"function_{i:04d}_score_{score:.2f}.png"

        figure_path = os.path.join(
            FIGURE_DIR,
            figure_filename
        )

        plot_dataset(
            record,
            X_train,
            Y_train,
            X_val,
            Y_val,
            figure_path
        )

        print(f"  Dataset saved: {dataset_path}")
        print(f"  Figure saved:  {figure_path}")
        print(f"  Training points:   {len(X_train)}")
        print(f"  Validation points: {len(X_val)}")

    print("\nAll functions processed!")


if __name__ == "__main__":
    print("hello world")
    process_all_records()
