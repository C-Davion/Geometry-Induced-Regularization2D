"""
Run the Jacobian check on several randomly chosen run folders of one or more base directories.

Example:
    python batch_study.py --base_dirs reg_experiments_attention_init2D --n_folders 5 \
        --n_runs 32 --device cuda --show --out_csv results.csv

Layout expected:
    base_dir/
        run_11.36_h10_1112_16/model_weights.pt
        run_.../model_weights.pt
        ...

For each chosen folder:
  * the model and data are loaded once and the full Jacobian is computed once (it does not
    depend on the seed),
  * for seed = 0 .. n_runs-1, a sample index is drawn with random.Random(seed), the gradient
    check is done, and (optionally) the figure is shown for `--show_seconds` seconds.
The same seeds are used for every folder, so folders whose dataset has the same size n get
the same sample indices.

One row per (folder, seed) is written to the CSV.
"""
import torch
import argparse
import os
import random
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# If you kept the name diagnostic.py for the study file, change the module name here.
from diagnostic_tangeant import (compute_jacobian, load_model_and_data, make_plot,
                            resolve_keys, run_check)
from buildnetwork import DEFAULT_DATASET_PATH
from loadrun import gather_runs


def list_run_folders(base_dir: str):#  To only contains runs that convergedmaybe pass this as args?, maybe to much?
    random_sampling=False
    runs=[]
    """Sub-folders of base_dir that contain a model_weights.pt."""
    runs.append(gather_runs(base_dirs=base_dir,folder_patterns='run_11.36_h10_*',random_sampling=random_sampling,init_scheme_filter='uniform_bias',
                               init_scale_filter=1.0,verbose=True))
    runs.append(gather_runs(base_dirs=base_dir,folder_patterns='run_11.36_h10_*',random_sampling=random_sampling,init_scheme_filter='uniform_bias',
                               init_scale_filter=4.5,verbose=True))
    runs=pd.concat(runs,ignore_index=True)
    weights=runs['weight_path'].tolist()
    stripped_weights=[weight.parent.name for weight in weights] 
    return sorted(stripped_weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dirs", nargs="+", required=True)
    ap.add_argument("--dataset_folder",required=False,help="folder containing the dataset.pt",default=DEFAULT_DATASET_PATH)
    ap.add_argument("--n_folders", type=int, required=True, help="number of folders drawn per base dir")
    ap.add_argument("--folder_seed", type=int, default=0, help="seed for the choice of folders")
    ap.add_argument("--n_runs", type=int, default=32, help="runs (= seeds 0..n_runs-1) per folder")
    ap.add_argument("--keys", nargs="+", default=["full"])
    ap.add_argument("--jac_method", default="batched", choices=["batched", "per_sample"])
    ap.add_argument("--chunk_size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--show", action="store_true", help="show each figure, then close it")
    ap.add_argument("--show_seconds", type=float, default=5.0)
    ap.add_argument("--out_csv", default="results.csv")
    args = ap.parse_args()

    rows = []
    try:
        for base_dir in args.base_dirs:
            available = list_run_folders(base_dir)
            if not available:
                print(f"[warning] no run folder with model_weights.pt in {base_dir}")
                continue
            k = min(args.n_folders, len(available))
            chosen = random.Random(args.folder_seed).sample(available, k)
            print(f"\n{base_dir}: {k}/{len(available)} folders selected: {chosen}")

            for folder in chosen:
                path = os.path.join(base_dir,folder)
                model = X = J = None
                try:
                    print(f"\n=== {path}")
                    model, X = load_model_and_data(path,args.dataset_folder, args.device, args.float32)
                    keys = resolve_keys(model, args.keys)
                    n = X.shape[0]
                    #J = compute_jacobian(model, X, keys, method=args.method, chunk_size=args.chunk_size)
                    J=jacobian(model,X,keys,jac_method=args.jac_method,chunk_size=args.chunk_size,copy_model=False)['J']
                    print(f"n = {n}, Jacobian shape = {tuple(J.shape)}")

                    for seed in range(args.n_runs):
                        idx = random.Random(seed).randrange(n)
                        metrics, plot_data = run_check(model, X, keys, idx, J)
                        rows.append({"base_dir": base_dir, "folder": folder, "seed": seed, **metrics})
                        print(f"  seed {seed:2d} | idx {idx:5d} | ||g||^2 = {metrics['grad_norm2']:.4e} "
                              f"| fd = {metrics['finite_diff']:.4e} | rel_diff = {metrics['rel_diff']:.2e}"
                              f"| row_max_err = {metrics["row_max_abs_err"]:.4e} | row_rel_err = {metrics['row_rel_err']:.2e}")

                        if args.show and plot_data is not None:
                            fig = make_plot(plot_data, f"\n{folder} | seed {seed}")
                            path_to_save_fig=f"{folder}_{seed}.png"
                            plt.savefig(path_to_save_fig)
                            plt.show(block=False)
                            plt.pause(args.show_seconds)
                            plt.close(fig)
                except Exception:
                    print(f"[error] folder {path} skipped:")
                    traceback.print_exc()
                finally:
                    del model, X, J
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        # saved even if interrupted (Ctrl+C) so completed runs are not lost
        df = pd.DataFrame(rows)
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n{len(df)} rows saved to {args.out_csv}")


if __name__ == "__main__":
    main()