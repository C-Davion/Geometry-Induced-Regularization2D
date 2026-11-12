from helper import load_dataset
import torch
from pathlib import Path
datasets=["generated_datasets\datasets\score_11.36_function_0000_.pt"]

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

if __name__=="__main__":
    dataset_path=datasets[0]
    out=_resolve_dataset_path(dataset_path,None)
    print(out)