import os
from pathlib import Path
import json
FIELDS = ["jac_rank_at_min_grad_norm", "p_rank_at_min_grad_norm","ndistinct_pattern_at_min_grad_norm"]
BASE_DIR= Path(__file__).resolve().parent / "reg_experiments_attention_init" #change here to swtich between SGD and init for Adam
GRAD_THRESHOLD=1.5e-5
FOLDER_PATTERN = "run_47.77_hxx_date_*"
import pandas as pd
import numpy as np


def _read_json_or_jsonl(path): #rewrite this function here instead of importing from expstats_attention.py because of the torch pandas import order DLL conflict.
    """
    Read a file that's either one JSON object/list, or JSONL (one JSON
    object per line). Always returns a list of dicts.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries
def load_run_config(run_dir):
    with open(Path(run_dir) / "config.json", "r") as f:
        return json.load(f)


def load_filtered_runs(base_dir=BASE_DIR, folder_pattern=FOLDER_PATTERN,
                        threshold=GRAD_THRESHOLD,random_sampling=False,init_scheme_filter='uniform_bias',init_scale_filter=1.0, verbose=True): #same as the above function
    """
    Scan base_dir for folders matching folder_pattern and keep the ones
    whose training_log.json reports min_normalized_grad_norm < threshold.
    """
    run_dirs = sorted(base_dir.glob(folder_pattern))
    records = []
    init_schemes={
        "uniform": "weight:he_uniform,bias:uniform,Attention:xavier_uniform,embedding:normal", #1.0 3.0 6.0
        "normal": "weight:he_normal,bias:normal,Attention:xavier_normal,embedding:normal" , #TEMP #3.0
        "uniform_bias":"weight:he_uniform,bias:uniform,Attention:xavier_uniform,embedding:normal" #1.0 6.0
    }

    if verbose:
        print(f"Found {len(run_dirs)} matching directories")
    
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        log_file = run_dir / "training_log.json"
        config = load_run_config(run_dir)
        init_scheme=config["initialisation_type"]
        init_scale=config["initalisation_scale"]
        dataset_path=config["dataset_path"]
        hidden_dim=config[ "hidden_dim"]

        entries = _read_json_or_jsonl(log_file)
        for data in entries:
            if not isinstance(data, dict):
                continue
           
            if random_sampling:
                if init_scheme== init_schemes[init_scheme_filter] and init_scale==init_scale_filter:
                    record = {"run": run_dir.name}
                    record['scale']=init_scale_filter
                    record['scheme']=init_scheme_filter
                    record['dataset']=dataset_path
                    record['weight_path']=run_dir.name+"/model_weights.pt"
                    record['is_random_sampling']=random_sampling
                    record['hidden_dim']= hidden_dim
                    records.append(record)
                    break
            else:
                norm = data.get("min_normalized_grad_norm")
                if norm is not None and  norm < threshold and  init_scheme== init_schemes[init_scheme_filter] and init_scale==init_scale_filter : #change this for filtering with init scheme.
                    record = {"run": run_dir.name}
                    record['scale']=init_scale_filter
                    record['scheme']=init_scheme_filter
                    record['dataset']=dataset_path
                    record['weight_path']=run_dir.name+"/model_weights.pt"
                    record['is_random_sampling']=random_sampling
                    record['hidden_dim']= hidden_dim
                    records.append(record)
                    break

    df = pd.DataFrame(records)

    if verbose:
        print(f"\nKept {len(df)} runs")

    return df


def gather_runs(base_dirs=BASE_DIR,folder_patterns=FOLDER_PATTERN,  threshold=GRAD_THRESHOLD,
                    random_sampling=False,init_scheme_filter='uniform_bias',init_scale_filter=1.0, verbose=True):
    
    """
    Apply the usual grad-norm filter (expstats_attention) inside every
    directory in base_dirs and concatenate the results. Adds a "base_dir"
    column, since a run's folder name alone no longer says where it lives.
    """
    if isinstance(base_dirs,str):
        base_dirs=[base_dirs]
    if isinstance(folder_patterns, str):
        folder_patterns = [folder_patterns]
    dfs = []
    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        if verbose:
            print(f"\n=== {base_dir} ===")
        if not base_dir.is_dir():
            print(f"Warning: {base_dir} does not exist, skipping.")
            continue
        for pattern in folder_patterns:
            df = load_filtered_runs(base_dir=base_dir, folder_pattern=pattern,
                                    threshold=threshold,  random_sampling=random_sampling,
                                    init_scheme_filter=init_scheme_filter,init_scale_filter=init_scale_filter,verbose=verbose,)
            if len(df):
                df = df.copy()
                df["base_dir"] = base_dir  #TODO to debug
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__=="__main__":
    
    base_dirs=[
               # 'reg_experiments_attention_init2D',
              'reg_experiments_attention_randomsampling']
    folder_patterns=['run_11.36_h10_*']
    runs=gather_runs(base_dirs,folder_patterns,threshold=GRAD_THRESHOLD,
                     random_sampling=True,init_scheme_filter='uniform_bias',init_scale_filter=1.0,verbose=True)
    print(runs.head()['weight_path'])
    print(runs.columns)