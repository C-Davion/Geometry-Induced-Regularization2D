from helper import load_dataset
datasets=["generated_datasets\datasets\score_11.36_function_0000_.pt"]

datas,hyperparam=load_dataset(datasets[0])

print(hyperparam)