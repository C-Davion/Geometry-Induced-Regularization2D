
# Geometry-Induced Regularization in 2D

This repository implements the 2D-input version of **Geometry-Induced Regularization**.

The objective is to study the relationship between neural network initialization, Jacobian geometry, and the behavior of the network when its parameters are perturbed.

The teacher function considered here maps:

\[
f:\mathbb{R}^2\rightarrow\mathbb{R}.
\]

The repository provides tools to:

- Generate 2D teacher functions.
- Generate training and validation datasets.
- Train neural networks from different initializations.
- Collect and filter training runs according to their initialization properties.
- Compute Jacobian ranks at initialization.
- Analyze the resulting distributions.
- Diagnose the correctness of the computed Jacobians.

---

# 1. Experimental Pipeline

The experimental pipeline consists of six steps.

## Step 1 — Generate a teacher function

Run:

```bash
python teacher_function_2D.py
```

This generates a JSON file containing all the parameters defining the teacher function.

The JSON file is saved in the same directory as the script and stores the parameters on a single line.

The teacher function is a mapping:

\[
f:\mathbb{R}^2\rightarrow\mathbb{R}.
\]

## Step 2 — Generate the dataset

Run:

```bash
python generated_dataset.py
```

This generates the dataset and the corresponding visualization.

The output directory has the following structure:

```text
generated_dataset/
│
├── datasets/
│   └── score_{function_score}_function_{function_id}.pt
│
└── figures/
    └── function_{function_id}_score_{function_score}.png
```

### Dataset (`.pt`)

The dataset file contains:

- `X_train`: training inputs in \(\mathbb{R}^2\).
- `Y_train`: training targets in \(\mathbb{R}\).
- `X_val`: validation inputs in \(\mathbb{R}^2\).
- `Y_val`: validation targets in \(\mathbb{R}\).
- Additional metadata, including the number of points and the dataset path.

### Figure (`.png`)

The figure displays the teacher function as a surface, together with the training and validation samples.

This provides a visual representation of the target function and the spatial distribution of the generated dataset.

## Step 3 — Train neural networks

Two training scripts are available:

**Standard training:**

```bash
python train_attention_init2D.py
```

**Parallel training:**

```bash
python train_attention_init_para.py
```

Training is designed primarily for GPU execution using PyTorch CUDA.

The output directory (`out-dir`) contains the results of the training run, including:

- Model weights at initialization.
- A configuration file describing the training setup.
- Training logs.

### Important: Initialization configuration

Some initialization parameters are not exposed as arguments of the training function.

In particular:

- Initialization scale.
- Initialization scheme.

These parameters must currently be modified directly in the corresponding source-code lines.

Make sure that the configuration saved alongside the training run accurately reflects the initialization parameters used.

### Random sampling experiments

To generate runs corresponding to random initialization sampling, it is recommended to use a separate output directory and set:

```python
epochs = -1
```

This allows initialization samples to be saved without performing the usual training procedure.

## Step 4 — Collect and filter training runs

Run:

```bash
python loadrun.py
```

This script builds a dataframe containing the available training runs.

The runs are filtered according to parameters extracted from their training configuration files, notably:

- Initialization scale.
- Initialization scheme.
- Gradient norm of the saved weights.

The resulting dataframe is saved as a CSV file.

### Special case: Random sampling

When `random_sampling=True`, the gradient-norm filter is not applied to the saved weights.

This is important when collecting initialization samples, since these runs are not intended to be filtered in the same way as trained models.

## Step 5 — Compute Jacobian ranks

Run:

```bash
python compute_ranks.py --csv <input_csv>
```

Replace `<input_csv>` with the CSV generated in Step 4.

For each row of the dataframe, the script loads the corresponding model weights:

```text
basedir/run_dir/model_weights.pt
```

It then computes the rank-related quantities of the Jacobian at the saved parameter values.

The resulting information is added to the dataframe, which is saved to a new CSV file.

The output CSV therefore contains both the original run metadata and the computed Jacobian information.

## Step 6 — Plot histograms

Run:

```bash
python histogramms.py --csv <input_csv>
```

The input should be the CSV generated in Step 5.

This script allows the user to:

- Select the runs to include in the analysis.
- Filter the dataframe according to the desired experimental parameters.
- Select the columns to visualize.
- Plot histograms of the selected quantities.

The histograms can then be used to compare the distributions obtained under different initialization schemes and experimental configurations.

---

# 2. Jacobian Diagnostics

Two scripts are available to investigate the correctness of the computed Jacobians:

```text
diagnostic.py
diagnostic_tangeant.py
```

These scripts provide complementary diagnostics for the Jacobian computations.

## 2.1. Batched diagnostic

`diagnostic.py` is used to investigate the Jacobian computation in the batched setting.

It checks the consistency of the computed Jacobian with the corresponding derivatives of the neural network.

## 2.2. Tangent diagnostic

`diagnostic_tangeant.py` investigates whether the computed Jacobian is correctly defined and whether it agrees with the directional derivatives of the network.

The diagnostic is performed for a given input directory:

```text
base_dir/run_dir/
```

The Jacobian is computed at the parameter values stored in:

```text
model_weights.pt
```

### Mathematical setup

Let the neural network be parameterized by:

\[
\theta\in\mathbb{R}^p.
\]

For a fixed input \(x\), consider the mapping:

\[
F_x:\mathbb{R}^p\rightarrow\mathbb{R},
\qquad
F_x(\theta)=\operatorname{net}(\theta,x).
\]

In the batched setting, the corresponding mapping is:

\[
F_X:\mathbb{R}^p\rightarrow\mathbb{R}^n,
\qquad
F_X(\theta)=\operatorname{net}(\theta,X),
\]

where \(X=(x_1,\ldots,x_n)\) is a batch of inputs.

The Jacobian is evaluated at the saved parameter values \(\theta\).

### 2.3. Directional derivative consistency

A sample \(x_i\) is randomly selected from the dataset \(X\).

The gradient of the network output with respect to its parameters is computed:

\[
g=\nabla_\theta\operatorname{net}(\theta,x_i).
\]

The diagnostic then considers the function:

\[
\phi(t)=\operatorname{net}(\theta+tg,x_i).
\]

By the chain rule, its derivative at zero should satisfy:

\[
\phi'(0)
=
\left\langle
\nabla_\theta\operatorname{net}(\theta,x_i),g
\right\rangle
=
\|g\|^2.
\]

Consequently, the first-order Taylor approximation around \(t=0\) is:

\[
\phi(t)\approx
\operatorname{net}(\theta,x_i)+t\|g\|^2.
\]

The diagnostic plots the following two functions on the same figure:

1. The actual network output under a parameter perturbation:

\[
t\mapsto\operatorname{net}(\theta+tg,x_i).
\]

2. Its first-order Taylor approximation:

\[
t\mapsto
\operatorname{net}(\theta,x_i)+t\|g\|^2.
\]

The two curves should be tangent at \(t=0\).

This provides a visual check of the consistency between the computed gradient and the actual behavior of the neural network under small parameter perturbations.

### 2.4. Error metrics

The script computes the following quantities.

**1. Absolute Jacobian row error**

The difference between the independently computed gradient and the corresponding row of the Jacobian:

\[
\operatorname{row\_abs\_err}
=
\|g-J[i,:]\|_\infty.
\]

**2. Relative Jacobian row error**

The absolute error normalized by the gradient norm:

\[
\operatorname{row\_rel\_err}
=
\frac{\operatorname{row\_abs\_err}}{\|g\|}.
\]

**3. Finite-difference diagnostic**

The finite-difference step is defined as:

\[
h=10^{-5}\|g\|.
\]

The current implementation evaluates:

\[
fd=\frac{\phi(h)+\phi(-h)}{2}.
\]

**4. Relative finite-difference error**

The script compares the finite-difference quantity with the expected directional derivative:

\[
\|g\|^2.
\]

The relative error is:

\[
\operatorname{rel\_err}
=
\frac{|fd-\|g\|^2|}{\|g\|^2}.
\]

**Important:** The finite-difference expression above is reproduced from the current implementation description. It should be checked against the actual code.

The standard centered finite-difference approximation of the first derivative is:

\[
\frac{\phi(h)-\phi(-h)}{2h},
\]

which should be compared with \(\|g\|^2\).

The symmetric expression involving the sum instead corresponds to a different quantity and requires an appropriate normalization and subtraction of \(\phi(0)\).

---

# 3. Suggested Documentation Improvements

The following additions would make the repository easier to use, reproduce experiments with, and maintain.

## 3.1. Add installation instructions

The README should specify:

- Python version.
- PyTorch version.
- CUDA version, if required.
- Required Python packages.
- Installation commands.

For example:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file should be provided if one does not already exist.

## 3.2. Document script arguments

Each script should have a short description of its command-line arguments.

In particular:

- Dataset generation parameters.
- Training parameters.
- Initialization scale and initialization scheme.
- Output directory.
- CSV input and output paths.
- Histogram selection parameters.

It would be useful to provide one complete command-line example for each script.

## 3.3. Document the output directory structure

The README should show the expected directory structure after a complete experiment.

For example:

```text
experiment/
│
├── generated_dataset/
│   ├── datasets/
│   └── figures/
│
├── training_runs/
│   └── run_*/
│       ├── model_weights.pt
│       ├── config.json
│       └── logs/
│
├── results/
│   ├── runs.csv
│   ├── runs_with_ranks.csv
│   └── figures/
│
└── README.md
```

The actual filenames and folder structure should be adjusted to match the implementation.

## 3.4. Specify the Jacobian rank convention

The README should explain precisely:

- Which Jacobian is computed.
- Its dimensions.
- Whether the Jacobian is evaluated on training or validation data.
- Which numerical rank definition is used.
- Which tolerance is used to determine the rank.
- Whether singular values are computed explicitly.

This is particularly important because numerical rank can depend strongly on the chosen tolerance.

## 3.5. Improve reproducibility

Document how random seeds are handled throughout the pipeline.

Ideally, the following should be reproducible:

- Teacher function generation.
- Dataset generation.
- Network initialization.
- Random sample selection in diagnostics.

The seed should be saved in the configuration file associated with each experiment.

## 3.6. Add a complete example experiment

A minimal end-to-end example would make the repository considerably easier to use.

It should show how to go from teacher function generation to the final histogram:

```bash
python teacher_function_2D.py
python generated_dataset.py
python train_attention_init2D.py
python loadrun.py
python compute_ranks.py --csv <input_csv>
python histogramms.py --csv <input_csv>
```

The exact arguments should be included once their defaults and interfaces are documented.

## 3.7. Document known limitations

It would be useful to explicitly mention:

- Whether CPU execution is supported.
- Whether CUDA is mandatory.
- Whether parallel training requires multiple GPUs.
- Whether all scripts assume the same directory structure.
- Whether model weights are saved before or after training.
- Whether the Jacobian computation scales with the number of parameters and samples.

## 3.8. Add a scientific reference section

If the repository is associated with a paper or a research project, include:

- A link to the corresponding paper or preprint.
- A short description of the experimental objective.
- The mathematical definitions of the main quantities.
- A citation format for users who reuse the code.

---

# 4. Recommended Future Improvements

Beyond documentation, a few changes to the codebase could improve its usability.

1. **Centralize configuration:** Move initialization parameters currently hardcoded in training scripts into configuration files or command-line arguments.

2. **Standardize experiment metadata:** Ensure every training run saves the same metadata fields, including seeds, initialization parameters, dataset identifiers, and model architecture.

3. **Make the pipeline easier to reproduce:** Consider adding a single orchestration script that executes the six experimental steps in sequence.

4. **Improve diagnostic robustness:** Add explicit checks for zero or near-zero gradient norms, numerical precision, and finite-difference step sizes.

5. **Separate raw and processed results:** Keep original training logs and CSV files separate from CSV files enriched with computed Jacobian quantities.

6. **Add automated tests:** Include small tests for dataset generation, Jacobian dimensions, Jacobian-gradient consistency, and numerical rank computation.

---

# 5. Repository Status

This repository is maintained separately from the original `GEometry_Induced_Regularization` repository.

It implements the corresponding experimental pipeline for two-dimensional inputs:

\[
\boxed{f:\mathbb{R}^2\rightarrow\mathbb{R}}
\]

The documentation and experimental workflow are intended to evolve independently.
