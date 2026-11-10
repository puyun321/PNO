# PNO: Parabolic Neural Operator

Code and data for **"Neural Operators for Solving General Parabolic Equations"** (P.-Y. Kow, P.-Z. Kow, J.-N. Wang).

The PNO is a spatial mesh-invariant neural operator for linear parabolic equations with space- and time-dependent coefficients. Its architecture follows Levi's parametrix construction. It is trained and evaluated against finite element (FEM) reference solutions computed with the MATLAB PDE Toolbox.

## Requirements

- Python 3 with `torch`, `numpy`, `scipy`, `matplotlib`
- MATLAB with the PDE Toolbox, needed only to regenerate the FEM solutions

## Data

`dataset/{main,lipschitz}/` contains two datasets:

| Dataset | Diffusion tensor a_ij | Splits |
|---|---|---|
| `main` | piecewise-constant (discontinuous) | train 1000 / test 100 / ood 100 |
| `lipschitz` | Lipschitz-continuous | train 800 / val 100 / test 100 / ood 100 |

Each dataset has three kinds of files:

- `matlab_problems/`: the coefficients of each problem instance
- `samples/`: the PNO inputs (`.npz`)
- `matlab_solutions/`: the FEM solutions (~8 GB, not in this repo)

The full dataset, including the FEM solutions, is on Zenodo: [10.5281/zenodo.24111220](https://doi.org/10.5281/zenodo.24111220).

## Quick start

Run all commands from the repository root. The trained checkpoints are in `runs_pno/`.

```bash
# evaluate on train/val/test and on OOD data
python pno.py evaluate     --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main
python pno.py evaluate_ood --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main

# per-sample figures (u_FEM vs u_PNO)
python pno.py plot_samples --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main
```

For the Lipschitz dataset, replace `main` with `lipschitz`.

## Training from scratch

```bash
python pno.py pretrain --out runs_pno/pretrained.pt                       # fit each block to its closed form
python pno.py train --dataset main --init runs_pno/pretrained.pt --out runs_pno/main/pno.pt
```

## Regenerating the data

1. **Problems.**
   - `main` train/test: `python export_matlab_problem.py --n_samples 1000 --seed 2000 --out dataset/main/matlab_problems/train`, and `--n_samples 100 --seed 5000 --out dataset/main/matlab_problems/test` for the test split.
   - `lipschitz` and both OOD sets are generated in MATLAB by the scripts in step 2.
2. **FEM solutions.** Run these in MATLAB with `matlab/` as the current folder: `run_problems_training.m`, `run_problems_testing.m`, `run_problems_lipschitz.m`, `run_problems_ood.m`.
3. **Samples.** Run this for each dataset and split:
   ```bash
   python import_matlab_solution.py --problem_dir dataset/main/matlab_problems/test \
       --solution_dir dataset/main/matlab_solutions/test --out dataset/main/samples/test
   ```

## Files

| File | Purpose |
|---|---|
| `pno.py` | PNO model: pretraining, training, evaluation, plots |
| `levi_operator.py`, `subnetworks.py` | closed-form Levi parametrix reference (`pno.py check_exact`) |
| `export_matlab_problem.py`, `matlab/` | problem generation and FEM solves |
| `import_matlab_solution.py`, `qmc_sampling.py` | resampling FEM solutions to sensor points |
| `plot_aij_field.py`, `figures/` | a_ij plots and paper figures |
| `results/` | evaluation outputs |
