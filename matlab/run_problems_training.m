% run_problems_training.m
% Convenience driver: solve every problem_instance_*.mat in
% ../dataset/main/matlab_problems/train/ with MATLAB's PDE Toolbox, writing
% reference_solution_*.mat into ../dataset/main/matlab_solutions/train/.
% Safe to re-run (skips already-solved instances, keeps going past any
% single failure) -- see solve_reference_pde_batch.m.
%
% ../dataset/main/matlab_problems/train holds 1000 instances (seed=2000,
% export_matlab_problem.py). cnn_operator.py's `train` command splits this
% pool itself (whole-sample, see split_train_val_test) into ~689 train /
% ~86 val / ~86 held-out test (after --max_abs_u filtering) via its default
% --val_frac 0.1 --test_frac 0.1 -- there is no separate train/val/test
% directory on disk, the split happens at train time.
%
% Usage: open this file in MATLAB and click Run (or type run_problems_training
% in the Command Window, with matlab/ as the current folder).

solve_reference_pde_batch(fullfile('..', 'dataset', 'main', 'matlab_problems', 'train'), ...
    fullfile('..', 'dataset', 'main', 'matlab_solutions', 'train'));
