% run_problems_testing.m
% Convenience driver: solve every problem_instance_*.mat in
% ../dataset/main/matlab_problems/test/ with MATLAB's PDE Toolbox, writing
% reference_solution_*.mat into ../dataset/main/matlab_solutions/test/. Safe
% to re-run (skips already-solved instances, keeps going past any single
% failure) -- see solve_reference_pde_batch.m.
%
% ../dataset/main/matlab_problems/test is a 100-instance INDEPENDENT extra
% test set (seed=5000, no overlap with the training pool's seed=2000 pool),
% held out entirely from the 1000-sample train/val/test pool for a fully
% separate FEM/Green's-function/2D-CNN comparison.
%
% Usage: open this file in MATLAB and click Run (or type run_problems_testing
% in the Command Window, with matlab/ as the current folder).

solve_reference_pde_batch(fullfile('..', 'dataset', 'main', 'matlab_problems', 'test'), ...
    fullfile('..', 'dataset', 'main', 'matlab_solutions', 'test'));
