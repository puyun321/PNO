% run_problems_lipschitz.m
% Convenience driver: generate a full train(800)/val(100)/test(100) parabolic
% -IBVP dataset with a Lipschitz-continuous a_ij (4 corner tensors bilinearly
% interpolated across a square, then restricted to the disk -- see
% generate_problems_lipschitz.m), then solve every problem_instance_*.mat in
% each split with MATLAB's PDE Toolbox, writing reference_solution_*.mat
% into the matching ../dataset/lipschitz/matlab_solutions/{train,val,test}
% split. Safe to re-run (solve_reference_pde_batch skips already-solved
% instances, keeps going past any single failure).
%
% All generated data lives under ../dataset/lipschitz (sibling to matlab/),
% separate from the main (piecewise-constant a_ij) dataset under
% ../dataset/main:
%   ../dataset/lipschitz/matlab_problems/{train,val,test}/problem_instance_*.mat
%   ../dataset/lipschitz/matlab_solutions/{train,val,test}/reference_solution_*.mat
%
% eig_low/eig_high=[1,2] and the b1,b2,c,beta,f,g,h amplitudes match
% ../dataset/main (export_matlab_problem.py) exactly -- only a_ij's
% construction (Lipschitz-continuous vs piecewise-constant) differs,
% isolating that one variable for the FEM vs Green's-function vs 2D-CNN
% comparison.
%
% Usage: open this file in MATLAB and click Run (or type
% run_problems_lipschitz in the Command Window, with matlab/ as the current
% folder).

problem_root = fullfile('..', 'dataset', 'lipschitz', 'matlab_problems');
solution_root = fullfile('..', 'dataset', 'lipschitz', 'matlab_solutions');

generate_problems_lipschitz(800, 100, 100, 3000, problem_root);

for split = {'train', 'val', 'test'}
    s = split{1};
    solve_reference_pde_batch(fullfile(problem_root, s), fullfile(solution_root, s));
end
