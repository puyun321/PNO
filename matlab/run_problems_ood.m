% run_problems_ood.m
% Convenience driver for the two out-of-distribution (OOD) test sets
% (paper Sec. 3.5): a_ij eigenvalues in [2,3] instead of [1,2], and the
% b1, b2, c, beta, f, h amplitudes doubled; g = 0 as in training.
%   ../dataset/main/matlab_problems/ood       (piecewise-constant a_ij, seed 9000)
%   ../dataset/lipschitz/matlab_problems/ood  (Lipschitz a_ij, seed 9500)
% Each set is then solved with the PDE Toolbox into the matching
% ../dataset/<name>/matlab_solutions/ood folder.
%
% Usage: run with matlab/ as the current folder.

generate_problems_testing_eig2to3(100, 9000, fullfile('..', 'dataset', 'main', 'matlab_problems', 'ood'));
generate_problems_lipschitz(0, 0, 100, 9500, fullfile('..', 'dataset', 'lipschitz', 'matlab_problems'), [2 3], 2, 'ood');

for name = {'main', 'lipschitz'}
    n = name{1};
    solve_reference_pde_batch(fullfile('..', 'dataset', n, 'matlab_problems', 'ood'), ...
                              fullfile('..', 'dataset', n, 'matlab_solutions', 'ood'));
end
