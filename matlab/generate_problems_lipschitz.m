function generate_problems_lipschitz(n_train, n_val, n_test, seed, out_root, eig_range, amp_scale, split_dir)
%GENERATE_PROBLEMS_LIPSCHITZ  Generate a full train/validation/test parabolic
%-IBVP dataset (Eq. 1.4) with a Lipschitz-continuous a_ij diffusion tensor,
%replacing this project's usual piecewise-constant-per-quadrant a_ij
%(export_matlab_problem.py / generate_problems_testing_eig2to3.m) with a
%DIFFERENT construction: draw 4 independent elliptic tensors at the 4
%corners of the SQUARE bounding box, then bilinearly interpolate a11, a12,
%a22 across that square -- the disk-shaped domain's a_ij is then whatever
%that square field restricts to (no extra step needed: the same grid+
%bilinear-lookup schema solve_reference_pde_batch.m / import_matlab_
%solution.py already use for a11_grid does the disk restriction).
%
% Why bilinear interpolation from 4 corners is already Lipschitz AND
% elliptic, no smoothing filter required:
%   - Bilinear interpolation is itself a convex combination of the 4 corner
%     values: at any (x,y) in the square, a_ij(x,y) = sum_k w_k(x,y) * A_k,
%     with w_k(x,y) = (product of two linear "hat" factors) >= 0 and
%     sum_k w_k(x,y) = 1 everywhere in the square (the standard bilinear
%     basis functions). A bilinear function is Lipschitz on a bounded
%     domain (its gradient is bounded), so a_ij is automatically Lipschitz
%     continuous -- no separate moving-average step is needed this time.
%   - Ellipticity is preserved for the SAME reason as any convex
%     combination: for any fixed unit vector v, v'*a_ij(x,y)*v is the same
%     convex combination of v'*A_k*v, k=1..4, each of which already lies in
%     [eig_low, eig_high] (that's how the 4 corner tensors were drawn).
%     A convex combination of numbers in [eig_low, eig_high] stays in
%     [eig_low, eig_high] for every v, so a_ij(x,y)'s eigenvalues stay in
%     that range everywhere in the square (hence everywhere on the disk).
%
% g is fixed to 0 (zero initial condition u(x,0)=0, Eq. 1.4b).
% b1, b2, c, beta, f, h are still drawn i.i.d. Normal(0, amplitude), same
% eig_low/eig_high=[1,2] and amplitudes as problems_training/problems_
% testing (export_matlab_problem.py) -- only a_ij's construction differs,
% so this dataset isolates a_ij's continuity as the one variable changed
% relative to the project's main (piecewise-constant) dataset.
%
% Because a_ij is no longer piecewise-constant, div(a_ij*grad(u)) !=
% a_ij*u_xixj; solve_reference_pde_batch.m already knows how to fold the
% correction term (corr1_grid/corr2_grid, precomputed here) into b1,b2 for
% exactly this a11_grid/a12_grid/a22_grid schema -- see that file's
% comments.
%
% Usage:
%   >> generate_problems_lipschitz()
%       800 train / 100 val / 100 test, seed 3000, written under
%       ../dataset/lipschitz/matlab_problems/{train,val,test}
%   >> generate_problems_lipschitz(800, 100, 100, 3000, fullfile('..','dataset','lipschitz','matlab_problems'))
%
% Optional OOD arguments (defaults reproduce the in-distribution data above):
%   eig_range  [eig_low eig_high] of every corner tensor   (default [1 2])
%   amp_scale  multiplies every b1,b2,c,beta,f,h amplitude  (default 1)
%   split_dir  if given, ALL n_test samples go to out_root/split_dir and
%              n_train/n_val must be 0 (e.g. an 'ood' split)
% Lipschitz OOD set mirroring the paper's Sec. 3.5 OOD (eigs in [2,3], 2x amplitudes):
%   >> generate_problems_lipschitz(0, 0, 100, 9500, fullfile('..','dataset','lipschitz','matlab_problems'), [2 3], 2, 'ood')

if nargin < 1 || isempty(n_train), n_train = 800; end
if nargin < 2 || isempty(n_val),   n_val = 100; end
if nargin < 3 || isempty(n_test),  n_test = 100; end
if nargin < 4 || isempty(seed),    seed = 3000; end
if nargin < 5 || isempty(out_root), out_root = fullfile('..', 'dataset', 'lipschitz', 'matlab_problems'); end
if nargin < 6 || isempty(eig_range), eig_range = [1.0, 2.0]; end
if nargin < 7 || isempty(amp_scale), amp_scale = 1.0; end
if nargin < 8, split_dir = ''; end

radius = 1.0;
center = [0.0, 0.0];
T = 1.0;
eig_low = eig_range(1);
eig_high = eig_range(2);
ngrid = 51;

% same amplitudes as problems_training/problems_testing (export_matlab_
% problem.py) -- only a_ij's construction changes in this dataset.
% amp_scale > 1 widens them for an OOD set.
amp_b1 = 0.6 * amp_scale; amp_b2 = 0.6 * amp_scale; amp_c = 0.6 * amp_scale;
amp_beta = 0.8 * amp_scale;
amp_f = 1.0 * amp_scale; amp_g = 1.0 * amp_scale; amp_h = 1.0 * amp_scale;

if isempty(split_dir)
    splits = {'train', 'val', 'test'};
    counts = [n_train, n_val, n_test];
else
    assert(n_train == 0 && n_val == 0, 'split_dir given: put all samples in n_test');
    splits = {split_dir};
    counts = n_test;
end

global_idx = 0;
for s = 1:numel(splits)
    split_name = splits{s};
    n = counts(s);
    out_dir = fullfile(out_root, split_name);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    for i = 0:n - 1
        rng(seed + global_idx, 'twister');  % reproducible, no seed reuse across splits

        [a11_grid, a12_grid, a22_grid, xg, yg] = draw_corner_bilinear_a_ij( ...
            center, radius, ngrid, eig_low, eig_high);

        hx = xg(2) - xg(1);
        hy = yg(2) - yg(1);
        [dA11dx, dA11dy] = gradient(a11_grid, hx, hy);
        [dA12dx, dA12dy] = gradient(a12_grid, hx, hy);
        [dA22dx, dA22dy] = gradient(a22_grid, hx, hy);
        corr1_grid = dA11dx + dA12dy;
        corr2_grid = dA12dx + dA22dy;

        lipschitz_L_est = max([abs(dA11dx(:)); abs(dA11dy(:)); ...
                                abs(dA12dx(:)); abs(dA12dy(:)); ...
                                abs(dA22dx(:)); abs(dA22dy(:))]);
        [lam_min, lam_max] = elliptic_range(a11_grid, a12_grid, a22_grid);

        b1   = draw_normal(0.0, amp_b1);
        b2   = draw_normal(0.0, amp_b2);
        c    = draw_normal(0.0, amp_c);
        beta = draw_normal(0.0, amp_beta);
        f    = draw_normal(0.0, amp_f);
        draw_normal(0.0, amp_g);  % consumed+discarded: keeps the RNG stream identical to the old random-g data
        g    = 0.0;               % zero initial condition u(x,0) = 0 (Eq. 1.4b)
        h    = draw_normal(0.0, amp_h);

        out_path = fullfile(out_dir, sprintf('problem_instance_%05d.mat', i));
        save(out_path, 'radius', 'center', 'T', 'b1', 'b2', 'c', 'beta', 'f', 'h', 'g', ...
            'xg', 'yg', 'a11_grid', 'a12_grid', 'a22_grid', 'corr1_grid', 'corr2_grid', ...
            'ngrid', 'lipschitz_L_est');

        fprintf('[%s %d/%d] wrote %s  Lipschitz_L~=%.4f  eig range=[%.4f, %.4f]\n', ...
            split_name, i + 1, n, out_path, lipschitz_L_est, lam_min, lam_max);

        global_idx = global_idx + 1;
    end
end

fprintf('\ndone: wrote %d train + %d val + %d test = %d instance(s) under %s\n', ...
    n_train, n_val, n_test, n_train + n_val + n_test, out_root);

end


function [a11_grid, a12_grid, a22_grid, xg, yg] = draw_corner_bilinear_a_ij( ...
    center, radius, ngrid, eig_low, eig_high)
% Draw 4 independent elliptic tensors, one per corner of the square
% bounding box [cx-radius,cx+radius] x [cy-radius,cy+radius], and bilinearly
% interpolate a11/a12/a22 across a fine ngrid x ngrid grid on that square.
cx = center(1); cy = center(2);
xcorners = [cx - radius, cx + radius];
ycorners = [cy - radius, cy + radius];

% corner_*(i,j) <-> (xcorners(j), ycorners(i)) -- interp2's convention.
corner_a11 = zeros(2, 2); corner_a12 = zeros(2, 2); corner_a22 = zeros(2, 2);
for i = 1:2
    for j = 1:2
        [a11, a12, a22] = draw_a_ij(eig_low, eig_high);
        corner_a11(i, j) = a11;
        corner_a12(i, j) = a12;
        corner_a22(i, j) = a22;
    end
end

xg = linspace(xcorners(1), xcorners(2), ngrid);
yg = linspace(ycorners(1), ycorners(2), ngrid);
[XG, YG] = meshgrid(xg, yg);

a11_grid = interp2(xcorners, ycorners, corner_a11, XG, YG, 'linear');
a12_grid = interp2(xcorners, ycorners, corner_a12, XG, YG, 'linear');
a22_grid = interp2(xcorners, ycorners, corner_a22, XG, YG, 'linear');
end


function [lam_min, lam_max] = elliptic_range(a11_grid, a12_grid, a22_grid)
tr = a11_grid + a22_grid;
dt = a11_grid .* a22_grid - a12_grid .^ 2;
disc = sqrt(max(0, (tr / 2) .^ 2 - dt));
lam_min = min(tr(:) / 2 - disc(:));
lam_max = max(tr(:) / 2 + disc(:));
end


function [a11, a12, a22] = draw_a_ij(eig_low, eig_high)
lam1 = eig_low + (eig_high - eig_low) * rand();
lam2 = eig_low + (eig_high - eig_low) * rand();
th = pi * rand();
ct = cos(th); st = sin(th);
a11 = lam1 * ct^2 + lam2 * st^2;
a22 = lam1 * st^2 + lam2 * ct^2;
a12 = (lam1 - lam2) * st * ct;
end


function x = draw_normal(mu, sigma)
x = mu + sigma * randn();
end
