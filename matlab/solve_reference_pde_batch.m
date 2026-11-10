function solve_reference_pde_batch(problem_dir, solution_dir)
% solve_reference_pde_batch  Solve every problem_instance_*.mat in
% problem_dir with MATLAB's PDE Toolbox transient (parabolic) solver, and
% write a matching reference_solution_*.mat into solution_dir.
%
% Each problem instance is the parabolic IBVP (Eq. 1.4):
%
%   Lu := sum_ij a_ij(x,t) d2u/dxi dxj + sum_i b_i(x,t) du/dxi + c(x,t) u - du/dt = f  in D_T
%   u(x,0) = g(x)                                                                       on D
%   du/dnu + beta(x,t) u = h                                                            on (dD)_T
%
% with f, g, h, beta, b_1, b_2, c given directly as CONSTANTS (single random
% scalars, export_matlab_problem.py) -- NOT manufactured from a known exact
% solution, and not functions of (x,t). The FEM solution computed here is the
% ground truth used by benchmark_compare.py to score both the Green's-function
% (Levi parametrix) reconstruction and the 2D-CNN operator.
%
% Requires: Partial Differential Equation Toolbox.
%
% Usage:
%   >> solve_reference_pde_batch('problems', 'solutions')
%   >> solve_reference_pde_batch(fullfile('..','dataset','main','matlab_problems','train'), ...
%          fullfile('..','dataset','main','matlab_solutions','train'))
%
% Safe to re-run: instances that already have a reference_solution_*.mat are
% skipped (resume after an interruption instead of re-solving from scratch),
% and a failure on one instance is caught and logged so it doesn't stop the
% rest of the batch (e.g. solving all 1000 of the training pool).

if nargin < 1, problem_dir = 'problems'; end
if nargin < 2, solution_dir = 'solutions'; end
if ~exist(solution_dir, 'dir'), mkdir(solution_dir); end

files = dir(fullfile(problem_dir, 'problem_instance_*.mat'));
fprintf('found %d problem instance(s) in %s\n', numel(files), problem_dir);

n_solved = 0;
n_skipped = 0;
failed = {};

for k = 1:numel(files)
    in_path = fullfile(files(k).folder, files(k).name);
    idx_str = regexp(files(k).name, '\d+', 'match');
    out_path = fullfile(solution_dir, sprintf('reference_solution_%s.mat', idx_str{end}));

    if exist(out_path, 'file')
        fprintf('[%d/%d] skip %s (already solved -> %s)\n', k, numel(files), files(k).name, out_path);
        n_skipped = n_skipped + 1;
        continue;
    end

    fprintf('[%d/%d] solving %s ...\n', k, numel(files), files(k).name);
    try
        solve_one(in_path, out_path);
        n_solved = n_solved + 1;
    catch err
        fprintf('    !! FAILED on %s: %s\n', files(k).name, err.message);
        failed{end + 1} = files(k).name; %#ok<AGROW>
    end
end

fprintf('\ndone: %d solved, %d skipped (already done), %d failed out of %d\n', ...
    n_solved, n_skipped, numel(failed), numel(files));
if ~isempty(failed)
    fprintf('failed instances:\n');
    fprintf('  %s\n', failed{:});
end

end


function solve_one(in_path, out_path)

S = load(in_path);

radius = double(S.radius);
cx = double(S.center(1));
cy = double(S.center(2));
T  = double(S.T);

% a_ij's schema comes in two flavors:
%   - piecewise-CONSTANT: 4 independent random matrices, one per quadrant
%     of the disk split by x=cx, y=cy (export_matlab_problem.py:
%     draw_a_ij_quadrants).
%   - Lipschitz-continuous: a_ij given on a fine grid, obtained by
%     bilinearly interpolating 4 corner tensors across the square bounding
%     box (generate_problems_lipschitz.m). Detected by the presence of
%     'a11_grid'.
is_lipschitz = isfield(S, 'a11_grid');
if is_lipschitz
    xg = double(S.xg(:).');
    yg = double(S.yg(:).');
    a11_grid = double(S.a11_grid);
    a12_grid = double(S.a12_grid);
    a22_grid = double(S.a22_grid);
    corr1_grid = double(S.corr1_grid);
    corr2_grid = double(S.corr2_grid);
else
    a11c = double(S.a11_corners(:).');
    a12c = double(S.a12_corners(:).');
    a22c = double(S.a22_corners(:).');
end

% --- coefficients: every one of b_1,b_2,c,beta,f,h,g is a single constant
% scalar (export_matlab_problem.py), not a function of (x,t). Kept as
% function handles (ignoring their arguments) purely so the PDE-Toolbox
% wiring below stays uniform; MATLAB's implicit scalar expansion broadcasts
% them against state.u/state.ux/state.uy automatically. ---------------------
b1_fun   = @(x1,x2,t) S.b1;
b2_fun   = @(x1,x2,t) S.b2;
c_fun    = @(x1,x2,t) S.c;
beta_fun = @(x1,x2,t) S.beta;
f_fun    = @(x1,x2,t) S.f;
h_fun    = @(x1,x2,t) S.h;
g_fun    = @(x1,x2)   S.g;

% --- geometry: disk of given radius/center ------------------------------
model = createpde();
gd = [1; cx; cy; radius];
geom = decsg(gd);
geometryFromEdges(model, geom);

% --- PDE coefficients ----------------------------------------------------
% MATLAB canonical transient form:  d*du/dt - div(c_pde*grad(u)) + a_pde*u = f_pde
% Our equation:  u_t = a_ij*u_xixj + b_i*u_xi + c*u - f   (rearranged from Lu=f,
% NON-divergence form, sum_ij a_ij d2u/dxidxj -- see the file header).
%
% If a_ij is piecewise CONSTANT (one value per quadrant), div(a_ij*grad(u)) =
% a_ij*u_xixj (divergence form = non-divergence form) everywhere except
% exactly on the two dividing lines x=cx, y=cy -- a measure-zero set, so no
% correction term is needed (same shortcut as the single-constant case).
% Elements of the mesh straddling a quadrant boundary just get whichever
% quadrant's a_ij each quadrature point inside them falls into; that's the
% standard way FEM absorbs a jump discontinuity that isn't aligned with the
% mesh, no special interface handling required.
%
% If instead a_ij is Lipschitz-continuous (generate_problems_lipschitz.m),
% it genuinely varies with x, so
%   div(a_ij*grad(u)) = a_ij*u_xixj + sum_i (sum_j d(a_ij)/dxj) * u_xi
%                      = a_ij*u_xixj + Corr_1*u_x1 + Corr_2*u_x2.
% Rearranging Lu=f into MATLAB's divergence form then requires absorbing
% -Corr_i*u_xi into the transport coefficients: b_i_eff = b_i - Corr_i.
% corr1_grid/corr2_grid (precomputed in the generator from the smoothed
% a_ij grid) supply exactly Corr_1, Corr_2 here.
%   d=1, a_pde=0,  f_pde = b1_eff*u_x + b2_eff*u_y + c*u - f
if is_lipschitz
    c_tensor_fun = @(location,state) a_ij_grid_c_tensor(location.x, location.y, xg, yg, a11_grid, a12_grid, a22_grid);
    f_transport_fun = @(location,state) ...
        (b1_fun(location.x, location.y, state.time) - grid_interp(location.x, location.y, xg, yg, corr1_grid)) .* state.ux + ...
        (b2_fun(location.x, location.y, state.time) - grid_interp(location.x, location.y, xg, yg, corr2_grid)) .* state.uy;
else
    c_tensor_fun = @(location,state) a_ij_quadrant_c_tensor(location.x, location.y, cx, cy, a11c, a12c, a22c);
    f_transport_fun = @(location,state) ...
        b1_fun(location.x, location.y, state.time) .* state.ux + ...
        b2_fun(location.x, location.y, state.time) .* state.uy;
end
specifyCoefficients(model, 'm', 0, 'd', 1, ...
    'c', c_tensor_fun, ...
    'a', 0, ...
    'f', @(location,state) ...
        f_transport_fun(location,state) + ...
        c_fun(location.x, location.y, state.time)  .* state.u  - ...
        f_fun(location.x, location.y, state.time));

% --- boundary condition ---------------------------------------------------
% du/dnu + beta*u = h, where nu is the paper's INWARD conormal
% (nu_i(x,t) = sum_j a_ij(x,t) n_j(x), n = inward normal, p.4).
% MATLAB's generalized Neumann/Robin BC is n_out.(c_pde*grad(u)) + q*u = g_bc,
% where n_out is the OUTWARD normal (MATLAB's own convention) -- the opposite
% sign from the paper's inward nu. Since c_pde=a_ij is symmetric,
% n_out.(c_pde*grad(u)) = -sum_j nu_j*u_xj = -du/dnu, so plugging in directly
% would solve du/dnu - beta*u = -h (sign-flipped BC) instead of the intended
% du/dnu + beta*u = h. Negating both q and g compensates for this:
%   -du/dnu + q*u = g_bc  with q=-beta, g_bc=-h
%   => -du/dnu - beta*u = -h  => du/dnu + beta*u = h.  (correct)
applyBoundaryCondition(model, 'neumann', 'Edge', 1:model.Geometry.NumEdges, ...
    'q', @(location,state) -beta_fun(location.x, location.y, state.time), ...
    'g', @(location,state) -h_fun(location.x, location.y, state.time));

% --- initial condition -----------------------------------------------------
setInitialConditions(model, @(location) g_fun(location.x, location.y));

% --- mesh + time stepping ---------------------------------------------------
generateMesh(model, 'Hmax', radius / 25, 'GeometricOrder', 'quadratic');

Nt = 51;                       % much finer than the Python pipeline's N=10,
tlist = linspace(0, T, Nt);    % so this is a trustworthy reference

results = solvepde(model, tlist);

u = results.NodalSolution;     % (numNodes x Nt)
nodes = model.Mesh.Nodes;      % (2 x numNodes)

save(out_path, 'u', 'nodes', 'tlist', 'radius', 'cx', 'cy', 'T');
fprintf('    -> %s  (%d nodes, %d time steps)\n', out_path, size(nodes, 2), Nt);

end


function idx = quadrant_index(x, y, cx, cy)
% Which quadrant each query point falls in, split by x=cx, y=cy:
% 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right -- matches the
% corner order in export_matlab_problem.py:draw_a_ij_quadrants (and
% import_matlab_solution.py's quadrant_index, 0-indexed there).
is_right = x >= cx;
is_top = y >= cy;
idx = ones(size(x));
idx(is_top & ~is_right) = 1;   % TL
idx(is_top & is_right) = 2;    % TR
idx(~is_top & ~is_right) = 3;  % BL
idx(~is_top & is_right) = 4;   % BR
end


function cT = a_ij_quadrant_c_tensor(x, y, cx, cy, a11c, a12c, a22c)
% PDE Toolbox's 4-row [c11;c21;c12;c22] symmetric c-tensor, one column per
% query point, looked up from whichever quadrant each point falls in.
idx = quadrant_index(x, y, cx, cy);
a11 = a11c(idx); a12 = a12c(idx); a22 = a22c(idx);
cT = [a11(:).'; a12(:).'; a12(:).'; a22(:).'];
end


function v = grid_interp(x, y, xg, yg, G)
% Bilinear lookup on the (xg,yg) grid, clamped to the grid extent so
% floating-point roundoff at the disk boundary (which lies exactly on the
% grid's edge, see generate_problems_lipschitz.m) can never fall
% just outside it and return NaN.
xq = min(max(x, xg(1)), xg(end));
yq = min(max(y, yg(1)), yg(end));
v = interp2(xg, yg, G, xq, yq, 'linear');
end


function cT = a_ij_grid_c_tensor(x, y, xg, yg, a11_grid, a12_grid, a22_grid)
% PDE Toolbox's 4-row [c11;c21;c12;c22] symmetric c-tensor, bilinearly
% interpolated from the Lipschitz-continuous a_ij grid.
a11 = grid_interp(x, y, xg, yg, a11_grid);
a12 = grid_interp(x, y, xg, yg, a12_grid);
a22 = grid_interp(x, y, xg, yg, a22_grid);
cT = [a11(:).'; a12(:).'; a12(:).'; a22(:).'];
end
