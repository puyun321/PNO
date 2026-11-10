"""Parabolic Neural Operator (PNO) -- the paper's Sec. 3 architecture
(Figure 3.1 / Figure 3.2, Eq. 3.4-3.5), implemented end-to-end in PyTorch so
that it can be TRAINED against FEM reference solutions by minimizing the
relative L2 error (Sec. 3.5).

Every learnable block is a "2D-CNN with filter size 1" acting on an
(MN)x(MN) array indexed by (p_k; p_l). A filter-size-1 convolution is the
same map applied independently to every array entry, so it is implemented
here as an nn.Linear stack on the channel axis of the gathered (p_k; p_l)
entries -- mathematically identical to Conv2d(kernel_size=1), but it lets us
evaluate only the causal entries (t_k > t_l), since every other entry is set
to 0 by the paper anyway.

Blocks (names follow Figure 3.1 / 3.2):

  N_{sigma1,theta1}      : a_ij(p_l) -> a^ij(p_l)            (ReLU, dashed)
  N^{1/2}_{sigma1,theta1}: a_ij(p_l) -> Cholesky factor of a^ij(p_l)
                           (shares theta1's hidden layers with N_{sigma1,theta1};
                            separate linear output head)
  N_det                  : a_ij(p_l) -> (2 sqrt(pi))^{-d} det(a_ij(p_l))^{-1/2}
  N^breve_{sigma1,theta2}: (Theta^(1), Xi^(6)) -> (t-tau)^{-d/2} exp(-|y|^2/(4(t-tau)))
                           (outputs the log, exponentiated afterwards -- a
                            computation step, keeps the kernel positive)
  N~^(1)_{theta3}        : ProdPoly(Xi^(1..7)) -> polynomial factor of LZ (Eq. 2.35)
  N^hat                  : ProdPoly(Phi^(1..5)) -> P(x,t;eta,sigma) (Eq. 2.40)

Composition (no further learnable parameters):

  N_{s1,t1,t2}(p_k;p_l) = N_det(Xi2) * N^breve(Theta1, Xi6)          (Eq. 2.33)
  N^(1)_{theta3}        = N_{s1,t1,t2} * N~^(1)_{theta3}              (Fig. 3.1)
  N^(m+1)_{theta3}      = sum_h N^(1)(p_k;p_h) N^(m)(p_h;p_l)          (Sec. 3.3)
  N~^(Ne)_{s1,t1,t2}    = N_{s1,t1,t2} + sum_m sum_h N_{s1,t1,t2} N^(m)    (Eq. 3.4)
  N_theta4(p_k)         = sum_l N~(p_k;p_l) f~(p_l)                        (Eq. 3.5)
  N_B                   = N^hat * N_{s1,t1,t2}                             (Fig. 3.2)
  N^(1)_{theta5}        = N_B + sum_m sum_h N_B(q_k;p_h) N^(m)(p_h;q_l)    (Eq. 2.42)
  Psi                   = -1/2 sum_l N^(1)_{theta5}(q_k;p_l) f~(p_l) - h
  phi                   = 2 Psi + 2 sum_m sum_h N^(m)_{theta5}(q_k;q_h) Psi(q_h)
  N_theta6(p_k)         = sum_l N~(p_k;q_l) phi(q_l)
  u                     = -N_theta4 + N_theta6 + g                          (Eq. 2.43)

Every sum above carries its quasi-Monte-Carlo weight (|D|/M * delta for
interior points, |dD|/M * delta for boundary points).

Note the boundary operator follows the paper (Remark 2.2 / Fig. 3.2): P is
applied INSIDE the (eta,sigma)-integral, i.e. to Z before the Neumann series,
not multiplied onto the already-summed Green's function afterwards (which is
what levi_operator.py's M_prefactor * N_tilde does).

Training is two-stage:
  1. `pretrain`: every block is fit by supervised regression to its closed
     form (inv / Cholesky / det / heat kernel / Eq. 2.35 bracket / Eq. 2.40 P)
     on synthetic inputs -- this gives the "initial guess".
  2. `train`: the whole PNO is fine-tuned end-to-end on FEM solutions with
     the relative L2 loss of Sec. 3.5.

Usage:
    python pno.py pretrain --out runs_pno/pretrained.pt
    python pno.py train --dataset main --init runs_pno/pretrained.pt --out runs_pno/main/pno.pt
    python pno.py evaluate --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main
    python pno.py evaluate_ood --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main
    python pno.py plot_samples --dataset main --ckpt runs_pno/main/pno.pt --out results/pno/main
    python pno.py check_exact --sample dataset/main/samples/test/sample_00000.npz
(run from the repository root; every path is relative to it)
"""

import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

D = 2  # spatial dimension
DET_CONST = (2.0 * math.sqrt(math.pi)) ** (-D)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def relu_mlp(in_dim, hidden, out_dim, n_hidden):
    """Filter-size-1 2D-CNN with ReLU (sigma1), as a channel-wise Linear stack."""
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ProdPoly(nn.Module):
    """Product-of-polynomials network ([CMB+22, Figure 7]): no activation
    function, only filter-size-1 linear maps and Hadamard products.

        z_0     = W_0 x + b_0
        z_{n+1} = (A_n z_n) * (B_n z_n) + C_n z_n      (n = 0..L-1)
        out     = w^T z_L + b

    Each multiplicative layer doubles the polynomial degree, so L layers
    represent polynomials of degree <= 2^L (L=3 -> degree 8 >= 7, Sec. 2.4).
    Inputs and output are rescaled by fixed (non-learnable) constants, which
    keeps the map a polynomial while keeping intermediate values O(1).
    """

    def __init__(self, in_dim, width, n_layers, in_scale, out_scale):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.A = nn.ModuleList(nn.Linear(width, width) for _ in range(n_layers))
        self.B = nn.ModuleList(nn.Linear(width, width) for _ in range(n_layers))
        self.C = nn.ModuleList(nn.Linear(width, width) for _ in range(n_layers))
        self.out = nn.Linear(width, 1)
        self.register_buffer("in_scale", torch.as_tensor(in_scale, dtype=torch.float32))
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))
        for lin in list(self.A) + list(self.B):
            nn.init.normal_(lin.weight, std=0.5 / math.sqrt(width))
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        z = self.inp(x / self.in_scale)
        for A, B, C in zip(self.A, self.B, self.C):
            z = A(z) * B(z) + C(z)
        return self.out(z).squeeze(-1) * self.out_scale


# ---------------------------------------------------------------------------
# closed forms (pretraining targets, and `exact` mode)
# ---------------------------------------------------------------------------

def exact_inv(a):
    a11, a12, a22 = a.unbind(-1)
    det = a11 * a22 - a12 ** 2
    return torch.stack([a22 / det, -a12 / det, a11 / det], -1)


def exact_chol_of_inv(a):
    """Lower-triangular (L11, L21, L22) with L L^T = a^{-1}."""
    i11, i12, i22 = exact_inv(a).unbind(-1)
    L11 = torch.sqrt(i11)
    L21 = i12 / L11
    L22 = torch.sqrt(torch.clamp(i22 - L21 ** 2, min=1e-12))
    return torch.stack([L11, L21, L22], -1)


def exact_det_prefactor(a):
    a11, a12, a22 = a.unbind(-1)
    return DET_CONST / torch.sqrt(a11 * a22 - a12 ** 2)


def exact_log_heat(y1, y2, rinv):
    """log[(t-tau)^{-d/2} exp(-|y|^2/(4(t-tau)))], d=2, rinv = 1/(t-tau)."""
    return torch.log(rinv) - (y1 ** 2 + y2 ** 2) * rinv / 4.0


def apply_sym(m, v1, v2):
    """(m11 m12; m12 m22) @ (v1, v2)."""
    m11, m12, m22 = m.unbind(-1)
    return m11 * v1 + m12 * v2, m12 * v1 + m22 * v2


def exact_LZ_bracket(xi):
    """Polynomial factor of LZ, Eq. 2.35, from the 15 Xi channels:
    [a_r(3), a_c(3), ainv_c(3), b_r(2), c_r(1), rinv(1), dx(2)]."""
    a_r, a_c, ai = xi[..., 0:3], xi[..., 3:6], xi[..., 6:9]
    b1, b2, c = xi[..., 9], xi[..., 10], xi[..., 11]
    rinv, dx, dy = xi[..., 12], xi[..., 13], xi[..., 14]
    da = a_r - a_c
    w1, w2 = apply_sym(ai, dx, dy)
    t1 = -0.5 * rinv * (da[..., 0] * ai[..., 0] + 2 * da[..., 1] * ai[..., 1] + da[..., 2] * ai[..., 2])
    t2 = 0.25 * rinv ** 2 * (da[..., 0] * w1 ** 2 + 2 * da[..., 1] * w1 * w2 + da[..., 2] * w2 ** 2)
    t3 = -0.5 * rinv * (b1 * w1 + b2 * w2)
    return t1 + t2 + t3 + c


def exact_P(phi):
    """P(x,t;eta,sigma), Eq. 2.40, from the 9 Phi channels:
    [beta_r(1), rinv(1), ainv_c(3), dx(2), nu_r(2)]."""
    beta, rinv, ai = phi[..., 0], phi[..., 1], phi[..., 2:5]
    dx, dy, nu1, nu2 = phi[..., 5], phi[..., 6], phi[..., 7], phi[..., 8]
    w1, w2 = apply_sym(ai, dx, dy)
    return 2 * beta - rinv * (nu1 * w1 + nu2 * w2)


# ---------------------------------------------------------------------------
# the PNO
# ---------------------------------------------------------------------------

XI_SCALE = [2.0] * 3 + [2.0] * 3 + [1.0] * 3 + [1.0] * 2 + [1.0] + [10.0] + [2.0] * 2
PHI_SCALE = [2.0, 10.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]


class PNO(nn.Module):
    def __init__(self, hidden1=64, hidden_det=32, hidden2=64, pp_width=64, K=6, exact=False):
        super().__init__()
        self.K = K
        self.exact = exact
        # theta1: shared hidden layers, two heads (inverse, Cholesky factor of inverse)
        self.theta1_trunk = relu_mlp(3, hidden1, hidden1, n_hidden=3)
        self.theta1_inv = nn.Linear(hidden1, 3)
        self.theta1_chol = nn.Linear(hidden1, 3)
        self.det_net = relu_mlp(3, hidden_det, 1, n_hidden=3)
        self.heat_net = relu_mlp(3, hidden2, 1, n_hidden=4)          # theta2
        self.theta3 = ProdPoly(15, pp_width, 3, XI_SCALE, out_scale=10.0)
        self.phat = ProdPoly(9, pp_width, 2, PHI_SCALE, out_scale=5.0)

    # --- column-only (filter-size-1, constant along rows) blocks ----------------
    def theta1(self, a):
        if self.exact:
            return exact_inv(a), exact_chol_of_inv(a)
        h = torch.relu(self.theta1_trunk(a))
        return self.theta1_inv(h), self.theta1_chol(h)

    def det(self, a):
        if self.exact:
            return exact_det_prefactor(a)
        return self.det_net(a).squeeze(-1)

    def log_heat(self, y1, y2, rinv):
        if self.exact:
            return exact_log_heat(y1, y2, rinv)
        return self.heat_net(torch.stack([y1, y2, rinv / 10.0], -1)).squeeze(-1)

    def bracket(self, xi):
        return exact_LZ_bracket(xi) if self.exact else self.theta3(xi)

    def P(self, phi):
        return exact_P(phi) if self.exact else self.phat(phi)

    # --- forward on one sample --------------------------------------------------
    def forward(self, geo, s):
        """geo: Geometry (shared mesh); s: dict of per-sample tensors.
        Returns u at the interior mesh points, shape (N*M,)."""
        ni, nb, n = geo.n_int, geo.n_bnd, geo.n_any
        a_any = torch.cat([s["a_int"], s["a_bnd"]], 0)                    # (n,3)
        ainv, L = self.theta1(a_any)                                      # Xi3, N^{1/2}
        detp = self.det(a_any)                                            # N_det

        # N_{s1,t1,t2} on every causal (row, col) pair of the any x any mesh
        r, c = geo.zr, geo.zc
        dx, dy, rinv = geo.zdx, geo.zdy, geo.zrinv
        Lc = L[c]
        y1 = Lc[:, 0] * dx + Lc[:, 1] * dy                                 # Theta1 = L^T (p_k - p_l)
        y2 = Lc[:, 2] * dy
        zval = detp[c] * torch.exp(self.log_heat(y1, y2, rinv))
        Z = torch.zeros(n, n, device=zval.device, dtype=zval.dtype).index_put((r, c), zval)

        # N^(1)_theta3 on (interior row, any col): ProdPoly(Xi) * N_{s1,t1,t2}
        sel = geo.lz_sel                                                   # pairs with interior rows
        rr, cc = r[sel], c[sel]
        xi = torch.cat([s["a_int"][rr], a_any[cc], ainv[cc], s["b_int"][rr], s["c_int"][rr, None],
                        rinv[sel, None], dx[sel, None], dy[sel, None]], -1)
        LZ = torch.zeros(ni, n, device=zval.device, dtype=zval.dtype).index_put((rr, cc), self.bracket(xi) * zval[sel])

        wi, wb = geo.quad_w_int, geo.quad_w_bnd
        LZ_ii = LZ[:, :ni]
        powers = [LZ]
        for _ in range(1, self.K):
            powers.append(wi * (LZ_ii @ powers[-1]))
        series = sum(powers)                                               # sum_m N^(m)_theta3, (ni, n)

        # N~^(Ne)_{s1,t1,t2}, rows = interior only (that's all u needs)
        Z_int = Z[:ni]
        N_tilde = Z_int + wi * (Z_int[:, :ni] @ series)                    # Eq. 3.4

        f = s["f_int"]
        N4 = wi * (N_tilde[:, :ni] @ f)                                    # Eq. 3.5

        # boundary: N_B = N^hat * N_{s1,t1,t2} on (boundary row, any col)
        selb = geo.b_sel
        rb, cb = r[selb] - ni, c[selb]
        phi_in = torch.cat([s["beta_bnd"][rb, None], rinv[selb, None], ainv[cb],
                            dx[selb, None], dy[selb, None], s["nu_bnd"][rb]], -1)
        NB = torch.zeros(nb, n, device=zval.device, dtype=zval.dtype).index_put((rb, cb), self.P(phi_in) * zval[selb])
        N5 = NB + wi * (NB[:, :ni] @ series)                               # Eq. 2.42, (nb, n)
        M_bi, M_bb = N5[:, :ni], N5[:, ni:]
        Psi = -0.5 * wi * (M_bi @ f) - s["h_bnd"]

        Mk = M_bb
        acc = Mk @ Psi
        for _ in range(1, self.K):
            Mk = wb * (M_bb @ Mk)
            acc = acc + Mk @ Psi
        phi = 2 * Psi + 2 * wb * acc

        N6 = wb * (N_tilde[:, ni:] @ phi)
        return -N4 + N6 + s["g_int"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Geometry:
    """Mesh shared by every sample (same Halton points, same time mesh)."""

    def __init__(self, d, device):
        N, M = d["u_interior"].shape
        Mb = d["boundary_xy"].shape[0]
        t_mesh = d["t_mesh"]
        xy = np.concatenate([np.tile(d["interior_xy"], (N, 1)), np.tile(d["boundary_xy"], (N, 1))])
        t = np.concatenate([np.repeat(t_mesh, M), np.repeat(t_mesh, Mb)])
        self.N, self.M, self.Mb = N, M, Mb
        self.n_int, self.n_bnd = N * M, N * Mb
        self.n_any = self.n_int + self.n_bnd
        dt = t[:, None] - t[None, :]
        r, c = np.nonzero(dt > 1e-9)                                     # causal pairs only
        T, radius = float(d["T"]), float(d["radius"])
        dt_mesh = T / (N + 1)
        self.quad_w_int = (math.pi * radius ** 2 / M) * dt_mesh
        self.quad_w_bnd = (2 * math.pi * radius / Mb) * dt_mesh
        tt = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
        self.zr, self.zc = tt(r, torch.long), tt(c, torch.long)
        self.zdx, self.zdy = tt(xy[r, 0] - xy[c, 0]), tt(xy[r, 1] - xy[c, 1])
        self.zrinv = tt(1.0 / dt[r, c])
        self.lz_sel = tt(np.nonzero(r < self.n_int)[0], torch.long)
        self.b_sel = tt(np.nonzero(r >= self.n_int)[0], torch.long)
        self.interior_xy = d["interior_xy"]
        self.radius = radius


def load_sample(path, device):
    d = np.load(path)
    N, M = d["u_interior"].shape
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    st = lambda *ks: tt(np.stack([d[k].reshape(-1) for k in ks], -1))
    normal = np.tile(d["boundary_normal"], (N, 1))
    a11b, a12b, a22b = (d[k].reshape(-1) for k in ("a11_b", "a12_b", "a22_b"))
    nu = np.stack([a11b * normal[:, 0] + a12b * normal[:, 1],
                   a12b * normal[:, 0] + a22b * normal[:, 1]], -1)     # conormal nu_i = a_ij n_j
    return dict(
        a_int=st("a11", "a12", "a22"), a_bnd=st("a11_b", "a12_b", "a22_b"),
        b_int=st("b1", "b2"), c_int=tt(d["c"].reshape(-1)), f_int=tt(d["f_tilde"].reshape(-1)),
        beta_bnd=tt(d["beta"].reshape(-1)), h_bnd=tt(d["h"].reshape(-1)), nu_bnd=tt(nu),
        g_int=tt(np.tile(d["g"], N)), u=tt(d["u_interior"].reshape(-1)),
    )


def rel_l2(u_pred, u_true):
    return torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)


DATASETS = {
    # discontinuous (piecewise-constant per quadrant) a_ij: train pool + separate test
    "main": dict(pool="dataset/main/samples/train", val=None, test="dataset/main/samples/test",
                 ood="dataset/main/samples/ood"),
    # Lipschitz-continuous a_ij: explicit train/val/test
    "lipschitz": dict(pool="dataset/lipschitz/samples/train", val="dataset/lipschitz/samples/val",
                      test="dataset/lipschitz/samples/test", ood="dataset/lipschitz/samples/ood"),
}
# ood: paper Sec. 3.5 OOD shift -- a_ij eigenvalues in [2,3] (training: [1,2]) and
# 2x amplitudes for b1,b2,c,beta,f,h; scored with the trained model, no retraining.


def max_abs_u(path):
    return float(np.abs(np.load(path)["u_interior"]).max())


def _posix(paths):
    """Store paths with '/' so checkpoints written on Windows load on Linux/macOS."""
    return [p.replace("\\", "/") for p in paths]


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    ck["splits"] = {k: _posix(v) if isinstance(v, list) else v for k, v in ck["splits"].items()}
    return ck


def make_splits(dataset, val_frac, max_abs, seed):
    """train / val / test path lists. Blow-up samples (max|u| > max_abs) are
    removed from TRAINING only (they would dominate the loss); val and test
    are scored on every sample."""
    cfg = DATASETS[dataset]
    pool = sorted(glob.glob(os.path.join(cfg["pool"], "sample_*.npz")))
    if cfg["val"] is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pool))
        n_val = int(round(val_frac * len(pool)))
        val = [pool[i] for i in sorted(perm[:n_val])]
        train = [pool[i] for i in sorted(perm[n_val:])]
    else:
        train, val = pool, sorted(glob.glob(os.path.join(cfg["val"], "sample_*.npz")))
    test = sorted(glob.glob(os.path.join(cfg["test"], "sample_*.npz")))
    n_before = len(train)
    if max_abs is not None:
        train = [p for p in train if max_abs_u(p) <= max_abs]
    return dict(train=_posix(train), val=_posix(val), test=_posix(test), n_train_before_filter=n_before)


# ---------------------------------------------------------------------------
# stage 1: pretraining each block on its closed form
# ---------------------------------------------------------------------------

def random_elliptic(n, lo, hi, gen, device):
    lam1 = lo + (hi - lo) * torch.rand(n, generator=gen, device=device)
    lam2 = lo + (hi - lo) * torch.rand(n, generator=gen, device=device)
    th = math.pi * torch.rand(n, generator=gen, device=device)
    ct, st = torch.cos(th), torch.sin(th)
    return torch.stack([lam1 * ct ** 2 + lam2 * st ** 2, (lam1 - lam2) * st * ct,
                        lam1 * st ** 2 + lam2 * ct ** 2], -1)


def random_disk(n, gen, device):
    r = torch.sqrt(torch.rand(n, generator=gen, device=device))
    th = 2 * math.pi * torch.rand(n, generator=gen, device=device)
    return r * torch.cos(th), r * torch.sin(th)


def random_rinv(n, gen, device, N=10, T=1.0):
    k = torch.randint(1, N, (n,), generator=gen, device=device).float()   # t-tau = k*delta
    return (N + 1) / (k * T)


def pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    model = PNO(K=args.K).to(device)
    lo, hi = args.eig_low, args.eig_high

    def fit(name, params, loss_fn, steps, lr):
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for it in range(1, steps + 1):
            loss, _ = loss_fn()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % max(1, steps // 5) == 0 or it == 1:
                print(f"  [{name}] step {it:6d}/{steps}  loss={loss.item():.3e}", flush=True)
        with torch.no_grad():
            _, metric = loss_fn(n=200000)
        print(f"  [{name}] held-out: {metric}", flush=True)
        return metric

    B = args.batch
    report = {}

    # theta1 (inverse + Cholesky heads) and N_det
    def loss_theta1(n=B):
        a = random_elliptic(n, lo, hi, gen, device)
        ti, tc = exact_inv(a), exact_chol_of_inv(a)
        pi, pc = model.theta1(a)
        loss = ((pi - ti) ** 2).mean() + ((pc - tc) ** 2).mean()
        return loss, dict(inv_max_abs_err=(pi - ti).abs().max().item(),
                          chol_max_abs_err=(pc - tc).abs().max().item())
    report["theta1"] = fit("theta1 inv+chol", list(model.theta1_trunk.parameters()) + list(model.theta1_inv.parameters())
                           + list(model.theta1_chol.parameters()), loss_theta1, args.steps, 1e-3)

    def loss_det(n=B):
        a = random_elliptic(n, lo, hi, gen, device)
        t = exact_det_prefactor(a)
        p = model.det(a)
        return ((p - t) ** 2).mean() / DET_CONST ** 2, dict(det_max_rel_err=((p - t).abs() / t).max().item())
    report["det"] = fit("N_det", model.det_net.parameters(), loss_det, args.steps, 1e-3)

    # theta2: heat kernel at y = L^T (x - xi), x, xi in the closed disk
    def heat_inputs(n):
        a = random_elliptic(n, lo, hi, gen, device)
        L = exact_chol_of_inv(a)
        x1, x2 = random_disk(n, gen, device)
        e1, e2 = random_disk(n, gen, device)
        dx, dy = x1 - e1, x2 - e2
        return L[:, 0] * dx + L[:, 1] * dy, L[:, 2] * dy, random_rinv(n, gen, device)

    def loss_heat(n=B):
        y1, y2, rinv = heat_inputs(n)
        t = exact_log_heat(y1, y2, rinv)
        p = model.log_heat(y1, y2, rinv)
        w = torch.exp(t) / torch.exp(t).mean()                    # weight by kernel size
        loss = ((p - t) ** 2 * (w + 0.05)).mean()
        rel = (torch.exp(p) - torch.exp(t)).abs() / torch.exp(t).max()
        return loss, dict(heat_max_err_rel_to_peak=rel.max().item(),
                          heat_mean_err_rel_to_peak=rel.mean().item())
    report["theta2"] = fit("theta2 heat", model.heat_net.parameters(), loss_heat, args.steps_heat, 1e-3)

    # theta3: Eq. 2.35 bracket, weighted by the exact Z it multiplies
    def xi_inputs(n):
        a_r = random_elliptic(n, lo, hi, gen, device)
        a_c = random_elliptic(n, lo, hi, gen, device)
        ai = exact_inv(a_c)
        b = 0.6 * 1.5 * torch.randn(n, 2, generator=gen, device=device)
        c = 0.6 * 1.5 * torch.randn(n, 1, generator=gen, device=device)
        rinv = random_rinv(n, gen, device)
        x1, x2 = random_disk(n, gen, device)
        e1, e2 = random_disk(n, gen, device)
        xi = torch.cat([a_r, a_c, ai, b, c, rinv[:, None], (x1 - e1)[:, None], (x2 - e2)[:, None]], -1)
        L = exact_chol_of_inv(a_c)
        y1 = L[:, 0] * xi[:, 13] + L[:, 1] * xi[:, 14]
        y2 = L[:, 2] * xi[:, 14]
        z = exact_det_prefactor(a_c) * torch.exp(exact_log_heat(y1, y2, rinv))
        return xi, z

    def make_poly_loss(inputs_fn, exact_fn, net_fn):
        def loss_fn(n=B):
            x, z = inputs_fn(n)
            t, p = exact_fn(x), net_fn(x)
            zw = z / z.mean()
            loss = (((p - t) * zw) ** 2).mean() + 1e-3 * ((p - t) ** 2).mean() / (t ** 2).mean()
            weighted_rel = (((p - t) * z) ** 2).sum().sqrt() / ((t * z) ** 2).sum().sqrt()
            return loss, dict(rel_err_of_value_times_Z=weighted_rel.item())
        return loss_fn
    report["theta3"] = fit("theta3 ProdPoly", model.theta3.parameters(),
                           make_poly_loss(xi_inputs, exact_LZ_bracket, model.theta3), args.steps_poly, 1e-3)

    # N^hat: Eq. 2.40 P on (boundary row, any col), weighted by Z
    def phi_inputs(n):
        a_r = random_elliptic(n, lo, hi, gen, device)          # a at the boundary point (for nu)
        a_c = random_elliptic(n, lo, hi, gen, device)
        ai = exact_inv(a_c)
        th = 2 * math.pi * torch.rand(n, generator=gen, device=device)
        x1, x2 = torch.cos(th), torch.sin(th)                  # boundary point (unit circle)
        n1, n2 = -x1, -x2                                      # inward normal
        nu1, nu2 = apply_sym(a_r, n1, n2)
        e1, e2 = random_disk(n, gen, device)
        beta = 0.8 * 1.5 * torch.randn(n, generator=gen, device=device)
        rinv = random_rinv(n, gen, device)
        dx, dy = x1 - e1, x2 - e2
        phi = torch.stack([beta, rinv, ai[:, 0], ai[:, 1], ai[:, 2], dx, dy, nu1, nu2], -1)
        L = exact_chol_of_inv(a_c)
        z = exact_det_prefactor(a_c) * torch.exp(exact_log_heat(L[:, 0] * dx + L[:, 1] * dy, L[:, 2] * dy, rinv))
        return phi, z
    report["phat"] = fit("N^hat ProdPoly", model.phat.parameters(),
                         make_poly_loss(phi_inputs, exact_P, model.phat), args.steps_poly, 1e-3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(state=model.state_dict(), K=args.K, pretrain_report=report), args.out)
    print(f"saved -> {args.out}")


# ---------------------------------------------------------------------------
# stage 2: end-to-end training on FEM
# ---------------------------------------------------------------------------

def score(model, geo, paths, device, return_preds=False):
    model.eval()
    errs, preds = [], []
    with torch.no_grad():
        for p in paths:
            s = load_sample(p, device)
            u = model(geo, s)
            errs.append(rel_l2(u, s["u"]).item())
            if return_preds:
                preds.append(u.cpu().numpy())
    return (np.array(errs), preds) if return_preds else np.array(errs)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    splits = make_splits(args.dataset, args.val_frac, args.max_abs_u, args.split_seed)
    print(f"[{args.dataset}] train={len(splits['train'])} (of {splits['n_train_before_filter']} before "
          f"max|u|<={args.max_abs_u} filter)  val={len(splits['val'])}  test={len(splits['test'])}", flush=True)
    geo = Geometry(np.load(splits["train"][0]), device)

    ck = torch.load(args.init, map_location=device)
    model = PNO(K=ck["K"]).to(device)
    model.load_state_dict(ck["state"])

    print("caching samples on device ...", flush=True)
    train_s = [load_sample(p, device) for p in splits["train"]]

    def mean_rel(paths):
        e = score(model, geo, paths, device)
        return float(np.mean(e)), float(np.median(e))

    init_tr = mean_rel(splits["train"])
    init_va = mean_rel(splits["val"])
    print(f"initial guess (pretrained blocks): train mean={init_tr[0]:.4f} med={init_tr[1]:.4f} | "
          f"val mean={init_va[0]:.4f} med={init_va[1]:.4f}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = [dict(epoch=0, train_mean=init_tr[0], train_median=init_tr[1], val_mean=init_va[0], val_median=init_va[1])]
    best_val, best_state = init_va[0], {k: v.clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(train_s))
        tot = 0.0
        opt.zero_grad()
        for i, idx in enumerate(perm):
            s = train_s[idx]
            loss = rel_l2(model(geo, s), s["u"])
            if not torch.isfinite(loss):
                continue
            (loss / args.accum).backward()
            tot += loss.item()
            if (i + 1) % args.accum == 0 or i == len(perm) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        sched.step()
        va = mean_rel(splits["val"])
        history.append(dict(epoch=ep, train_loss=tot / len(perm), val_mean=va[0], val_median=va[1]))
        print(f"epoch {ep:4d}/{args.epochs}  train relL2={tot / len(perm):.4f}  val mean={va[0]:.4f} "
              f"med={va[1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if va[0] < best_val:
            best_val, best_state = va[0], {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(dict(state=best_state, K=model.K, splits=splits, epoch=ep, args=vars(args),
                            history=history), args.out)
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, K=model.K, splits=splits, args=vars(args), history=history), args.out)
    print(f"done. best val mean relL2={best_val:.4f}  saved -> {args.out}")


# ---------------------------------------------------------------------------
# evaluation: per-sample CSV, violin plot, per-sample FEM vs PNO figures
# ---------------------------------------------------------------------------

def plot_sample(u_pnr, u_fem, geo, title, out_path, n_show=4, grid_res=150):
    """Two rows, u_FEM and u_PNO, at n_show time slices; one colour bar shared by
    both rows (range = this sample's u_FEM min/max)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    N, M, R = geo.N, geo.M, geo.radius
    xy = geo.interior_xy
    gx = np.linspace(-R, R, grid_res)
    GX, GY = np.meshgrid(gx, gx)
    inside = GX ** 2 + GY ** 2 <= R ** 2

    def interp(v):
        g = griddata(xy, v, (GX, GY), method="linear")
        g = np.where(np.isnan(g), griddata(xy, v, (GX, GY), method="nearest"), g)
        return np.where(inside, g, np.nan)

    uf, up = u_fem.reshape(N, M), u_pnr.reshape(N, M)
    # skip the first level: no earlier quadrature node, so u_PNO = g = 0 there
    idxs = np.linspace(1, N - 1, min(n_show, N - 1)).astype(int)
    vmin, vmax = uf.min(), uf.max()
    rows = [(uf, r"$u_{\mathrm{FEM}}$"), (up, r"$u_{\mathrm{PNO}}$")]
    th = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, len(idxs), figsize=(3.0 * len(idxs), 5.8), constrained_layout=True)
    im = None
    for ri, (field, name) in enumerate(rows):
        for ci, ti in enumerate(idxs):
            ax = axes[ri, ci]
            im = ax.imshow(interp(field[ti]), extent=[-R, R, -R, R], origin="lower", cmap="RdBu_r",
                           vmin=vmin, vmax=vmax)
            ax.plot(R * np.cos(th), R * np.sin(th), "k", lw=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t = {(ti + 1) / (N + 1):.2f}", fontsize=22)
        axes[ri, 0].set_ylabel(name, fontsize=26)
    shown = up[idxs]                       # PNO values outside the FEM range -> colour-bar arrows
    over, under = shown.max() > vmax, shown.min() < vmin
    ext = "both" if over and under else "max" if over else "min" if under else "neither"
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, extend=ext)
    cb.ax.tick_params(labelsize=16)
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "validation": "#eb6834",
                "test": "#1baf7a", "ood": "#eda100"}
EXTRA_COLORS = ["#c1495e", "#7a5195", "#4f8a8b", "#374785"]


def violin(groups, out_path, title):
    """groups: list of (label, errors array). Styled like
    results/violin_train_val_test_ood.png: per-split colour (from the label's
    last line), jittered samples, IQR bar + median dot with its value, and a
    shaded band over the held-out (Test / OOD) groups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

    data = [np.asarray(g[1]) for g in groups]
    keys = [g[0].split("\n")[-1].strip().lower() for g in groups]
    extra = iter(EXTRA_COLORS)
    colors = [SPLIT_COLORS.get(k) or next(extra) for k in keys]
    pos = np.arange(1, len(groups) + 1)

    fig, ax = plt.subplots(figsize=(max(7.5, 4.0 + 1.9 * len(groups)), 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # density estimated in log10 space (the axis is log): a linear-space KDE
    # lets one huge outlier inflate the bandwidth and paint mass where no
    # samples are.
    from scipy.stats import gaussian_kde
    for x0, v, c in zip(pos, data, colors):
        lv = np.log10(v)
        grid = np.linspace(lv.min(), lv.max(), 400)
        dens = gaussian_kde(lv)(grid)
        half = 0.36 * dens / dens.max()
        ax.fill_betweenx(10 ** grid, x0 - half, x0 + half, facecolor=c, edgecolor=c,
                         alpha=0.55, linewidth=1.2, zorder=1)
    rng = np.random.default_rng(0)
    for x0, v, c in zip(pos, data, colors):
        ax.scatter(x0 + rng.uniform(-0.09, 0.09, len(v)), v, s=7, color=c, alpha=0.35, linewidths=0, zorder=2)
    for x0, v in zip(pos, data):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([x0, x0], [q1, q3], color=INK, lw=2, zorder=4, solid_capstyle="round")
        ax.scatter([x0], [med], s=34, color=SURFACE, edgecolor=INK, linewidth=1.6, zorder=5)
        ax.annotate(f"{med:.3f}", (x0, med), xytext=(11, 0), textcoords="offset points",
                    fontsize=10, color=INK, va="center", ha="left", fontweight="medium")

    ax.set_yscale("log")
    lo = min(v.min() for v in data); hi = max(v.max() for v in data)
    ax.set_ylim(lo * 0.8, hi * 1.8)
    ax.set_ylabel("relative $L^2$ error  (log scale)", fontsize=12, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{g[0]}\n(n={len(g[1])})" for g in groups], fontsize=11.5, color=INK)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", colors=INK, length=0)
    ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.9, zorder=0)
    ax.yaxis.grid(True, which="minor", color=GRID, linewidth=0.5, zorder=0, alpha=0.6)
    ax.set_axisbelow(True)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "left")
    ax.spines["left"].set_color(BASELINE)

    held = [x0 for x0, k in zip(pos, keys) if k in ("test", "ood")]
    if held:
        a, b = min(held) - 0.5, max(held) + 0.5
        ax.axvspan(a, b, color="#000000", alpha=0.025, zorder=0)
        ax.text((a + b) / 2, hi * 1.6, "held out from training", ha="center", va="top",
                fontsize=9.5, color=MUTED, style="italic")
    ax.set_xlim(0.4, len(groups) + 0.6)
    ax.set_title(title, fontsize=13, color=INK, pad=14, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_ckpt(args.ckpt, device)
    splits = ck["splits"]
    geo = Geometry(np.load(splits["test"][0]), device)
    model = PNO(K=ck["K"]).to(device)
    model.load_state_dict(ck["state"])
    os.makedirs(args.out, exist_ok=True)

    rows, groups, preds_all = [], [], {}
    for split in ("train", "val", "test"):
        errs, preds = score(model, geo, splits[split], device, return_preds=True)
        preds_all[split] = preds
        groups.append((split.capitalize() if split != "val" else "Validation", errs))
        for p, e in zip(splits[split], errs):
            rows.append(dict(split=split, sample=os.path.relpath(p), rel_l2=f"{e:.6f}", max_abs_u_fem=f"{max_abs_u(p):.4f}"))
        print(f"{split:5s} n={len(errs):4d}  mean={errs.mean():.4f}  median={np.median(errs):.4f}  "
              f"std={errs.std():.4f}  min={errs.min():.4f}  max={errs.max():.4f}")
    with open(os.path.join(args.out, "performance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {g[0]: dict(n=len(g[1]), mean=float(g[1].mean()), median=float(np.median(g[1])),
                          std=float(g[1].std()), min=float(g[1].min()), max=float(g[1].max())) for g in groups}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    violin(groups, os.path.join(args.out, "violin.png"), f"PNO vs FEM -- {args.dataset}")

    # per-sample FEM vs PNO figures for every test sample
    sdir = os.path.join(args.out, "test_samples")
    os.makedirs(sdir, exist_ok=True)
    for p, e, up in zip(splits["test"], groups[2][1], preds_all["test"]):
        uf = np.load(p)["u_interior"].reshape(-1)
        name = os.path.splitext(os.path.basename(p))[0]
        plot_sample(up, uf, geo, f"{args.dataset} test {name}   rel. L2 = {e:.4f}", os.path.join(sdir, f"{name}.png"))
    print(f"wrote {args.out}/performance.csv, summary.json, violin.png, test_samples/*.png")


def evaluate_ood(args):
    """Score a trained checkpoint on the dataset's OOD split (no retraining):
    performance.csv (split=ood), summary.json, violin.png, ood_samples/*.png."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_ckpt(args.ckpt, device)
    paths = sorted(glob.glob(os.path.join(DATASETS[args.dataset]["ood"], "sample_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no OOD samples in {DATASETS[args.dataset]['ood']}")
    geo = Geometry(np.load(paths[0]), device)
    model = PNO(K=ck["K"]).to(device)
    model.load_state_dict(ck["state"])
    os.makedirs(args.out, exist_ok=True)
    errs, preds = score(model, geo, paths, device, return_preds=True)
    rows = [dict(split="ood", sample=os.path.relpath(p), rel_l2=f"{e:.6f}", max_abs_u_fem=f"{max_abs_u(p):.4f}")
            for p, e in zip(paths, errs)]
    with open(os.path.join(args.out, "performance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = dict(OOD=dict(n=len(errs), mean=float(errs.mean()), median=float(np.median(errs)),
                            std=float(errs.std()), min=float(errs.min()), max=float(errs.max())))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"ood   n={len(errs):4d}  mean={errs.mean():.4f}  median={np.median(errs):.4f}  "
          f"std={errs.std():.4f}  min={errs.min():.4f}  max={errs.max():.4f}")
    violin([("OOD", errs)], os.path.join(args.out, "violin.png"), f"PNO vs FEM -- {args.dataset} OOD")
    sdir = os.path.join(args.out, "ood_samples")
    os.makedirs(sdir, exist_ok=True)
    for p, e, up in zip(paths, errs, preds):
        uf = np.load(p)["u_interior"].reshape(-1)
        name = os.path.splitext(os.path.basename(p))[0]
        plot_sample(up, uf, geo, f"{args.dataset} OOD {name}   rel. L2 = {e:.4f}", os.path.join(sdir, f"{name}.png"))
    print(f"wrote {args.out}/performance.csv, summary.json, violin.png, ood_samples/*.png")


def plot_samples_fixed(args):
    """Redraw every test and OOD per-sample figure (u_FEM and u_PNO rows)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_ckpt(args.ckpt, device)
    sets = [(ck["splits"]["test"], os.path.join(args.out, "test_samples"), "test")]
    ood = sorted(glob.glob(os.path.join(DATASETS[args.dataset]["ood"], "sample_*.npz")))
    if ood:
        sets.append((ood, os.path.join(args.out, "ood", "ood_samples"), "OOD"))
    geo = Geometry(np.load(sets[0][0][0]), device)
    model = PNO(K=ck["K"]).to(device)
    model.load_state_dict(ck["state"])
    for paths, sdir, tag in sets:
        errs, preds = score(model, geo, paths, device, return_preds=True)
        os.makedirs(sdir, exist_ok=True)
        for p, e, up in zip(paths, errs, preds):
            uf = np.load(p)["u_interior"].reshape(-1)
            name = os.path.splitext(os.path.basename(p))[0]
            plot_sample(up, uf, geo, f"{args.dataset} {tag} {name}   rel. L2 = {e:.4f}",
                        os.path.join(sdir, f"{name}.png"))
        print(f"  wrote {len(paths)} figures -> {sdir}")


def violin_combined(args):
    """--inputs LABEL=a.csv[+b.csv...]: every split present in a label's
    csv(s) (train/val/test/ood) becomes one violin."""
    groups = []
    for spec in args.inputs:
        label, paths = spec.split("=", 1)
        by = {}
        for path in paths.split("+"):
            for r in csv.DictReader(open(path, newline="")):
                by.setdefault(r["split"], []).append(float(r["rel_l2"]))
        for split, nice in (("train", "Train"), ("val", "Validation"), ("test", "Test"), ("ood", "OOD")):
            if split in by:
                groups.append((f"{label}\n{nice}" if label else nice, np.array(by[split])))
    violin(groups, args.out, args.title)
    print(f"saved -> {args.out}")


def check_exact(args):
    """Sanity check: with every block replaced by its closed form, the PNO
    should reproduce levi_operator.py --exact (up to the boundary-operator
    placement difference noted in the module docstring)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.sample)
    geo = Geometry(d, device)
    model = PNO(K=args.K, exact=True).to(device)
    s = load_sample(args.sample, device)
    with torch.no_grad():
        u = model(geo, s)
    print(f"PNO (exact blocks)     rel L2 vs FEM = {rel_l2(u, s['u']).item():.4f}")
    import levi_operator as lo
    u_lo, u_ex, _ = lo.solve(args.sample, lo.SubNetworks(use_exact=True), args.K)
    print(f"levi_operator --exact  rel L2 vs FEM = {np.linalg.norm(u_lo - u_ex) / np.linalg.norm(u_ex):.4f}")
    print(f"|PNO - levi| / |levi|  = {np.linalg.norm(u.cpu().numpy() - u_lo) / np.linalg.norm(u_lo):.4e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pretrain")
    pp.add_argument("--out", default="runs_pno/pretrained.pt")
    pp.add_argument("--K", type=int, default=6)
    pp.add_argument("--eig_low", type=float, default=0.9)
    pp.add_argument("--eig_high", type=float, default=2.1)
    pp.add_argument("--batch", type=int, default=16384)
    pp.add_argument("--steps", type=int, default=8000)
    pp.add_argument("--steps_heat", type=int, default=15000)
    pp.add_argument("--steps_poly", type=int, default=20000)
    pp.add_argument("--seed", type=int, default=0)

    pt = sub.add_parser("train")
    pt.add_argument("--dataset", choices=list(DATASETS), required=True)
    pt.add_argument("--init", default="runs_pno/pretrained.pt")
    pt.add_argument("--out", required=True)
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--accum", type=int, default=8, help="samples per optimizer step")
    pt.add_argument("--val_frac", type=float, default=0.1)
    pt.add_argument("--max_abs_u", type=float, default=10.0)
    pt.add_argument("--split_seed", type=int, default=0)
    pt.add_argument("--seed", type=int, default=0)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--dataset", choices=list(DATASETS), required=True)
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--out", required=True)

    po = sub.add_parser("evaluate_ood")
    po.add_argument("--dataset", choices=list(DATASETS), required=True)
    po.add_argument("--ckpt", required=True)
    po.add_argument("--out", required=True)

    pf = sub.add_parser("plot_samples")
    pf.add_argument("--dataset", choices=list(DATASETS), required=True)
    pf.add_argument("--ckpt", required=True)
    pf.add_argument("--out", required=True, help="results dir, e.g. results/pno/main")

    pv = sub.add_parser("violin")
    pv.add_argument("--inputs", nargs="+", required=True, metavar="LABEL=performance.csv")
    pv.add_argument("--out", required=True)
    pv.add_argument("--title", default="PNO vs FEM: overall performance")

    pc = sub.add_parser("check_exact")
    pc.add_argument("--sample", required=True)
    pc.add_argument("--K", type=int, default=6)

    args = p.parse_args()
    dict(pretrain=pretrain, train=train, evaluate=evaluate, evaluate_ood=evaluate_ood, plot_samples=plot_samples_fixed, violin=violin_combined, check_exact=check_exact)[args.cmd](args)


if __name__ == "__main__":
    main()
