"""Direct, from-the-paper implementation of the Levi-parametrix neural
operator (Sec. 2 and Sec. 3 of PY_PZ_JN_parabolic.pdf), evaluated against
one sample produced by generate_samples.py.

Only two components are gradient-trained (subnetworks.py):
  - N_sigma1,theta1 : matrix inversion (Sec. 2.3.1, p.11)
  - N_sigma1,theta2 : regularized heat kernel (Sec. 2.3.2, p.12)
Everything below is the DETERMINISTIC construction built on top of them:

  Z, the principal part (Eq. 2.2)
    -> LZ (Eq. 2.35)
    -> the Neumann/Volterra series (LZ)_k (Eq. 2.7-2.8)
    -> the approximate Green's function N~ (Eq. 2.38)
    -> the volume potential N_theta4 (Eq. 3.5)
    -> the boundary operator M, its own Neumann series M_k, and the
       boundary density phi (Sec. 2.2, Eq. 2.21-2.26)
    -> the boundary potential N_theta6 (Sec. 3.4)
    -> the final solution  u = -N_theta4 + N_theta6 + g   (Eq. 2.40)

Run with --exact to bypass the two trained sub-networks and use closed-form
matrix inversion / heat kernel instead. This isolates the Levi-parametrix
METHOD's own error (mesh size M/Mb, series truncation K) from the extra
error introduced by approximating N_sigma1,theta1 / N_sigma1,theta2 with
actual trained networks.

Usage:
    python levi_operator.py --sample dataset/main/samples/test/sample_00000.npz --K 6 --exact
"""

import argparse

import numpy as np
import torch

from subnetworks import ReLUMLP


# ---------------------------------------------------------------------------
# the two sub-networks (or exact closed forms)
# ---------------------------------------------------------------------------

def exact_inv(a11, a12, a22):
    det = a11 * a22 - a12 ** 2
    return a22 / det, -a12 / det, a11 / det


def cholesky2x2(ai11, ai12, ai22):
    """Lower-triangular L with L L^T = Ainv (closed form, vectorized)."""
    L11 = np.sqrt(np.maximum(ai11, 1e-12))
    L21 = ai12 / L11
    L22 = np.sqrt(np.maximum(ai22 - L21 ** 2, 1e-12))
    return L11, L21, L22


class SubNetworks:
    def __init__(self, ckpt_path=None, device="cpu", use_exact=False):
        self.use_exact = use_exact
        self.device = device
        if not use_exact:
            ckpt = torch.load(ckpt_path, map_location=device)
            self.inv_net = ReLUMLP(3, 96, 3, n_hidden=4).to(device)
            self.inv_net.load_state_dict(ckpt["inv_state"])
            self.inv_net.eval()
            self.heat_net = ReLUMLP(3, 128, 1, n_hidden=4).to(device)
            self.heat_net.load_state_dict(ckpt["heat_state"])
            self.heat_net.eval()

    def inv(self, a11, a12, a22):
        """(a_ij) -> (a^ij), Sec. 2.3.1."""
        if self.use_exact:
            return exact_inv(a11, a12, a22)
        shp = a11.shape
        x = torch.as_tensor(np.stack([a11.ravel(), a12.ravel(), a22.ravel()], axis=-1),
                             dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.inv_net(x).cpu().numpy()
        return out[:, 0].reshape(shp), out[:, 1].reshape(shp), out[:, 2].reshape(shp)

    def log_w(self, y1, y2, s):
        """log w(y,s), w(y,s)=exp(-|y|^2/s)/s, Sec. 2.3.2."""
        if self.use_exact:
            return -(y1 ** 2 + y2 ** 2) / s - np.log(s)
        shp = y1.shape
        x = torch.as_tensor(np.stack([y1.ravel(), y2.ravel(), s.ravel()], axis=-1),
                             dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.heat_net(x).cpu().numpy().reshape(shp)
        return out


# ---------------------------------------------------------------------------
# Z (Eq. 2.2) and LZ (Eq. 2.35)
#
# Derivation used to plug the heat-kernel sub-network in: writing
# Z(x,t;xi,tau) = A(xi,tau) w^{xi,tau}(x,t;xi,tau) (Eq. 2.2) and comparing
# to w(y,s)=s^{-1}exp(-|y|^2/s) (Sec 2.3.2, d=2) gives, with y = Cholesky
# factor of a^{ij}(xi,tau) applied to (x-xi):
#     Z(x,t;xi,tau) = [1/(pi*sqrt(det a_ij(xi,tau)))] * w(y, 4*(t-tau))
# (the a_ij determinant uses the EXACT, given coefficient function -- only
# its INVERSE a^{ij} is what N_sigma1,theta1 approximates).
# ---------------------------------------------------------------------------

def Z_matrix(nets, xr, tr, xc, tc, a11c, a12c, a22c):
    """Z(row=eval; col=freeze), using a_ij at the freeze (column) points.
    Returns (Z, causal_mask), shape (nr,nc); zero where tr<=tc.
    """
    dt = tr[:, None] - tc[None, :]
    causal = dt > 1e-9
    dt_safe = np.where(causal, dt, 1.0)

    ai11, ai12, ai22 = nets.inv(a11c, a12c, a22c)
    L11, L21, L22 = cholesky2x2(ai11, ai12, ai22)

    dx = xr[:, None, 0] - xc[None, :, 0]
    dy = xr[:, None, 1] - xc[None, :, 1]
    y1 = L11[None, :] * dx + L21[None, :] * dy
    y2 = L22[None, :] * dy
    s = 4.0 * dt_safe

    logw = nets.log_w(y1, y2, s)
    det_a = a11c * a22c - a12c ** 2
    Z = np.exp(logw) / (np.pi * np.sqrt(det_a)[None, :])
    return np.where(causal, Z, 0.0), causal


def LZ_matrix(nets, xr, tr, a11r, a12r, a22r, b1r, b2r, cr, xc, tc, a11c, a12c, a22c):
    """LZ(row=eval point; col=freeze point). Row MUST be an interior point
    (needs a_ij,b_i,c evaluated AT the row point, Eq. 2.35)."""
    dt = tr[:, None] - tc[None, :]
    causal = dt > 1e-9
    dt_safe = np.where(causal, dt, 1.0)

    ai11, ai12, ai22 = nets.inv(a11c, a12c, a22c)

    dx = xr[:, None, 0] - xc[None, :, 0]   # eta - xi
    dy = xr[:, None, 1] - xc[None, :, 1]
    w1 = ai11[None, :] * dx + ai12[None, :] * dy
    w2 = ai12[None, :] * dx + ai22[None, :] * dy

    d_a11 = a11r[:, None] - a11c[None, :]
    d_a12 = a12r[:, None] - a12c[None, :]
    d_a22 = a22r[:, None] - a22c[None, :]

    term1 = -1.0 / (2 * dt_safe) * (d_a11 * ai11[None, :] + 2 * d_a12 * ai12[None, :] + d_a22 * ai22[None, :])
    term2 = 1.0 / (4 * dt_safe ** 2) * (d_a11 * w1 ** 2 + 2 * d_a12 * w1 * w2 + d_a22 * w2 ** 2)
    term3 = -1.0 / (2 * dt_safe) * (b1r[:, None] * w1 + b2r[:, None] * w2)
    term4 = cr[:, None] + np.zeros_like(dt)
    bracket = term1 + term2 + term3 + term4

    Zmat, _ = Z_matrix(nets, xr, tr, xc, tc, a11c, a12c, a22c)
    return np.where(causal, bracket * Zmat, 0.0)


# ---------------------------------------------------------------------------
# Neumann/Volterra series (Eq. 2.7-2.8) -> approximate Green's function (Eq. 2.38)
# ---------------------------------------------------------------------------

def build_LZ_powers(nets, interior, any_pts, K, quad_w_int):
    LZ1 = LZ_matrix(nets, interior["xy"], interior["t"], interior["a11"], interior["a12"], interior["a22"],
                     interior["b1"], interior["b2"], interior["c"],
                     any_pts["xy"], any_pts["t"], any_pts["a11"], any_pts["a12"], any_pts["a22"])
    n_int = interior["xy"].shape[0]
    LZ_ii = LZ1[:, :n_int]  # interior-interior propagator (Eq. 2.8's connecting kernel)
    powers = [LZ1]
    for _ in range(1, K):
        powers.append(quad_w_int * (LZ_ii @ powers[-1]))
    return powers


def build_greens_function(nets, any_pts, n_int, LZ_powers, quad_w_int):
    """N~^(K)(row, col) over ALL mesh points (Eq. 2.38)."""
    Z_full, _ = Z_matrix(nets, any_pts["xy"], any_pts["t"], any_pts["xy"], any_pts["t"],
                          any_pts["a11"], any_pts["a12"], any_pts["a22"])
    Z_eta = Z_full[:, :n_int]  # Z(any eval; interior freeze) -- combines with (LZ)_k
    N_tilde = Z_full.copy()
    for LZk in LZ_powers:
        N_tilde = N_tilde + quad_w_int * (Z_eta @ LZk)
    return N_tilde


# ---------------------------------------------------------------------------
# boundary operator M (Eq. 2.39), Neumann series M_k (Eq. 2.26), density phi
# ---------------------------------------------------------------------------

def M_prefactor(nets, x_xy, t_x, a11x, a12x, a22x, normal_x, beta_x, xi_xy, tau, a11f, a12f, a22f):
    """2*beta(x,t) - (1/(t-tau)) * nu(x,t) . (a^ij(freeze) (x-xi))  (Eq. 2.39),
    where nu_i(x,t) = sum_j a_ij(x,t) n_j(x) is the CONORMAL (p.4) -- built
    from a_ij AT THE EVAL POINT x, not the freeze point xi."""
    dt = t_x[:, None] - tau[None, :]
    dt_safe = np.where(dt > 1e-9, dt, 1.0)
    ai11, ai12, ai22 = nets.inv(a11f, a12f, a22f)
    dx = x_xy[:, None, 0] - xi_xy[None, :, 0]
    dy = x_xy[:, None, 1] - xi_xy[None, :, 1]
    w1 = ai11[None, :] * dx + ai12[None, :] * dy
    w2 = ai12[None, :] * dx + ai22[None, :] * dy
    nu1 = a11x * normal_x[:, 0] + a12x * normal_x[:, 1]
    nu2 = a12x * normal_x[:, 0] + a22x * normal_x[:, 1]
    ndotw = nu1[:, None] * w1 + nu2[:, None] * w2
    return 2 * beta_x[:, None] - ndotw / dt_safe


# ---------------------------------------------------------------------------
# assemble points from a generate_samples.py sample
# ---------------------------------------------------------------------------

def load_points(path):
    d = np.load(path)
    N, M = d["u_interior"].shape
    Mb = d["boundary_xy"].shape[0]
    t_mesh = d["t_mesh"]

    interior = dict(
        xy=np.tile(d["interior_xy"], (N, 1)),
        t=np.repeat(t_mesh, M),
        a11=d["a11"].reshape(-1), a12=d["a12"].reshape(-1), a22=d["a22"].reshape(-1),
        b1=d["b1"].reshape(-1), b2=d["b2"].reshape(-1), c=d["c"].reshape(-1),
        f=d["f_tilde"].reshape(-1), u=d["u_interior"].reshape(-1), n=N * M,
    )
    boundary = dict(
        xy=np.tile(d["boundary_xy"], (N, 1)),
        t=np.repeat(t_mesh, Mb),
        a11=d["a11_b"].reshape(-1), a12=d["a12_b"].reshape(-1), a22=d["a22_b"].reshape(-1),
        beta=d["beta"].reshape(-1), h=d["h"].reshape(-1),
        normal=np.tile(d["boundary_normal"], (N, 1)),
        u=d["u_boundary"].reshape(-1), n=N * Mb,
    )
    meta = dict(radius=float(d["radius"]), T=float(d["T"]), N=N, M=M, Mb=Mb,
                g=d["g"], interior_xy0=d["interior_xy"], t_mesh=t_mesh)
    return interior, boundary, meta


def solve(sample_path, nets, K):
    interior, boundary, meta = load_points(sample_path)
    n_int, n_bnd = interior["n"], boundary["n"]

    any_pts = dict(
        xy=np.concatenate([interior["xy"], boundary["xy"]]),
        t=np.concatenate([interior["t"], boundary["t"]]),
        a11=np.concatenate([interior["a11"], boundary["a11"]]),
        a12=np.concatenate([interior["a12"], boundary["a12"]]),
        a22=np.concatenate([interior["a22"], boundary["a22"]]),
    )

    area = np.pi * meta["radius"] ** 2
    arc_len = 2 * np.pi * meta["radius"]
    dt_mesh = meta["T"] / (meta["N"] + 1)
    quad_w_int = (area / meta["M"]) * dt_mesh
    quad_w_bnd = (arc_len / meta["Mb"]) * dt_mesh

    print(f"mesh: N={meta['N']} M={meta['M']} Mb={meta['Mb']}  "
          f"(n_interior={n_int}, n_boundary={n_bnd}, n_any={n_int + n_bnd})  K={K}")

    LZ_powers = build_LZ_powers(nets, interior, any_pts, K, quad_w_int)
    N_tilde = build_greens_function(nets, any_pts, n_int, LZ_powers, quad_w_int)

    Nii = N_tilde[:n_int, :n_int]   # 內部 evaluate ← 內部 source
    Nib = N_tilde[:n_int, n_int:]   # 內部 evaluate ← 邊界 source
    Nbi = N_tilde[n_int:, :n_int]   # 邊界 evaluate ← 內部 source
    Nbb = N_tilde[n_int:, n_int:]   # 邊界 evaluate ← 邊界 source


    # volume potential N_theta4 (Eq. 3.5) # # 從 Ñ 切出「內部-內部」那一塊
    N_theta4 = quad_w_int * (Nii @ interior["f"]) # 對 f̃ 做體積分

    # boundary operator M and Psi (Eq. 2.21, 2.39)
    M_bb = M_prefactor(nets, boundary["xy"], boundary["t"], boundary["a11"], boundary["a12"], boundary["a22"],
                        boundary["normal"], boundary["beta"],
                        boundary["xy"], boundary["t"], boundary["a11"], boundary["a12"], boundary["a22"]) * Nbb
    M_bi = M_prefactor(nets, boundary["xy"], boundary["t"], boundary["a11"], boundary["a12"], boundary["a22"],
                        boundary["normal"], boundary["beta"],
                        interior["xy"], interior["t"], interior["a11"], interior["a12"], interior["a22"]) * Nbi
    Psi = -0.5 * quad_w_int * (M_bi @ interior["f"]) - boundary["h"]

    # Neumann series for the boundary density phi (Eq. 2.25-2.26)
    Mk_list = [M_bb]
    for _ in range(1, K):
        Mk_list.append(quad_w_bnd * (M_bb @ Mk_list[-1]))
    phi = 2 * Psi + 2 * quad_w_bnd * sum(Mk @ Psi for Mk in Mk_list)

    # boundary (single-layer) potential N_theta6
    N_theta6 = quad_w_bnd * (Nib @ phi)

    g_vec = np.tile(meta["g"], meta["N"])
    u_approx = -N_theta4 + N_theta6 + g_vec
    return u_approx, interior["u"], meta


def plot_result(u_approx, u_exact, meta, out_path, n_show=4):
    """Scatter-plot u_exact / u_approx / error side by side over a few time
    slices, in the same style as visualize.py's sample preview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, M = meta["N"], meta["M"]
    xy = meta["interior_xy0"]
    radius = meta["radius"]
    theta = np.linspace(0, 2 * np.pi, 200)
    bx, by = radius * np.cos(theta), radius * np.sin(theta)

    u_exact_g = u_exact.reshape(N, M)
    u_approx_g = u_approx.reshape(N, M)
    err_g = u_approx_g - u_exact_g

    idxs = np.linspace(0, N - 1, min(n_show, N)).astype(int)
    u_vmin, u_vmax = u_exact_g.min(), u_exact_g.max()
    err_absmax = np.abs(err_g).max()

    rows = [(u_exact_g, "u_exact", "RdBu_r", u_vmin, u_vmax),
            (u_approx_g, "u_approx", "RdBu_r", u_vmin, u_vmax),
            (err_g, "u_approx - u_exact", "RdBu_r", -err_absmax, err_absmax)]

    row_labels = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(3, len(idxs), figsize=(3.2 * len(idxs), 9), constrained_layout=True)
    for row, (field, name, cmap, vmin, vmax) in enumerate(rows):
        sc = None
        for col, ti in enumerate(idxs):
            ax = axes[row, col]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=field[ti], s=10, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.plot(bx, by, color="k", lw=1)
            ax.set_aspect("equal")
            ax.set_title(f"{name}, t idx={ti}")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=14, fontweight="bold", rotation=0, labelpad=20, va="center")
        fig.colorbar(sc, ax=axes[row, :].tolist(), shrink=0.85, label=name)

    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def plot_result_interpolated(u_approx, u_exact, meta, out_path, n_show=4, grid_res=200):
    """Same 3-row comparison as plot_result, but the scattered interior points
    are interpolated (scipy.interpolate.griddata, linear, masked to the disk)
    onto a regular grid first, so each panel renders as a smooth field instead
    of discrete dots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    N, M = meta["N"], meta["M"]
    xy = meta["interior_xy0"]
    radius = meta["radius"]
    theta = np.linspace(0, 2 * np.pi, 200)
    bx, by = radius * np.cos(theta), radius * np.sin(theta)

    gx = np.linspace(-radius, radius, grid_res)
    gy = np.linspace(-radius, radius, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    inside = GX ** 2 + GY ** 2 <= radius ** 2

    u_exact_g = u_exact.reshape(N, M)
    u_approx_g = u_approx.reshape(N, M)
    err_g = u_approx_g - u_exact_g

    def interp(field_1d):
        # linear interpolation only covers the convex hull of the scattered
        # points, which falls short of the disk boundary (interior points are
        # sampled strictly inside r<radius); fill that outer ring with
        # nearest-neighbor extrapolation so the whole disk is covered.
        grid = griddata(xy, field_1d, (GX, GY), method="linear")
        nearest = griddata(xy, field_1d, (GX, GY), method="nearest")
        grid = np.where(np.isnan(grid), nearest, grid)
        grid = np.where(inside, grid, np.nan)
        return grid

    idxs = np.linspace(0, N - 1, min(n_show, N)).astype(int)
    u_vmin, u_vmax = u_exact_g.min(), u_exact_g.max()
    err_absmax = np.abs(err_g).max()

    rows = [(u_exact_g, "u_exact", "RdBu_r", u_vmin, u_vmax),
            (u_approx_g, "u_approx", "RdBu_r", u_vmin, u_vmax),
            (err_g, "u_approx - u_exact", "RdBu_r", -err_absmax, err_absmax)]

    row_labels = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(3, len(idxs), figsize=(3.2 * len(idxs), 9), constrained_layout=True)
    for row, (field, name, cmap, vmin, vmax) in enumerate(rows):
        sc = None
        for col, ti in enumerate(idxs):
            ax = axes[row, col]
            sc = ax.imshow(interp(field[ti]), extent=[-radius, radius, -radius, radius],
                            origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.plot(bx, by, color="k", lw=1)
            ax.set_aspect("equal")
            ax.set_title(f"{name}, t idx={ti}")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=14, fontweight="bold", rotation=0, labelpad=20, va="center")
        fig.colorbar(sc, ax=axes[row, :].tolist(), shrink=0.85, label=name)

    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=str, default="dataset/main/samples/test/sample_00000.npz")
    p.add_argument("--subnet", type=str, default="runs_subnetworks/subnetworks.pt")
    p.add_argument("--exact", action="store_true", help="bypass trained sub-networks, use closed forms")
    p.add_argument("--K", type=int, default=6, help="Neumann series truncation order")
    p.add_argument("--plot", type=str, default=None,
                    help="if set, save a u_exact/u_approx/error comparison figure to this path")
    p.add_argument("--plot_interp", type=str, default=None,
                    help="if set, save a spatially-interpolated (griddata) version of --plot to this path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = SubNetworks(ckpt_path=args.subnet, device=device, use_exact=args.exact)

    u_approx, u_exact, meta = solve(args.sample, nets, args.K)

    err = u_approx - u_exact
    rel_l2 = np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(u_exact ** 2))
    print(f"\nmode = {'EXACT (closed-form sub-formulas)' if args.exact else 'NEURAL (trained sub-networks)'}")
    print(f"relative L2 error (interior mesh): {rel_l2:.4f}")
    print(f"max |u_approx - u_exact| = {np.max(np.abs(err)):.4f}   "
          f"(u_exact range [{u_exact.min():.3f}, {u_exact.max():.3f}])")

    if args.plot:
        plot_result(u_approx, u_exact, meta, args.plot)
    if args.plot_interp:
        plot_result_interpolated(u_approx, u_exact, meta, args.plot_interp)


if __name__ == "__main__":
    main()
