"""Train the two sub-networks that the paper's construction is actually
fit by gradient descent on (everything else in the paper is a deterministic
formula built on top of these two -- see levi_operator.py):

  - N_sigma1,theta1  (Sec. 2.3.1, p.11, Eq. 2.27-2.28):
        approximates matrix inversion  (a_ij) -> (a^ij)
        for symmetric a_ij with c0 <= eig(a_ij) <= c1  (d=2 here).

  - N_sigma1,theta2  (Sec. 2.3.2, p.12, Eq. 2.29-2.32):
        approximates the regularized heat kernel building block
            w(y,s) = s^{-d/2} exp(-|y|^2/s),   (y,s) in O_eps
        (we only need it away from s=0, matching the paper's own claim,
        which is stated on O_eps = box x (eps,T), not on [0,eps]).

Both are plain ReLU MLPs (sigma1 in the paper's notation, Sec. 1.4, p.5),
trained by ordinary supervised regression against the exact formulas -- the
paper's role for them is purely "these elementary maps are representable by
a small ReLU network"; the PDE itself never appears in this file.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


class ReLUMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_hidden=4):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# N_sigma1,theta1 : matrix inversion, Sec. 2.3.1
# ---------------------------------------------------------------------------

def exact_inv2x2(a11, a12, a22):
    """Closed-form inverse of [[a11,a12],[a12,a22]] (ground truth for training)."""
    det = a11 * a22 - a12 ** 2
    return a22 / det, -a12 / det, a11 / det


def sample_spd_matrices(rng, n, c0, c1):
    """Random symmetric matrices with eigenvalues in [c0,c1] (Eq. 1.1/1.2, p.4),
    same eigen/rotation construction used in generate_samples.py."""
    lam1 = rng.uniform(c0, c1, n)
    lam2 = rng.uniform(c0, c1, n)
    th = rng.uniform(0.0, np.pi, n)
    ct, st = np.cos(th), np.sin(th)
    a11 = lam1 * ct ** 2 + lam2 * st ** 2
    a22 = lam1 * st ** 2 + lam2 * ct ** 2
    a12 = (lam1 - lam2) * st * ct
    return a11, a12, a22


def train_inv_net(c0, c1, device, epochs=6000, batch=4096, hidden=96, seed=0):
    net = ReLUMLP(3, hidden, 3, n_hidden=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        a11, a12, a22 = sample_spd_matrices(rng, batch, c0, c1)
        x = torch.as_tensor(np.stack([a11, a12, a22], axis=1), dtype=torch.float32, device=device)
        i11, i12, i22 = exact_inv2x2(a11, a12, a22)
        y = torch.as_tensor(np.stack([i11, i12, i22], axis=1), dtype=torch.float32, device=device)

        pred = net(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 1000 == 0 or ep == 1:
            print(f"  [N_inv]  epoch {ep:5d}/{epochs}  mse={loss.item():.3e}")

    # held-out sup-error check (Eq. 2.27 asks for W^{1,infty} error < eps; we
    # report sup-norm of the value error as a practical proxy)
    a11, a12, a22 = sample_spd_matrices(rng, 20000, c0, c1)
    x = torch.as_tensor(np.stack([a11, a12, a22], axis=1), dtype=torch.float32, device=device)
    i11, i12, i22 = exact_inv2x2(a11, a12, a22)
    y = np.stack([i11, i12, i22], axis=1)
    with torch.no_grad():
        pred = net(x).cpu().numpy()
    sup_err = np.max(np.abs(pred - y))
    print(f"  [N_inv]  held-out sup error = {sup_err:.4e}")
    return net


# ---------------------------------------------------------------------------
# N_sigma1,theta2 : regularized heat kernel, Sec. 2.3.2
# ---------------------------------------------------------------------------

def exact_w(y1, y2, s):
    return np.exp(-(y1 ** 2 + y2 ** 2) / s) / s


def exact_log_w(y1, y2, s):
    """log(w(y,s)) computed directly (avoids exp-underflow-then-log(0)=-inf)."""
    return -(y1 ** 2 + y2 ** 2) / s - np.log(s)


def train_heat_net(y_box, eps, T, device, epochs=6000, batch=4096, hidden=128, seed=1):
    """Train on O_eps = (-y_box,y_box)^2 x (eps,T) -- the paper's own claim
    (Eq. 2.31) is only made on this set, not on [0,eps] where w is singular.
    We oversample small s (log-uniform) since that is where w varies fastest.
    """
    net = ReLUMLP(3, hidden, 1, n_hidden=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)

    log_eps, log_T = np.log(eps), np.log(T)
    for ep in range(1, epochs + 1):
        y1 = rng.uniform(-y_box, y_box, batch)
        y2 = rng.uniform(-y_box, y_box, batch)
        s = np.exp(rng.uniform(log_eps, log_T, batch))  # log-uniform in (eps,T)

        x = torch.as_tensor(np.stack([y1, y2, s], axis=1), dtype=torch.float32, device=device)
        # the network is trained on log(w) for numerical stability (w spans
        # several orders of magnitude on O_eps, and can underflow to exactly
        # 0 in float64 for large |y|^2/s -- compute log(w) directly instead
        # of log(exp(...)) to avoid log(0)=-inf); w = exp(net output).
        logw = torch.as_tensor(exact_log_w(y1, y2, s), dtype=torch.float32, device=device)
        pred = net(x).squeeze(-1)
        loss = torch.mean((pred - logw) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if ep % 1000 == 0 or ep == 1:
            print(f"  [N_heat] epoch {ep:5d}/{epochs}  mse(log w)={loss.item():.3e}")

    y1 = rng.uniform(-y_box, y_box, 20000)
    y2 = rng.uniform(-y_box, y_box, 20000)
    s = np.exp(rng.uniform(log_eps, log_T, 20000))
    target = exact_w(y1, y2, s)
    x = torch.as_tensor(np.stack([y1, y2, s], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_w = torch.exp(net(x).squeeze(-1)).cpu().numpy()
    rel_err = np.abs(pred_w - target) / (np.abs(target) + 1e-8)
    print(f"  [N_heat] held-out median relative error = {np.median(rel_err):.4e}, "
          f"90th pct = {np.quantile(rel_err, 0.9):.4e}")
    return net


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c0", type=float, default=0.6)
    p.add_argument("--c1", type=float, default=1.8)
    p.add_argument("--y_box", type=float, default=3.0, help="half-width of the y-box in O_eps")
    p.add_argument("--eps", type=float, default=0.02, help="lower cutoff for s (avoids the s=0 singularity)")
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--out", type=str, default="runs_subnetworks")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    print("Training N_sigma1,theta1 (matrix inversion)...")
    inv_net = train_inv_net(args.c0, args.c1, device)

    print("Training N_sigma1,theta2 (regularized heat kernel)...")
    heat_net = train_heat_net(args.y_box, args.eps, args.T, device)

    torch.save({
        "inv_state": inv_net.state_dict(),
        "heat_state": heat_net.state_dict(),
        "c0": args.c0, "c1": args.c1, "y_box": args.y_box, "eps": args.eps, "T": args.T,
    }, os.path.join(args.out, "subnetworks.pt"))
    print(f"Saved -> {os.path.join(args.out, 'subnetworks.pt')}")


if __name__ == "__main__":
    main()
