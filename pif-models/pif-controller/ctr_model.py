"""Differentiable concentric-tube-robot kinematics + physics-informed hybrid model.

Physics core (CTRKinematics)
    Torsionally-rigid Kirchhoff-rod model (Webster & Jones 2010 style):
    the backbone is split into sections where the set of overlapping tubes is constant.
    In each section the equilibrium curvature is the stiffness-weighted average of the
    pre-curvatures of the *curved* tubes, rotated by each tube's angle:
        u_xy = sum_i E_i I_i k_i [cos a_i, sin a_i]  /  sum_i E_i I_i
    Each section is an exact constant-curvature arc, composed with SE(3) closed forms.
    Everything is written in torch => autograd gives the exact Jacobian dTip/dq, and the
    physical parameters (curvatures, stiffness ratios, transmission offsets, marker/base
    frame offsets) are LEARNABLE, with Bayesian priors.

Hybrid model (HybridCTRModel)
    physics + small bounded residual MLP that absorbs what the rod model ignores
    (torsional wind-up, friction/backlash -> uses last-motion direction as input, tube clearance...).

Physics-informed loss (pinn_loss)
    data (Huber on tip position + rotation)  + pairwise increment loss (cancels constant
    frame/marker bias) + residual-magnitude prior + residual smoothness (Lipschitz) prior
    + parameter priors around the nominal physical values.
    Outputs are valid SE(3) by construction (rotation via exponential map).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- SO(3)/SE(3) helpers
def skew(v: torch.Tensor) -> torch.Tensor:
    """(B,3) -> (B,3,3)"""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([
        torch.stack([z, -v[:, 2], v[:, 1]], -1),
        torch.stack([v[:, 2], z, -v[:, 0]], -1),
        torch.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def _ab(x2: torch.Tensor):
    """A = sin(x)/x, B = (1-cos x)/x^2 with x^2 given; NaN-safe values and gradients at 0."""
    small = x2 < 1e-6
    x2s = torch.where(small, torch.ones_like(x2), x2)
    xs = torch.sqrt(x2s)
    A = torch.where(small, 1 - x2 / 6 + x2 * x2 / 120, torch.sin(xs) / xs)
    B = torch.where(small, 0.5 - x2 / 24 + x2 * x2 / 720, (1 - torch.cos(xs)) / x2s)
    return A, B


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    A, B = _ab((w * w).sum(-1))
    K = skew(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + A[:, None, None] * K + B[:, None, None] * (K @ K)


def arc(kx, ky, ell):
    """Constant-curvature arc with curvature vector (kx, ky, 0) [1/m] and length ell [m].
    Returns rotation (B,3,3) and displacement (B,3) expressed in the arc's start frame."""
    x2 = (kx * kx + ky * ky) * ell * ell
    A, B = _ab(x2)
    z = torch.zeros_like(kx)
    K = skew(torch.stack([kx, ky, z], -1))
    I = torch.eye(3, dtype=kx.dtype, device=kx.device).expand_as(K)
    R = I + (ell * A)[:, None, None] * K + (ell * ell * B)[:, None, None] * (K @ K)
    d = torch.stack([ell * ell * B * ky, -ell * ell * B * kx, ell * A], -1)
    return R, d


def mv(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------------- physics core
class CTRKinematics(nn.Module):
    PRIOR_SIGMA = dict(dk=0.15, dei=0.5, alpha_off=0.1, beta_off_mm=5.0,
                       w_base=0.05, t_base_mm=5.0, w_tip=0.05, t_tip_mm=5.0)

    def __init__(self, rc):
        super().__init__()
        f = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer("L", f(rc.total_length))
        self.register_buffer("Ls", f(rc.straight_length))
        self.register_buffer("k0", f(rc.precurvature))
        self.register_buffer("ei0", f(rc.rel_stiffness))
        # learnable, dimensionless / O(1)-scaled corrections around nominal values
        self.dk = nn.Parameter(torch.zeros(3))           # log-multiplicative pre-curvature error
        self.dei = nn.Parameter(torch.zeros(3))          # log-multiplicative stiffness error
        self.alpha_off = nn.Parameter(torch.zeros(3))    # rad, rotation zero offsets
        self.beta_off_mm = nn.Parameter(torch.zeros(3))  # mm, translation zero offsets
        self.w_base = nn.Parameter(torch.zeros(3))       # rad, base-marker/robot frame rotation error
        self.t_base_mm = nn.Parameter(torch.zeros(3))    # mm
        self.w_tip = nn.Parameter(torch.zeros(3))        # rad, tip-marker mounting rotation error
        self.t_tip_mm = nn.Parameter(torch.zeros(3))     # mm

    def prior_loss(self):
        return sum(((getattr(self, n) / s) ** 2).sum() for n, s in self.PRIOR_SIGMA.items())

    def _sections(self, q):
        B = q.shape[0]
        alpha = q[:, :3] + self.alpha_off
        beta = q[:, 3:] + 1e-3 * self.beta_off_mm
        tip = beta + self.L                     # distal end of each tube along s
        cs = beta + self.Ls                     # start of each tube's curved part
        bps = torch.cat([torch.zeros(B, 1, dtype=q.dtype, device=q.device), tip, cs], 1).clamp(min=0.0)
        bps, _ = torch.sort(bps, 1)             # 7 breakpoints -> 6 sections
        ell = bps[:, 1:] - bps[:, :-1]
        m = (0.5 * (bps[:, 1:] + bps[:, :-1])).unsqueeze(2)      # (B,6,1) section midpoints
        present = ((m >= beta.unsqueeze(1)) & (m <= tip.unsqueeze(1))).to(q.dtype)   # (B,6,3)
        curved = present * (m >= cs.unsqueeze(1)).to(q.dtype)
        ei = self.ei0 * torch.exp(self.dei)
        k = self.k0 * torch.exp(self.dk)
        den = (present * ei).sum(2).clamp(min=1e-9)
        c = curved * ei * k
        kx = (c * torch.cos(alpha).unsqueeze(1)).sum(2) / den
        ky = (c * torch.sin(alpha).unsqueeze(1)).sum(2) / den
        return kx, ky, ell

    def integrate(self, q, n_sub: int = 0):
        """Returns tip position (B,3), rotation (B,3,3) and optionally backbone points (B,N,3)."""
        kx, ky, ell = self._sections(q)
        B = q.shape[0]
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(B, 3, 3)
        p = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
        pts = [p]
        for j in range(ell.shape[1]):
            if n_sub:
                for s in range(1, n_sub):
                    _, d = arc(kx[:, j], ky[:, j], ell[:, j] * s / n_sub)
                    pts.append(p + mv(R, d))
            Rs, d = arc(kx[:, j], ky[:, j], ell[:, j])
            p = p + mv(R, d)
            R = R @ Rs
            pts.append(p)
        # tip-marker mounting error (tip frame), then base-frame error (global)
        p = p + mv(R, (1e-3 * self.t_tip_mm).expand(B, 3))
        R = R @ so3_exp(self.w_tip.expand(B, 3))
        Ro = so3_exp(self.w_base.expand(B, 3))
        to = 1e-3 * self.t_base_mm
        p = mv(Ro, p) + to
        R = Ro @ R
        if n_sub:
            pts = torch.stack(pts, 1)
            pts = (Ro.unsqueeze(1) @ pts.unsqueeze(-1)).squeeze(-1) + to
            return p, R, pts
        return p, R, None

    def forward(self, q):
        p, R, _ = self.integrate(q)
        return p, R


# ----------------------------------------------------------------------------- hybrid model
class HybridCTRModel(nn.Module):
    P_SCALE = 5e-3      # 1 unit of raw net output = 5 mm
    R_SCALE = 0.1       # 1 unit = 0.1 rad

    def __init__(self, rc, hidden: int = 64):
        super().__init__()
        self.phys = CTRKinematics(rc)
        self.net = nn.Sequential(nn.Linear(15, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 6))
        nn.init.zeros_(self.net[-1].weight)     # start as the pure physics model
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(q, d):
        # periodic in the rotation angles; d = sign of last motion of each joint (backlash / friction state)
        return torch.cat([torch.sin(q[:, :3]), torch.cos(q[:, :3]), q[:, 3:] / 0.1, d], 1)

    def forward(self, q, d=None, return_res: bool = False):
        if d is None:
            d = torch.zeros_like(q)
        p0, R0 = self.phys(q)
        r = self.net(self.features(q, d))
        p = p0 + r[:, :3] * self.P_SCALE
        R = R0 @ so3_exp(r[:, 3:] * self.R_SCALE)
        return (p, R, r) if return_res else (p, R)

    @torch.no_grad()
    def backbone(self, q, d=None, n_sub: int = 8):
        """Backbone polyline (N,3) for one configuration q (1,6); tip corrected by the residual."""
        _, _, pts = self.phys.integrate(q, n_sub)
        p_h, _ = self.forward(q, d)
        corr = p_h - pts[:, -1]
        frac = torch.linspace(0, 1, pts.shape[1], dtype=q.dtype).view(1, -1, 1)
        return (pts + frac * corr.unsqueeze(1))[0]


# ----------------------------------------------------------------------------- physics-informed loss
def pinn_loss(model: HybridCTRModel, q, d, p_m, R_m, w: dict):
    p, R, r = model(q, d, return_res=True)
    l_pos = F.huber_loss(p * 1e3, p_m * 1e3, delta=3.0)                  # mm
    l_rot = F.huber_loss(R * 30.0, R_m * 30.0, delta=3.0)                # 1 unit ~ 0.03 rad
    perm = torch.randperm(q.shape[0])
    l_pair = F.huber_loss((p - p[perm]) * 1e3, (p_m - p_m[perm]) * 1e3, delta=3.0)
    l_res = (r ** 2).mean()                                              # model mismatch should be small
    f = model.features(q, d)
    f2 = f.clone()
    f2[:, :9] = f2[:, :9] + 0.05 * torch.randn_like(f2[:, :9])
    l_smooth = ((model.net(f2) - r) / 0.05).pow(2).mean()               # residual must be smooth in q
    l_prior = model.phys.prior_loss()                                    # stay near physical parameters
    loss = (w["pos"] * l_pos + w["rot"] * l_rot + w["pair"] * l_pair
            + w["res"] * l_res + w["smooth"] * l_smooth + w["prior"] * l_prior)
    return loss, dict(pos=l_pos.item(), rot=l_rot.item(), pair=l_pair.item(),
                      res=l_res.item(), smooth=l_smooth.item(), prior=l_prior.item())
