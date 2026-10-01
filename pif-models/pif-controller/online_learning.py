"""Live model serving + online (background-thread) physics-informed training."""
from __future__ import annotations
import copy
import threading
import time
from collections import deque

import numpy as np
import torch

from ctr_model import HybridCTRModel, pinn_loss

torch.set_num_threads(2)


# ----------------------------------------------------------------------------- replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.q = np.zeros((capacity, 6), np.float32)
        self.d = np.zeros((capacity, 6), np.float32)
        self.p = np.zeros((capacity, 3), np.float32)
        self.R = np.zeros((capacity, 3, 3), np.float32)
        self.n = 0
        self.i = 0
        self.lock = threading.Lock()

    def __len__(self):
        return self.n

    def add(self, q, d, p, R):
        with self.lock:
            self.q[self.i], self.d[self.i], self.p[self.i], self.R[self.i] = q, d, p, R
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, bs, recent_frac=0.5, recent_window=200):
        """Half of each batch from the most recent samples (adapt fast), half uniform (don't forget)."""
        with self.lock:
            if self.n == 0:
                return None
            n_rec = int(bs * recent_frac)
            k = min(self.n, recent_window)
            rec = (self.i - 1 - np.random.randint(0, k, n_rec)) % self.cap
            uni = np.random.randint(0, self.n, bs - n_rec)
            idx = np.concatenate([rec, uni])
            return tuple(torch.from_numpy(a[idx]) for a in (self.q, self.d, self.p, self.R))

    def save(self, path):
        with self.lock:
            n = self.n
            np.savez(path, q=self.q[:n], d=self.d[:n], p=self.p[:n], R=self.R[:n])

    def load(self, path):
        z = np.load(path)
        for a, b, c, e in zip(z["q"], z["d"], z["p"], z["R"]):
            self.add(a, b, c, e)


# ----------------------------------------------------------------------------- model server (inference)
class ModelServer:
    """Holds the copy of the model used by the control loop; the trainer publishes new weights into it."""

    def __init__(self, rc, hidden=64):
        self.model = HybridCTRModel(rc, hidden).eval()
        self.lock = threading.Lock()

    def update(self, state_dict):
        with self.lock:
            self.model.load_state_dict(state_dict)

    @staticmethod
    def _t(x):
        return torch.as_tensor(np.asarray(x, np.float32))[None]

    def pose(self, q, d=None):
        with self.lock, torch.no_grad():
            p, R = self.model(self._t(q), None if d is None else self._t(d))
        return p[0].numpy().astype(float), R[0].numpy().astype(float)

    def backbone(self, q, d=None, n_sub=8):
        with self.lock:
            return self.model.backbone(self._t(q), None if d is None else self._t(d), n_sub).numpy().astype(float)

    def jacobian(self, q, d=None):
        """Body-frame tip twist Jacobian J (6x6): [v_tip; w_tip] (expressed in the tip frame) = J @ dq.
        Also returns the predicted tip position and rotation."""
        qt = self._t(q)[0]
        dt = self._t(np.zeros(6) if d is None else d)
        with self.lock:
            m = self.model
            with torch.no_grad():
                p0, R0 = m(qt[None], dt)
            p0, R0 = p0[0], R0[0]

            def f(x):
                p, R = m(x[None], dt)
                Rr = R0.T @ R[0]
                w = 0.5 * torch.stack([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
                return torch.cat([R0.T @ (p[0] - p0), w])

            J = torch.autograd.functional.jacobian(f, qt, vectorize=True)
        return J.numpy().astype(float), p0.numpy().astype(float), R0.numpy().astype(float)


# ----------------------------------------------------------------------------- online trainer
class OnlineTrainer:
    def __init__(self, rc, lc, server: ModelServer, buffer: ReplayBuffer | None = None):
        self.lc = lc
        self.server = server
        self.model = copy.deepcopy(server.model).train()
        self.opt = torch.optim.Adam([
            dict(params=list(self.model.phys.parameters()), lr=lc.lr_phys),
            dict(params=list(self.model.net.parameters()), lr=lc.lr_net, weight_decay=1e-4)])
        self.buffer = buffer if buffer is not None else ReplayBuffer(lc.buffer)
        self.enabled = True
        self.stats = dict(loss=float("nan"), pos_mm=float("nan"), rot_deg=float("nan"), n=0)
        self._stop = threading.Event()
        self._thread = None

    # -- data -------------------------------------------------------------
    def add_sample(self, q, d, p, R):
        """Called from the control loop. Also tracks the *prequential* error: how wrong the
        live model was on this sample before learning from it."""
        p_pred, R_pred = self.server.pose(q, d)
        e = np.linalg.norm(p_pred - p) * 1e3
        c = np.clip((np.trace(R_pred.T @ R) - 1) / 2, -1, 1)
        a = np.degrees(np.arccos(c))
        s = self.stats
        s["pos_mm"] = e if s["n"] == 0 else 0.9 * s["pos_mm"] + 0.1 * e
        s["rot_deg"] = a if s["n"] == 0 else 0.9 * s["rot_deg"] + 0.1 * a
        s["n"] += 1
        self.buffer.add(q, d, p, R)

    # -- optimisation ------------------------------------------------------
    def train_steps(self, n: int):
        loss = None
        for _ in range(n):
            batch = self.buffer.sample(self.lc.batch)
            if batch is None:
                return None
            loss, _ = pinn_loss(self.model, *batch, self.lc.weights)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
        if loss is not None:
            self.stats["loss"] = loss.item()
            self.server.update(self.model.state_dict())
        return None if loss is None else loss.item()

    def _loop(self):
        while not self._stop.is_set():
            if self.enabled and len(self.buffer) >= self.lc.min_samples:
                self.train_steps(self.lc.steps_per_cycle)
            time.sleep(self.lc.cycle_period)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict(sd)
        self.server.update(sd)


# ----------------------------------------------------------------------------- sample gating
class SampleGate:
    """Only let 'clean' samples into training: robot settled, marker tracking good, sample novel."""

    def __init__(self, lc):
        self.lc = lc
        self.hist = deque()
        self.last_q = None
        self.last_t = -1e9
        self.scale = np.array([1, 1, 1, 10, 10, 10.0])

    def accept(self, t, q, meas):
        lc = self.lc
        self.hist.append((t, q.copy()))
        while self.hist and t - self.hist[0][0] > lc.settle_s + 0.05:
            self.hist.popleft()
        if not meas.ok or t - self.hist[0][0] < 0.9 * lc.settle_s:
            return False
        Q = np.array([h[1] for h in self.hist])
        span = Q.max(0) - Q.min(0)
        if (span[:3] > lc.settle_tol_alpha).any() or (span[3:] > lc.settle_tol_beta).any():
            return False
        if self.last_q is not None:
            if np.linalg.norm((q - self.last_q) * self.scale) < lc.novelty and t - self.last_t < lc.max_sample_period:
                return False
        self.last_q, self.last_t = q.copy(), t
        return True


class DirectionTracker:
    """Remembers the sign of the last motion of every joint (feeds the friction/backlash input)."""

    def __init__(self):
        self.d = np.zeros(6)
        self.prev = None
        self.thr = np.array([5e-4] * 3 + [5e-5] * 3)

    def update(self, q):
        if self.prev is not None:
            dq = q - self.prev
            m = np.abs(dq) > self.thr
            self.d[m] = np.sign(dq[m])
        self.prev = q.copy()
        return self.d.copy()
