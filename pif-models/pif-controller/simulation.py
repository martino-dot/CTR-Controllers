"""Simulated 'real' robot + camera so the whole pipeline can be tested without hardware.

The simulated truth deliberately differs from the nominal physics model:
  * different pre-curvatures / stiffness / zero offsets / frame offsets,
  * tube-tube torsional coupling (not in the rod model),
  * backlash that depends on the last direction of motion,
  * ArUco-like measurement noise and random dropouts.
"""
from __future__ import annotations
import time
import numpy as np
import torch

from ctr_model import CTRKinematics
from robot_interface import RobotInterface
from vision import TipMeasurement, rot_exp


class SimRobot(RobotInterface):
    def __init__(self, rc, seed=0):
        self.rc = rc
        self.rng = np.random.default_rng(seed)
        self.truth = CTRKinematics(rc)
        with torch.no_grad():
            self.truth.dk += torch.tensor([0.07, -0.06, 0.05])
            self.truth.dei += torch.tensor([0.20, -0.15, 0.10])
            self.truth.alpha_off += torch.tensor([0.05, -0.04, 0.03])
            self.truth.beta_off_mm += torch.tensor([2.0, -1.5, 1.0])
            self.truth.t_base_mm += torch.tensor([1.5, -1.0, 0.5])
            self.truth.w_base += torch.tensor([0.01, -0.01, 0.02])
            self.truth.t_tip_mm += torch.tensor([1.0, 0.5, 0.0])
        self.q = np.array(rc.home, float)
        self.q_target = self.q.copy()
        self.d_true = np.zeros(6)
        self.vmax = 2.0 * np.array(rc.max_speed)

    def update(self, dt):
        step = np.clip(self.q_target - self.q, -self.vmax * dt, self.vmax * dt)
        moved = np.abs(step) > 1e-9
        self.d_true[moved] = np.sign(step[moved])
        self.q = self.q + step

    def command(self, q):
        self.q_target = np.array(q, float)

    def read_q(self):
        return self.q.copy()

    def true_pose(self, q=None, d=None):
        q = self.q if q is None else np.asarray(q, float)
        d = self.d_true if d is None else np.asarray(d, float)
        qe = q.copy()
        a = q[:3]
        qe[:3] = a + 0.10 * np.sin(a - np.roll(a, -1)) - 0.02 * d[:3]      # torsional coupling + backlash
        qe[3:] = q[3:] - 0.0003 * d[3:]                                      # linear backlash 0.3 mm
        with torch.no_grad():
            p, R = self.truth(torch.tensor(qe, dtype=torch.float32)[None])
        return p[0].numpy().astype(float), R[0].numpy().astype(float)


class SimTracker:
    """Fake camera: looks at the robot from the side, returns noisy tip poses."""
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    dist = np.zeros(5)

    def __init__(self, robot: SimRobot, noise_p=3e-4, noise_r=0.005, dropout=0.03):
        self.robot, self.noise_p, self.noise_r, self.dropout = robot, noise_p, noise_r, dropout
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])      # robot z -> image right, robot x -> image down
        self.T = np.eye(4)
        self.T[:3, :3] = R
        self.T[:3, 3] = [-0.22, -0.02, 0.5]

    def read(self):
        rng = self.robot.rng
        p, R = self.robot.true_pose()
        p = p + rng.normal(0, self.noise_p, 3)
        R = R @ rot_exp(rng.normal(0, self.noise_r, 3))
        return TipMeasurement(t=time.time(), ok=bool(rng.random() > self.dropout), p=p, R=R,
                              T_cam_robot=self.T, reproj=0.5, frame=np.zeros((480, 640, 3), np.uint8),
                              K=self.K, dist=self.dist)

    def close(self):
        pass
