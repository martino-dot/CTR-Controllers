"""Headless end-to-end test on the simulator (no camera, no motors, no GUI).

  1. physics model vs. 'true' robot BEFORE learning
  2. online learning from noisy, gated 'ArUco' samples -> error should drop
  3. tip-frame joystick: command +z (forward) in the tip frame and check the TRUE tip moves that way
  4. inverse kinematics round trip
Run:  python selftest.py
"""
import time
import numpy as np
import torch

from config import Config
from online_learning import ModelServer, OnlineTrainer, DirectionTracker, SampleGate
from simulation import SimRobot, SimTracker
from controller import TipController


def test_error(sim, server, n=300, seed=1):
    rng = np.random.default_rng(seed)
    ctrl = TipController(cfg.robot, cfg.control, server)
    ep, ea = [], []
    for _ in range(n):
        q = ctrl.js.project(np.concatenate([rng.uniform(-np.pi, np.pi, 3),
                                            rng.uniform(cfg.robot.beta_min, cfg.robot.beta_max)]))
        d = rng.choice([-1.0, 1.0], 6)
        p_t, R_t = sim.true_pose(q, d)
        p, R = server.pose(q, d)
        ep.append(np.linalg.norm(p - p_t) * 1e3)
        ea.append(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_t) - 1) / 2, -1, 1))))
    return np.mean(ep), np.percentile(ep, 95), np.mean(ea)


cfg = Config()
torch.manual_seed(0)
sim = SimRobot(cfg.robot, seed=0)
tracker = SimTracker(sim)
server = ModelServer(cfg.robot, cfg.learn.hidden)
trainer = OnlineTrainer(cfg.robot, cfg.learn, server)
ctrl = TipController(cfg.robot, cfg.control, server)

# ---- 1. before learning ------------------------------------------------------------
m, p95, a = test_error(sim, server)
print(f"[1] before learning : tip error mean {m:6.2f} mm  p95 {p95:6.2f} mm  orient {a:5.2f} deg")

# ---- 2. online learning from random exploration (fast-forwarded simulation) ---------------
rng = np.random.default_rng(3)
dt, t_sim = 1 / 30, 0.0
dirs, gate = DirectionTracker(), SampleGate(cfg.learn)
n_added, t0 = 0, time.time()
arrived = None
while n_added < 400 and t_sim < 3600:
    if np.linalg.norm(sim.q_target - sim.q) < 1e-6:
        arrived = t_sim if arrived is None else arrived
    else:
        arrived = None
    if arrived is not None and t_sim - arrived > 0.6:  # stand still long enough to give valid samples
        arrived = None
        q_t = ctrl.js.project(np.concatenate([rng.uniform(-np.pi, np.pi, 3),
                                              rng.uniform(cfg.robot.beta_min, cfg.robot.beta_max)]))
        sim.command(q_t)
    sim.update(dt)
    t_sim += dt
    meas = tracker.read()
    meas.t = t_sim                                   # simulated clock
    q_act = sim.read_q()
    d = dirs.update(q_act)
    if gate.accept(t_sim, q_act, meas):
        trainer.add_sample(q_act, d, meas.p, meas.R)
        n_added += 1
        if len(trainer.buffer) >= cfg.learn.min_samples and n_added % 5 == 0:
            trainer.train_steps(25)                  # what the background thread does, but synchronous
            if n_added % 100 == 0:
                print(f"    {n_added:4d} samples, prequential err {trainer.stats['pos_mm']:.2f} mm, "
                      f"loss {trainer.stats['loss']:.2f}")
trainer.train_steps(600)
m2, p952, a2 = test_error(sim, server)
print(f"[2] after learning  : tip error mean {m2:6.2f} mm  p95 {p952:6.2f} mm  orient {a2:5.2f} deg "
      f"({n_added} samples, {time.time() - t0:.0f}s wall)")

# ---- 3. tip-frame joystick: press 'forward' for 3 s -----------------------------------------
sim.q = sim.q_target = np.array(cfg.robot.home, float)
sim.d_true[:] = 0
q_cmd = sim.q.copy()
p0, R0 = sim.true_pose()
cmd = np.array([0, 0, 1.0, 0, 0, 0])
for _ in range(90):
    meas = tracker.read()
    q_cmd, info = ctrl.step(q_cmd, dirs.d, cmd, meas if meas.ok else None, dt)
    sim.command(q_cmd)
    sim.update(dt)
p1, R1 = sim.true_pose()
disp = R0.T @ (p1 - p0)                              # displacement in the ORIGINAL tip frame
exp = cfg.control.lin_speed * 3
print(f"[3] forward 3 s     : displacement in tip frame = {np.round(disp * 1e3, 1)} mm "
      f"(commanded {exp * 1e3:.0f} mm along +z)")

# sideways (tip +x) 2 s
p0, R0 = sim.true_pose()
cmd = np.array([1.0, 0, 0, 0, 0, 0])
for _ in range(60):
    meas = tracker.read()
    q_cmd, info = ctrl.step(q_cmd, dirs.d, cmd, meas if meas.ok else None, dt)
    sim.command(q_cmd)
    sim.update(dt)
p1, _ = sim.true_pose()
print(f"    strafe +x 2 s   : displacement in tip frame = {np.round(R0.T @ (p1 - p0) * 1e3, 1)} mm "
      f"(commanded {cfg.control.lin_speed * 2e3:.0f} mm along +x)")

# ---- 4. IK round trip -----------------------------------------------------------------------
q_goal = ctrl.js.project(np.array([0.5, -0.4, 0.3, -0.08, -0.05, -0.02]))
p_goal, _ = server.pose(q_goal)
q_ik, err = ctrl.ik(p_goal, np.array(cfg.robot.home, float))
print(f"[4] IK              : residual {err * 1e3:.3f} mm, q = {np.round(q_ik, 3)}")
