"""Tip-frame ("looking from the tip forwards") joystick controller and inverse kinematics.

Joystick command  -> 6-vector in the TIP frame: [vx, vy, vz, wx, wy, wz] in [-1, 1]
   frame convention: +z = forward along the tip tangent, +x = right, +y = down (like a camera)
   (this is defined by how you mounted the tip marker / T_tip_marker in config.py)
   wx = pitch, wy = yaw. Roll (wz) cannot be commanded independently on a CTR and is ignored.
Pipeline: twist -> (camera correction of the tip frame) -> damped-least-squares with the model
Jacobian (autograd through the physics-informed model) -> joint step -> limits/speed clamp.
"""
from __future__ import annotations
import time
import numpy as np

from online_learning import ModelServer


def vee_np(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot_err_np(Ra, Rb):
    """Small-rotation vector taking frame a to frame b, expressed in frame a."""
    M = Ra.T @ Rb
    return 0.5 * vee_np(M - M.T)


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


class JointSpace:
    def __init__(self, rc):
        self.rc = rc
        self.L = np.array(rc.total_length)
        self.bmin, self.bmax = np.array(rc.beta_min), np.array(rc.beta_max)
        self.vmax = np.array(rc.max_speed)
        self.home = np.array(rc.home, float)
        self.scale = np.array([1, 1, 1, 0.1, 0.1, 0.1])      # joint scaling so alpha[rad] ~ beta[0.1 m]

    def project(self, q):
        """Clip to limits and enforce mechanically valid tube nesting / carriage spacing."""
        rc, L, g = self.rc, self.L, self.rc.carriage_gap
        q = np.array(q, float)
        q[:3] = np.clip(q[:3], -rc.alpha_limit, rc.alpha_limit)
        b = q[3:]
        for _ in range(2):
            b = np.clip(b, self.bmin, self.bmax)
            b[1] = np.clip(b[1], b[0] + g, b[0] + L[0] - L[1])      # carriage order and tip order: tube1 longest
            b[2] = np.clip(b[2], b[1] + g, b[1] + L[1] - L[2])
        q[3:] = np.clip(b, self.bmin, self.bmax)
        return q

    def approach(self, q, target, dt):
        """Move q toward target with per-joint speed limits (straight line in joint space)."""
        dq = target - q
        r = np.max(np.abs(dq) / (self.vmax * dt))
        if r > 1:
            dq = dq / r
        return self.project(q + dq)

    def limit_step(self, dq, dt):
        r = np.max(np.abs(dq) / (self.vmax * dt))
        return dq / r if r > 1 else dq


def dls(Jz, e, lam):
    return Jz.T @ np.linalg.solve(Jz @ Jz.T + lam ** 2 * np.eye(Jz.shape[0]), e)


class TipController:
    def __init__(self, rc, cc, server: ModelServer):
        self.rc, self.cc, self.server = rc, cc, server
        self.js = JointSpace(rc)
        self.hold_target = None          # (p, R) in robot base frame
        self.rows = [0, 1, 2, 3, 4]      # position + tangent direction
        self.rw = np.array([1, 1, 1, cc.ang_scale, cc.ang_scale])

    # ---- hold-position servo (uses the camera measurement as feedback) -------
    def set_hold(self, meas):
        self.hold_target = (meas.p.copy(), meas.R.copy()) if meas is not None and meas.ok else None

    def _hold_twist(self, meas):
        pt, Rt = self.hold_target
        e_p = meas.R.T @ (pt - meas.p)
        e_w = rot_err_np(meas.R, Rt)
        cc = self.cc
        v = np.zeros(6)
        v[:3] = cc.hold_kp_lin * e_p
        v[3:5] = cc.hold_kp_ang * e_w[:2]
        n = np.linalg.norm(v[:3])
        if n > cc.lin_speed:
            v[:3] *= cc.lin_speed / n
        n = np.linalg.norm(v[3:5])
        if n > cc.ang_speed:
            v[3:5] *= cc.ang_speed / n
        return v

    # ---- one control step ----------------------------------------------------
    def step(self, q, d, cmd, meas, dt):
        """q: current joint command, d: direction memory, cmd: joystick 6-vector in [-1,1],
        meas: TipMeasurement or None. Returns (q_new, info)."""
        cc, js = self.cc, self.js
        cmd = np.asarray(cmd, float)
        J, p_pred, R_pred = self.server.jacobian(q, d)
        v = cmd * np.array([cc.lin_speed] * 3 + [cc.ang_speed] * 3)
        use_meas = meas is not None and meas.ok
        R_err = np.eye(3)
        if use_meas:
            Re = R_pred.T @ meas.R            # predicted tip frame -> measured tip frame
            if rot_angle(Re) < 0.6:           # ignore obviously bad measurements
                R_err = Re
        holding = self.hold_target is not None and use_meas and np.linalg.norm(cmd[:5]) < 1e-6
        if holding:
            v = self._hold_twist(meas)
        # the user sees/commands in the *measured* tip frame; convert to the model's tip frame
        v_model = np.concatenate([R_err @ v[:3], R_err @ v[3:]])
        t = np.concatenate([v_model[:3], v_model[3:5] * cc.ang_scale]) * dt
        Jz = (J[:5] * self.rw[:, None]) @ np.diag(js.scale)
        dz = dls(Jz, t, cc.damping)
        # null-space posture term (1-D redundancy): drift towards home without moving the tip
        Jp = Jz.T @ np.linalg.inv(Jz @ Jz.T + cc.damping ** 2 * np.eye(5))
        N = np.eye(6) - Jp @ Jz
        dz += N @ (cc.nullspace_gain * dt * (js.home - q) / js.scale)
        dq = js.limit_step(dz * js.scale, dt)
        q_new = js.project(q + dq)
        sv = np.linalg.svd(Jz, compute_uv=False)
        return q_new, dict(holding=holding, sv_min=float(sv[-1]), pred_tip=p_pred)

    # ---- inverse kinematics (predict the motor positions for a desired tip pose) ------
    def ik(self, target_p, q0, d=None, target_R=None, iters=120, tol=3e-4):
        js, cc = self.js, self.cc
        q = np.array(q0, float)
        for _ in range(iters):
            J, p, R = self.server.jacobian(q, d)
            e = R.T @ (target_p - p)
            rows, w = [0, 1, 2], np.ones(3)
            if target_R is not None:
                e = np.concatenate([e, rot_err_np(R, target_R)[:2] * cc.ang_scale])
                rows, w = self.rows, self.rw
            if np.linalg.norm(e[:3]) < tol and (target_R is None or np.linalg.norm(e[3:]) < tol):
                break
            Jz = (J[rows] * w[:, None]) @ np.diag(js.scale)
            dz = dls(Jz, e, 0.02)
            m = np.max(np.abs(dz))
            if m > 0.3:
                dz *= 0.3 / m
            q = js.project(q + dz * js.scale)
        J, p, R = self.server.jacobian(q, d)
        return q, float(np.linalg.norm(target_p - p))


# ----------------------------------------------------------------------------- input devices
class KeyboardTeleop:
    """Keyboard 'virtual joystick' (tip frame). Each key press is a short velocity pulse; holding repeats it.
         W/S forward/back (tip z)   A/D left/right (tip x)   R/F up/down (tip -y/+y)
         I/K pitch up/down          J/L yaw left/right"""
    MAP = {ord('w'): (2, +1), ord('s'): (2, -1), ord('d'): (0, +1), ord('a'): (0, -1),
           ord('f'): (1, +1), ord('r'): (1, -1), ord('i'): (3, +1), ord('k'): (3, -1),
           ord('l'): (4, +1), ord('j'): (4, -1)}

    def __init__(self, pulse_s):
        self.pulse = pulse_s
        self.until = {}

    def poll(self, key, now):
        if key in self.MAP:
            self.until[key] = now + self.pulse
        cmd = np.zeros(6)
        for k, t_end in list(self.until.items()):
            if now < t_end:
                ax, s = self.MAP[k]
                cmd[ax] += s
            else:
                del self.until[k]
        return np.clip(cmd, -1, 1)


class GamepadTeleop:
    """pygame gamepad (axis indices differ between pads - edit AXES).
       left stick: tip x/y   right stick: y = forward/back, x = yaw   triggers: pitch"""
    AXES = dict(lx=0, ly=1, rx=3, ry=4, lt=2, rt=5)

    def __init__(self, deadzone=0.15):
        import pygame
        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.pg, self.dz = pygame, deadzone

    def poll(self, key=None, now=None):
        self.pg.event.pump()

        def ax(name):
            try:
                v = self.js.get_axis(self.AXES[name])
            except Exception:
                return 0.0
            return 0.0 if abs(v) < self.dz else v
        cmd = np.zeros(6)
        cmd[0], cmd[1], cmd[2], cmd[4] = ax('lx'), ax('ly'), -ax('ry'), ax('rx')
        cmd[3] = (ax('rt') - ax('lt')) / 2
        return np.clip(cmd, -1, 1)
