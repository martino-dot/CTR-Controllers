"""Webcam + ArUco pose estimation -> tip pose in the robot base frame."""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------- SE(3) helpers (numpy)
def rot_exp(w):
    return cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))[0]


def rot_log(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rt_to_T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = rot_exp(rvec)
    T[:3, 3] = t
    return T


def vec6_to_T(v):
    return rt_to_T(v[:3], v[3:])


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


@dataclass
class TipMeasurement:
    t: float = 0.0
    ok: bool = False
    p: np.ndarray | None = None          # tip position in robot base frame [m]
    R: np.ndarray | None = None          # tip orientation in robot base frame
    T_cam_robot: np.ndarray | None = None
    reproj: float = 0.0
    frame: np.ndarray | None = None
    K: np.ndarray | None = None
    dist: np.ndarray | None = None


class PoseFilter:
    """Exponential filter on SE(3). alpha = weight of the newest measurement."""

    def __init__(self, alpha):
        self.a, self.T = alpha, None

    def __call__(self, T):
        if self.T is None:
            self.T = T.copy()
            return self.T
        a, Tn = self.a, self.T.copy()
        Tn[:3, 3] = (1 - a) * self.T[:3, 3] + a * T[:3, 3]
        Tn[:3, :3] = self.T[:3, :3] @ rot_exp(a * rot_log(self.T[:3, :3].T @ T[:3, :3]))
        self.T = Tn
        return Tn

    def reset(self):
        self.T = None


# ----------------------------------------------------------------------------- tracker
class ArucoTracker:
    def __init__(self, vc):
        self.vc = vc
        self.cap = cv2.VideoCapture(vc.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, vc.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vc.height)
        self.cap.set(cv2.CAP_PROP_FPS, vc.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {vc.cam_index}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if os.path.exists(vc.calib_file):
            z = np.load(vc.calib_file)
            self.K, self.dist = z["K"].astype(float), z["dist"].astype(float).ravel()
            self.calibrated = True
            self.reproj_thresh = vc.max_reproj_px
        else:
            f = 0.5 * w / np.tan(np.radians(65) / 2)     # rough 65 deg HFOV guess
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
            self.dist = np.zeros(5)
            self.calibrated = False
            # the approximate intrinsics above are wrong enough that even a perfectly-tracked
            # marker can show several pixels of reprojection error - don't let the strict
            # calibrated-camera threshold silently reject every real detection.
            self.reproj_thresh = max(vc.max_reproj_px, 15.0)
            print(f"[vision] WARNING: {vc.calib_file} not found - using approximate intrinsics "
                  f"(reprojection-error rejection loosened to {self.reproj_thresh:.0f}px accordingly). "
                  "Run calibrate_camera.py for accurate poses.")
        self.T_robot_bm = vec6_to_T(np.array(vc.base_marker_in_robot, float))
        self.T_tip_tm = vec6_to_T(np.array(vc.tip_marker_in_tip, float))
        d = getattr(cv2.aruco, vc.aruco_dict)
        dic = cv2.aruco.getPredefinedDictionary(d)
        try:
            prm = cv2.aruco.DetectorParameters()
        except AttributeError:
            prm = cv2.aruco.DetectorParameters_create()
        prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            det = cv2.aruco.ArucoDetector(dic, prm)
            self._detect = det.detectMarkers
        else:
            self._detect = lambda g: cv2.aruco.detectMarkers(g, dic, parameters=prm)
        self.f_base, self.f_tip = PoseFilter(vc.base_alpha), PoseFilter(vc.tip_alpha)
        self._prevR = {}
        self._last_p = None
        self._jump_rejects = 0
        self._diag_state, self._diag_t = None, 0.0

    def _diag(self, state: str, msg: str, period: float = 2.0):
        """Print msg only when `state` changes, and at most once every `period` s while unchanged -
        so a genuine tracking failure is never silent, but the console doesn't get spammed."""
        now = time.time()
        if state != self._diag_state or now - self._diag_t > period:
            print(f"[vision] {msg}")
            self._diag_state, self._diag_t = state, now

    def _pnp(self, corners, size, key):
        obj = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32) * size / 2
        pts = corners.reshape(4, 2).astype(np.float32)
        n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, pts, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if n == 0:
            return None
        errs = np.asarray(errs).ravel()
        best = int(np.argmin(errs))
        prevR = self._prevR.get(key)
        if n > 1 and prevR is not None and errs.max() < 2.0 * max(errs.min(), 0.3):
            # planar-pose ambiguity: pick the solution consistent with the previous frame
            dist = [np.linalg.norm(rot_log(prevR.T @ rot_exp(r))) for r in rvs]
            best = int(np.argmin(dist))
        T = rt_to_T(rvs[best].ravel(), tvs[best].ravel())
        self._prevR[key] = T[:3, :3]
        return T, float(errs[best])

    def read(self) -> TipMeasurement:
        ok, frame = self.cap.read()
        m = TipMeasurement(t=time.time(), frame=frame, K=self.K, dist=self.dist)
        if not ok:
            self._diag("no_frame", "camera read failed (cable/USB bandwidth/driver issue?)")
            return m
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        found = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, i in zip(corners, ids.ravel()):
                if i == self.vc.base_id:
                    found["base"] = self._pnp(c, self.vc.base_size, "base")
                elif i == self.vc.tip_id:
                    found["tip"] = self._pnp(c, self.vc.tip_size, "tip")
        if found.get("base"):
            self.f_base(found["base"][0])
        if self.f_base.T is None:
            seen = sorted(int(i) for i in ids.ravel()) if ids is not None else []
            if not seen:
                self._diag("none", "no markers detected at all - check lighting/focus, or run "
                          "debug_aruco.py to see what the camera actually sees")
            else:
                self._diag("no_base", f"base marker (id={self.vc.base_id}) not seen yet; ids in frame: {seen}")
            return m
        T_cam_robot = self.f_base.T @ inv_T(self.T_robot_bm)      # base marker is static -> filtered / remembered
        m.T_cam_robot = T_cam_robot
        if not found.get("tip"):
            self.f_tip.reset()
            self._diag("no_tip", f"base OK but tip marker (id={self.vc.tip_id}) not visible right now")
            return m
        T_tm, err = found["tip"]
        if err > self.reproj_thresh:
            self._diag("bad_reproj", f"tip detected but rejected: reprojection error {err:.1f}px "
                      f"> threshold {self.reproj_thresh:.0f}px (blurry/angled/too-small marker, "
                      f"or camera needs calibration)")
            return m
        T_tm = self.f_tip(T_tm)
        T = inv_T(T_cam_robot) @ T_tm @ inv_T(self.T_tip_tm)
        if self._last_p is not None and np.linalg.norm(T[:3, 3] - self._last_p) > self.vc.max_jump_m \
                and self._jump_rejects < 3:
            self._jump_rejects += 1                                 # single-frame glitch (ambiguity flip)
            self._diag("jump", f"tip jumped > {self.vc.max_jump_m * 1e3:.0f}mm in one frame, rejecting "
                      "(pose-ambiguity flip or a bad detection)", period=5.0)
            return m
        self._jump_rejects = 0
        self._last_p = T[:3, 3].copy()
        m.ok, m.p, m.R, m.reproj = True, T[:3, 3].copy(), T[:3, :3].copy(), err
        self._diag("ok", f"tracking OK (reprojection {err:.1f}px)", period=10.0)
        return m

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------------------- drawing
def _project(pts, T_cam_robot, K, dist):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), rot_log(T_cam_robot[:3, :3]),
                              T_cam_robot[:3, 3], K, dist)
    return uv.reshape(-1, 2)


def draw_scene(frame, meas: TipMeasurement, backbones=(), poses=()):
    """backbones: [(pts(N,3), BGR)], poses: [(p, R, BGR or None, length)] (None = RGB axes)."""
    if frame is None or meas.T_cam_robot is None:
        return
    Tcr, K, dist = meas.T_cam_robot, meas.K, meas.dist
    for pts, col in backbones:
        uv = _project(pts, Tcr, K, dist)
        cv2.polylines(frame, [uv.astype(np.int32).reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
    axis_cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for p, R, col, length in poses:
        pts = np.vstack([p, p + R[:, 0] * length, p + R[:, 1] * length, p + R[:, 2] * length])
        uv = _project(pts, Tcr, K, dist)
        if not np.all(np.abs(uv) < 1e5):
            continue
        o = tuple(int(v) for v in uv[0])
        for k in range(3):
            cv2.line(frame, o, tuple(int(v) for v in uv[k + 1]), col or axis_cols[k], 2, cv2.LINE_AA)


def draw_text(frame, lines, org=(10, 22), color=(255, 255, 255)):
    if frame is None:
        return
    for i, s in enumerate(lines):
        y = org[1] + 20 * i
        cv2.putText(frame, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)