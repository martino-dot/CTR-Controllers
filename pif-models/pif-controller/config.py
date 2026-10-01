"""Central configuration. EDIT THESE VALUES to match YOUR robot, camera and markers.

Joint vector convention used everywhere:
    q = [alpha1, alpha2, alpha3, beta1, beta2, beta3]
    alpha_i : rotation of tube i about the insertion axis           [rad]
    beta_i  : translation of tube i's base along the insertion axis [m]
              (<= 0: tube base sits behind the fixed robot origin s = 0)
Tubes are numbered innermost (1) -> outermost (3).
Robot base frame: origin at s = 0, +z along the insertion axis (tip direction).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RobotConfig:
    # ---- tube geometry (measure these!) ---------------------------------
    total_length: tuple = (0.45, 0.33, 0.21)      # m, full length of each tube
    straight_length: tuple = (0.30, 0.20, 0.12)   # m, straight proximal part (rest is pre-curved)
    precurvature: tuple = (6.7, 8.3, 10.0)        # 1/m, pre-set curvature of each curved section
    rel_stiffness: tuple = (1.0, 2.5, 6.0)        # relative bending stiffness E*I (only ratios matter)

    # ---- joint limits ----------------------------------------------------
    alpha_limit: float = 4 * np.pi                # +-rad allowed rotation of each tube
    beta_min: tuple = (-0.20, -0.14, -0.08)       # m
    beta_max: tuple = (-0.02, -0.01, 0.0)         # m
    carriage_gap: float = 0.005                   # m, min spacing between translation carriages
    home: tuple = (0, 0, 0, -0.10, -0.06, -0.02)
    max_speed: tuple = (1.0, 1.0, 1.0, 0.015, 0.015, 0.015)   # rad/s, m/s

    # ---- serial motor interface ---------------------------------------------
    # steps_per_unit is only used by the legacy raw-step GenericAsciiRobot. For MarlinRobot,
    # steps/unit lives in the firmware (Configuration.h DEFAULT_AXIS_STEPS_PER_UNIT / M92);
    # the host only ever sends physical units (mm, degrees).
    steps_per_unit: tuple = (509.3, 509.3, 509.3, 1.6e6, 1.6e6, 1.6e6)  # steps/rad, steps/m
    motor_signs: tuple = (1, 1, 1, 1, 1, 1)

    # ---- Marlin (BigTreeTech Octopus etc.) axis mapping ----------------------
    # q = [alpha1, alpha2, alpha3, beta1, beta2, beta3] -> these Marlin axis letters, in order.
    # Default here: X/Y/Z are the three z-axis (translation) carriages, A/B/C are the three
    # rotation stages, A = tube closest to the tip (tube 1 / innermost). EDIT if your wiring
    # pairs the letters with different tubes, or in a different order.
    marlin_axes: tuple = ("A", "B", "C", "X", "Y", "Z")
    marlin_baud: int = 250000            # must match Marlin's Configuration.h BAUDRATE


@dataclass
class VisionConfig:
    cam_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    calib_file: str = "camera_calib.npz"          # npz with K (3x3) and dist (5,) from a checkerboard calibration
    aruco_dict: str = "DICT_ARUCO_ORIGINAL"
    base_id: int = 1                              # fixed marker on the robot base
    tip_id: int = 0                               # marker on the robot tip
    base_size: float = 0.030                      # m, black-square side length
    tip_size: float = 0.010
    # pose of the base marker expressed in the robot base frame:  (rvec[3], t[3]) [rad, m]
    base_marker_in_robot: tuple = (0, 0, 0, 0, 0, 0)
    # pose of the tip marker expressed in the tip frame (z along tube tangent): (rvec[3], t[3])
    tip_marker_in_tip: tuple = (0, 0, 0, 0, 0, 0)
    max_reproj_px: float = 3.0
    tip_alpha: float = 0.7                        # filter weight of the newest tip measurement
    base_alpha: float = 0.2                       # base marker is static -> filter harder
    max_jump_m: float = 0.03                      # reject single-frame tip jumps larger than this
    # the base marker only needs to be seen ONCE (see vision.BaseLock); these control how that
    # one-time lock is established: require this many consecutive sightings within tolerance.
    base_lock_frames: int = 8
    base_lock_tol_m: float = 0.008                 # m (small/distant markers have noisier PnP depth)
    base_lock_tol_rad: float = 0.12                # rad (~7 deg) - initial lock only needs to be roughly
    # right; rotation estimates from a near-frontal marker view are inherently noisy frame-to-frame
    # (classic planar-PnP conditioning), and it keeps refining via base_alpha every time the base
    # reappears afterward. Mounting the base marker larger and/or at a non-frontal angle to the
    # camera reduces this noise and locks faster.


@dataclass
class LearnConfig:
    hidden: int = 64
    lr_phys: float = 3e-3
    lr_net: float = 2e-3
    batch: int = 64
    buffer: int = 5000
    min_samples: int = 30
    steps_per_cycle: int = 5
    cycle_period: float = 0.05                    # s between training bursts in the background thread
    weights: dict = field(default_factory=lambda: dict(
        pos=1.0, rot=1.0, pair=0.5, res=0.02, smooth=0.01, prior=0.02))
    # sample gating (only train on settled, novel, well-tracked samples)
    settle_s: float = 0.3
    settle_tol_alpha: float = 0.003
    settle_tol_beta: float = 3e-4
    novelty: float = 0.02
    max_sample_period: float = 3.0


@dataclass
class ControlConfig:
    control_rate: float = 30.0                    # Hz
    lin_speed: float = 0.010                      # m/s at full stick
    ang_speed: float = 0.25                       # rad/s at full stick
    ang_scale: float = 0.05                       # m/rad weighting of angular task rows
    damping: float = 0.01                         # DLS lambda
    nullspace_gain: float = 0.2                   # 1/s pull towards home in the 1-D null space
    hold_kp_lin: float = 2.0                      # 1/s
    hold_kp_ang: float = 2.0                      # 1/s
    key_pulse_s: float = 0.2                      # keyboard "tap" duration


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)
    control: ControlConfig = field(default_factory=ControlConfig)