"""Physics-informed learning controller for a 6-motor concentric tube robot with ArUco feedback.

  python main.py teleop  --sim                          # everything simulated (no camera / motors)
  python main.py teleop  --serial /dev/ttyACM0          # real robot (Marlin/Octopus) + webcam
  python main.py teleop  --serial /dev/ttyACM0 --zero-home  # ...and first re-zero at robot.home
  python main.py teleop  --dry-run                      # webcam + virtual joystick, motors not driven
  python main.py collect --serial COM4 --n 300          # automatic data collection sweep -> ctr_data.npz
  python main.py train   --data ctr_data.npz             # offline (pre)training -> ctr_model.pt
  python main.py predict --q 0 0 0 -100 -60 -20          # forward: motors (deg, mm) -> tip pose
  python main.py predict --target 20 -10 250             # inverse: tip xyz [mm] in base frame -> motors

--serial talks to a Marlin-firmware board (e.g. BigTreeTech Octopus) over G-code; see
robot_interface.MarlinRobot and config.RobotConfig.marlin_axes for the axis mapping and
required firmware setup. Before the very first run (or after any power-cycle/lost steps),
physically position the robot at its home pose and pass --zero-home once to register it.

Live keys (click the video window):
  W/S fwd/back   A/D left/right   R/F up/down   I/K pitch   J/L yaw   (all in the TIP frame)
  M  toggle  TELEOP <-> PREDICT(preview, motors do not move)      ENTER  execute the previewed pose
  H  hold current tip pose (camera-feedback servo)                 T  pause/resume online learning
  Z  home    +/-  speed      P  save model + data                  ESC/X  quit
"""
import argparse
import time

import cv2
import numpy as np

from config import Config
from online_learning import ModelServer, OnlineTrainer, DirectionTracker, SampleGate, ReplayBuffer
from controller import TipController, KeyboardTeleop, GamepadTeleop
from vision import draw_scene, draw_text

MODEL_FILE, DATA_FILE = "ctr_model.pt", "ctr_data.npz"


# ----------------------------------------------------------------------------- setup helpers
def build_io(args, cfg):
    if args.sim:
        from simulation import SimRobot, SimTracker
        robot = SimRobot(cfg.robot)
        return robot, SimTracker(robot)
    from vision import ArucoTracker
    from robot_interface import MarlinRobot, NullRobot
    if args.serial:
        robot = MarlinRobot(cfg.robot, args.serial, args.baud)
        if getattr(args, "zero_home", False):
            robot.zero_at(np.array(cfg.robot.home, float))
            print("[marlin] zeroed at robot.home - make sure you positioned it there first!")
    elif args.dry_run:
        robot = NullRobot(cfg.robot)
    else:
        raise SystemExit("Choose one of --sim, --serial PORT or --dry-run")
    return robot, ArucoTracker(cfg.vision)


def build_learning(cfg, load=True):
    server = ModelServer(cfg.robot, cfg.learn.hidden)
    trainer = OnlineTrainer(cfg.robot, cfg.learn, server)
    if load:
        try:
            trainer.load(MODEL_FILE)
            print(f"[model] loaded {MODEL_FILE}")
        except FileNotFoundError:
            print("[model] no saved model - starting from the nominal physics model")
        try:
            trainer.buffer.load(DATA_FILE)
            print(f"[data] loaded {len(trainer.buffer)} samples from {DATA_FILE}")
        except FileNotFoundError:
            pass
    return server, trainer


def deg_mm(q):
    return np.concatenate([np.degrees(q[:3]), q[3:] * 1e3])


# ----------------------------------------------------------------------------- live teleop / predict
def run_teleop(args, cfg):
    robot, tracker = build_io(args, cfg)
    server, trainer = build_learning(cfg)
    trainer.enabled = robot.learns
    trainer.start()
    ctrl = TipController(cfg.robot, cfg.control, server)
    js = ctrl.js
    keyboard = KeyboardTeleop(cfg.control.key_pulse_s)
    pad = GamepadTeleop() if args.gamepad else None
    dirs, gate = DirectionTracker(), SampleGate(cfg.learn)

    q_cmd = robot.read_q()
    q_virt = q_cmd.copy()
    exec_target = None
    mode, speed, cmd = "TELEOP", 1.0, np.zeros(6)
    period, t_prev = 1.0 / cfg.control.control_rate, time.time()
    win = "CTR control"

    try:
        while True:
            now = time.time()
            dt = float(np.clip(now - t_prev, 1e-3, 0.1))
            t_prev = now

            # 1) sense: camera pose of the tip + motor state
            meas = tracker.read()
            robot.update(dt)
            q_act = robot.read_q()
            d = dirs.update(q_act)

            # 2) learn: settled, well-tracked samples go into the online training buffer
            if robot.learns and trainer.enabled and gate.accept(now, q_act, meas):
                trainer.add_sample(q_act, d, meas.p, meas.R)

            # 3) act: joystick -> tip twist -> model Jacobian -> joint command
            c = pad.poll() if pad else cmd
            info = {}
            if mode == "TELEOP":
                q_cmd, info = ctrl.step(q_cmd, d, c * speed, meas if meas.ok else None, dt)
                q_virt = q_cmd.copy()
            else:                                            # PREDICT: preview only, motors stay put
                q_virt, info = ctrl.step(q_virt, d, c * speed, None, dt)
                if exec_target is not None:
                    q_cmd = js.approach(q_cmd, exec_target, dt)
                    if np.linalg.norm(q_cmd - exec_target) < 1e-6:
                        exec_target = None
            robot.command(q_cmd)

            # 4) display
            frame = meas.frame
            if frame is not None:
                bb = [(server.backbone(q_cmd, d), (0, 255, 0))]
                poses = []
                p_pred, R_pred = server.pose(q_cmd, d)
                poses.append((p_pred, R_pred, (0, 255, 0), 0.02))
                if meas.ok:
                    poses.append((meas.p, meas.R, None, 0.02))
                if mode == "PREDICT":
                    bb.append((server.backbone(q_virt, d), (255, 255, 0)))
                    pv, Rv = server.pose(q_virt, d)
                    poses.append((pv, Rv, (255, 255, 0), 0.02))
                draw_scene(frame, meas, bb, poses)
                s = trainer.stats
                err = f"{s['pos_mm']:.1f}mm/{s['rot_deg']:.1f}deg" if s["n"] else "-"
                lines = [
                    f"{mode}{' [HOLD]' if info.get('holding') else ''}  learn:{'ON' if trainer.enabled else 'off'}  "
                    f"tracking:{'OK' if meas.ok else 'LOST'}  speed x{speed:.2f}",
                    f"cmd  a(deg) {np.round(deg_mm(q_cmd)[:3], 1)}  b(mm) {np.round(deg_mm(q_cmd)[3:], 1)}",
                    f"samples {len(trainer.buffer)}  live model err {err}  loss {s['loss']:.2f}",
                ]
                if mode == "PREDICT":
                    lines.insert(2, f"preview a(deg) {np.round(deg_mm(q_virt)[:3], 1)}  b(mm) {np.round(deg_mm(q_virt)[3:], 1)}"
                                    "   [ENTER=execute]")
                draw_text(frame, lines)
                cv2.imshow(win, frame)

            # 5) keys (also serves as the window event pump) + pace the loop
            wait = max(1, int((period - (time.time() - now)) * 1000))
            key = cv2.waitKey(wait) & 0xFF
            cmd = keyboard.poll(key, time.time())
            if key in (27, ord('x')):
                break
            elif key == ord('m'):
                mode = "PREDICT" if mode == "TELEOP" else "TELEOP"
                q_virt, exec_target = q_cmd.copy(), None
            elif key == 13 and mode == "PREDICT":
                exec_target = q_virt.copy()
            elif key == ord('h'):
                ctrl.set_hold(meas) if ctrl.hold_target is None else setattr(ctrl, "hold_target", None)
            elif key == ord('t'):
                trainer.enabled = not trainer.enabled and robot.learns
            elif key == ord('z'):
                exec_target, mode = js.home.copy(), "PREDICT"
                q_virt = js.home.copy()
            elif key in (ord('+'), ord('=')):
                speed = min(speed * 1.25, 4.0)
            elif key in (ord('-'), ord('_')):
                speed = max(speed / 1.25, 0.1)
            elif key == ord('p'):
                trainer.save(MODEL_FILE)
                trainer.buffer.save(DATA_FILE)
                print(f"[saved] {MODEL_FILE}, {DATA_FILE}")
    finally:
        trainer.stop()
        robot.stop()
        if len(trainer.buffer):
            trainer.save(MODEL_FILE)
            trainer.buffer.save(DATA_FILE)
            print(f"[saved] {MODEL_FILE}, {DATA_FILE} ({len(trainer.buffer)} samples)")
        robot.close()
        tracker.close()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------- automatic data collection
def run_collect(args, cfg):
    """Drive the robot to random valid configurations, wait until settled, average camera poses."""
    robot, tracker = build_io(args, cfg)
    js = TipController(cfg.robot, cfg.control, ModelServer(cfg.robot)).js
    buf, dirs = ReplayBuffer(max(args.n, 10)), DirectionTracker()
    rng = np.random.default_rng(args.seed)
    q_cmd, dt = robot.read_q(), 1.0 / cfg.control.control_rate
    try:
        while len(buf) < args.n:
            target = js.project(np.concatenate([rng.uniform(-np.pi, np.pi, 3),
                                                rng.uniform(cfg.robot.beta_min, cfg.robot.beta_max)]))
            while np.linalg.norm(q_cmd - target) > 1e-6:                    # move (speed limited)
                q_cmd = js.approach(q_cmd, target, dt)
                robot.command(q_cmd)
                robot.update(dt)
                dirs.update(robot.read_q())
                tracker.read()
                time.sleep(dt if not args.sim else 0)
            t_end, P, Rs = time.time() + cfg.learn.settle_s, [], []       # settle, then average ~15 frames
            while len(P) < 15 and time.time() < t_end + 3.0:
                robot.update(dt)
                m = tracker.read()
                if time.time() > t_end and m.ok:
                    P.append(m.p)
                    Rs.append(m.R)
                time.sleep(dt if not args.sim else 0)
            if len(P) >= 8:
                U, _, Vt = np.linalg.svd(np.mean(Rs, 0))                      # project mean onto SO(3)
                buf.add(robot.read_q(), dirs.d.copy(), np.median(P, 0), U @ Vt)
                print(f"\rcollected {len(buf)}/{args.n}", end="")
    finally:
        buf.save(args.out)
        print(f"\n[saved] {buf.n} samples -> {args.out}")
        robot.close()
        tracker.close()


# ----------------------------------------------------------------------------- offline training
def run_train(args, cfg):
    server, trainer = build_learning(cfg, load=False)
    trainer.buffer.load(args.data)
    n = len(trainer.buffer)
    print(f"[train] {n} samples")
    for i in range(args.epochs // 100):
        loss = trainer.train_steps(100)
        if (i + 1) % 5 == 0:
            q, d, p, R = trainer.buffer.sample(min(n, 512), recent_frac=0)
            pp, _ = server.model(q, d)
            print(f"  step {(i + 1) * 100:6d}  loss {loss:7.3f}  tip err {(pp - p).norm(dim=1).mean() * 1e3:6.2f} mm")
    trainer.save(args.model)
    print(f"[saved] {args.model}")
    print("learned physical corrections:",
          {k: np.round(v.detach().numpy(), 3).tolist() for k, v in trainer.model.phys.named_parameters()})


# ----------------------------------------------------------------------------- offline prediction
def run_predict(args, cfg):
    server, trainer = build_learning(cfg)
    ctrl = TipController(cfg.robot, cfg.control, server)
    home = np.array(cfg.robot.home, float)
    q0 = home if args.q is None else np.concatenate([np.radians(args.q[:3]), np.array(args.q[3:]) * 1e-3])
    if args.dq is not None:
        dq = np.concatenate([np.radians(args.dq[:3]), np.array(args.dq[3:]) * 1e-3])
        q0 = ctrl.js.project(q0 + dq)
    if args.target is not None:
        q, err = ctrl.ik(np.array(args.target) * 1e-3, q0)
        print(f"IK solution (a1..a3 [deg], b1..b3 [mm]): {np.round(deg_mm(q), 2)}   residual {err * 1e3:.2f} mm")
        q0 = q
    p, R = server.pose(q0)
    print(f"motors (deg, mm): {np.round(deg_mm(q0), 2)}")
    print(f"predicted tip position [mm]: {np.round(p * 1e3, 2)}")
    print(f"predicted tip direction (z axis): {np.round(R[:, 2], 4)}")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("teleop", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--sim", action="store_true", help="simulated robot + camera")
        s.add_argument("--serial", help="serial port of the Marlin board, e.g. /dev/ttyACM0 or COM4")
        s.add_argument("--baud", type=int, default=None, help="default: config.robot.marlin_baud")
        s.add_argument("--zero-home", action="store_true",
                       help="send G92 to register the robot's CURRENT physical pose as robot.home "
                            "(position it there first!). Needed after every power-cycle/lost steps.")
        s.add_argument("--dry-run", action="store_true", help="camera only, motors not driven")
        if name == "teleop":
            s.add_argument("--gamepad", action="store_true")
        else:
            s.add_argument("--n", type=int, default=300)
            s.add_argument("--out", default=DATA_FILE)
            s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("train")
    s.add_argument("--data", default=DATA_FILE)
    s.add_argument("--model", default=MODEL_FILE)
    s.add_argument("--epochs", type=int, default=3000, help="gradient steps")
    s = sub.add_parser("predict")
    s.add_argument("--q", type=float, nargs=6, help="current motors: a1 a2 a3 [deg] b1 b2 b3 [mm]")
    s.add_argument("--dq", type=float, nargs=6, help="planned motor motion, same units")
    s.add_argument("--target", type=float, nargs=3, help="desired tip xyz [mm] in robot base frame")
    args = ap.parse_args()
    cfg = Config()
    dict(teleop=run_teleop, collect=run_collect, train=run_train, predict=run_predict)[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
