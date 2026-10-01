"""Motor interface. MarlinRobot talks to a Marlin-firmware board (e.g. BigTreeTech Octopus)."""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
import numpy as np


class RobotInterface(ABC):
    learns = True          # False => measurements do not correspond to real motion (do not train)

    def update(self, dt: float):
        """Called every control cycle (used by the simulator to advance its physics)."""

    @abstractmethod
    def command(self, q: np.ndarray):
        """Send absolute joint targets q = [a1,a2,a3 (rad), b1,b2,b3 (m)]."""

    @abstractmethod
    def read_q(self) -> np.ndarray:
        """Best estimate of the actual joint positions."""

    def stop(self):
        pass

    def close(self):
        pass


class NullRobot(RobotInterface):
    """Dry-run: nothing moves. Useful to test marker tracking and the virtual joystick preview."""
    learns = False

    def __init__(self, rc):
        self.q = np.array(rc.home, float)

    def command(self, q):
        self.q = np.array(q, float)

    def read_q(self):
        return self.q.copy()


class GenericAsciiRobot(RobotInterface):
    """Generic ASCII serial protocol, kept as a template for non-Marlin firmware:
         PC -> MCU :  "MOVE s1 s2 s3 s4 s5 s6\\n"   absolute targets in motor steps
         MCU -> PC :  "POS s1 s2 s3 s4 s5 s6\\n"    (optional) measured step counts
         PC -> MCU :  "STOP\\n"
       Motors 1-3 rotate tubes 1-3, motors 4-6 translate tubes 1-3.
       If the MCU does not report POS, the last commanded position is used as the actual one.
       For a BigTreeTech Octopus / Marlin setup, use MarlinRobot below instead."""

    def __init__(self, rc, port: str, baud: int = 115200, has_feedback: bool = False):
        import serial                                            # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=0.005)
        self.scale = np.array(rc.steps_per_unit, float) * np.array(rc.motor_signs, float)
        self.has_feedback = has_feedback                         # True once your MCU streams POS lines
        self.q_meas = np.array(rc.home, float)
        self.q_cmd = np.array(rc.home, float)
        self._buf = b""
        self.command(self.q_cmd)

    def command(self, q):
        self.q_cmd = np.array(q, float)
        steps = np.round(self.q_cmd * self.scale).astype(int)
        self.ser.write(("MOVE " + " ".join(str(s) for s in steps) + "\n").encode())

    def read_q(self):
        if not self.has_feedback:
            return self.q_cmd.copy()
        self._buf += self.ser.read(self.ser.in_waiting or 1)
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            parts = line.decode(errors="ignore").split()
            if len(parts) == 7 and parts[0] == "POS":
                self.q_meas = np.array([float(x) for x in parts[1:]]) / self.scale
        return self.q_meas.copy()

    def stop(self):
        self.ser.write(b"STOP\n")

    def close(self):
        self.stop()
        self.ser.close()


class MarlinRobot(RobotInterface):
    """Marlin-firmware controller (e.g. BigTreeTech Octopus V1) driven over serial with G-code.

    Axis mapping is config.robot.marlin_axes, in q order [a1,a2,a3,b1,b2,b3]. The default
    ("A","B","C","X","Y","Z") matches: X/Y/Z = the three z-axis translation carriages,
    A/B/C = the three rotation stages, A = tube closest to the tip. Edit config.py if your
    wiring differs. Requires Marlin built with extra linear axes named A/B/C, e.g. in
    Configuration.h:
        #define LINEAR_AXES 6
        #define AXIS4_NAME 'A'
        #define AXIS5_NAME 'B'
        #define AXIS6_NAME 'C'
    with DEFAULT_AXIS_STEPS_PER_UNIT / MAX_FEEDRATE / MAX_ACCELERATION set for A/B/C
    (steps per *degree*, since this class sends A/B/C in degrees and X/Y/Z in mm).

    The host sends absolute-position G1 moves every control tick and treats "ok" as pure flow
    control: a new line is only queued once the previous one has been acknowledged, and a tick
    is silently skipped (never blocked on) if Marlin hasn't caught up yet, so a slow/backlogged
    controller never stalls the vision + learning loop. There is no encoder feedback: read_q()
    returns the last commanded position (open loop) - the ArUco tip measurement is the real
    feedback used for learning and control, exactly as it would be for a perfect open-loop
    stepper system.

    Homing: rotary A/B/C axes normally have no endstops, so this class does not send G28.
    Instead, physically move the robot to a known reference configuration and call
    `zero_at(q_ref)` once (e.g. zero_at(config.robot.home)) to tell Marlin "this is where you
    are right now", via G92. Do this every time the board power-cycles or loses step sync.
    """

    def __init__(self, rc, port: str | None = None, baud: int | None = None,
                 sync_feedrate_limits: bool = True, ser=None, boot_wait: float = 2.5):
        self.rc = rc
        self.axes = list(rc.marlin_axes)
        self.signs = np.array(rc.motor_signs, float)
        if ser is not None:
            self.ser = ser                     # dependency injection, e.g. for testing
        else:
            import serial                       # pip install pyserial
            self.ser = serial.Serial(port, baud or rc.marlin_baud, timeout=0)
        self._buf = b""
        self._waiting = False
        self._dropped = 0
        self.q_cmd = np.array(rc.home, float)
        self._handshake(boot_wait)
        if sync_feedrate_limits:
            self.push_feedrate_limits()
        self.command(self.q_cmd)

    # ---- unit conversion ---------------------------------------------------
    def _to_native(self, q):
        """q [rad, rad, rad, m, m, m] (signed by motor_signs) -> {axis_letter: native value}."""
        qs = np.asarray(q, float) * self.signs
        native = np.empty(6)
        native[:3] = np.degrees(qs[:3])
        native[3:] = qs[3:] * 1000.0
        return dict(zip(self.axes, native))

    def _from_native(self, vals: dict):
        q = self.q_cmd.copy()
        for i, letter in enumerate(self.axes):
            if letter not in vals:
                continue
            v = vals[letter]
            q[i] = (np.radians(v) if i < 3 else v / 1000.0) * self.signs[i]
        return q

    # ---- low-level serial ---------------------------------------------------
    def _write(self, line: str):
        self.ser.write((line.strip() + "\n").encode("ascii"))

    def _send_and_wait(self, line: str, timeout: float = 5.0) -> str:
        """Blocking send, used only during setup (handshake, M203, zero_at)."""
        self._write(line)
        buf, t0 = b"", time.time()
        while time.time() - t0 < timeout:
            buf += self.ser.read(max(1, getattr(self.ser, "in_waiting", 0) or 1))
            low = buf.lower()
            if b"\nok" in low or low.startswith(b"ok"):
                return buf.decode(errors="ignore")
            time.sleep(0.005)
        raise TimeoutError(f"Marlin did not acknowledge: {line!r} (got {buf!r})")

    def _handshake(self, boot_wait: float):
        time.sleep(boot_wait)                              # let the board finish its DTR-triggered reset/reboot
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self._send_and_wait("M110 N0")                      # reset line numbering (harmless, no checksums used)
        self._send_and_wait("G21")                           # millimeters
        self._send_and_wait("G90")                           # absolute positioning

    def _pump(self):
        """Non-blocking: consume whatever Marlin has sent since the last call."""
        try:
            n = self.ser.in_waiting
        except Exception:
            n = 0
        if n:
            self._buf += self.ser.read(n)
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            s = line.decode(errors="ignore").strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("ok"):
                self._waiting = False
            elif s.startswith(("Error", "!!")):
                print(f"[marlin] {s}")
                self._waiting = False               # don't deadlock the control loop on a firmware error

    # ---- RobotInterface ---------------------------------------------------
    def update(self, dt: float):
        self._pump()

    def command(self, q):
        self.q_cmd = np.array(q, float)
        self._pump()
        if self._waiting:
            self._dropped += 1                      # Marlin's queue hasn't caught up; skip this tick
            return
        native = self._to_native(self.q_cmd)
        self._write("G1 " + " ".join(f"{l}{native[l]:.4f}" for l in self.axes) + " F100000")
        self._waiting = True

    def read_q(self):
        return self.q_cmd.copy()           # open loop; the ArUco measurement is the real feedback

    # ---- setup / diagnostics ---------------------------------------------------
    def push_feedrate_limits(self):
        """Push config.robot.max_speed into Marlin's M203 so its planner also respects them
        (belt-and-suspenders: the control loop already rate-limits every joint itself)."""
        v = np.array(self.rc.max_speed, float)
        native = dict(zip(self.axes, np.concatenate([np.degrees(v[:3]) * 60, v[3:] * 1000 * 60])))
        self._send_and_wait("M203 " + " ".join(f"{l}{native[l]:.2f}" for l in self.axes))

    def zero_at(self, q_ref):
        """Tell Marlin its CURRENT physical position corresponds to joint vector q_ref.
        Call once after manually positioning the robot at a known reference (e.g. robot.home)."""
        native = self._to_native(q_ref)
        self._send_and_wait("G92 " + " ".join(f"{l}{native[l]:.4f}" for l in self.axes))
        self.q_cmd = np.array(q_ref, float)

    def poll_position(self, timeout: float = 1.0):
        """Blocking M114 query -> Marlin's own logical position, decoded back to joint units.
        Diagnostic only (e.g. detect skipped steps): NOT used in the control loop, and not the
        same as true feedback since Marlin just reports what it thinks it commanded."""
        resp = self._send_and_wait("M114", timeout)
        vals = {}
        for tok in resp.split("Count")[0].split():   # ignore the trailing "Count X:.. Y:.. Z:.." step counts
            if ":" in tok:
                k, _, v = tok.partition(":")
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
        return self._from_native(vals)

    def stop(self):
        try:
            self._write("M410")                    # quick stop
        except Exception:
            pass

    def close(self):
        self.stop()
        try:
            self.ser.close()
        except Exception:
            pass
