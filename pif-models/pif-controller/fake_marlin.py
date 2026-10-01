"""A tiny fake Marlin device (pyserial-like interface) for exercising MarlinRobot without hardware."""
import re
import numpy as np


class FakeMarlinSerial:
    """Implements the subset of pyserial's API MarlinRobot uses: write(), read(), in_waiting,
    reset_input_buffer(). Emulates Marlin's G1/G92/M114/M203/M110 responses, including queueing
    G1 moves with a small artificial delay before 'ok' (to exercise the flow-control / drop path)."""

    def __init__(self, axes, ok_delay_ticks=0):
        self.axes = axes
        self.q = {a: 0.0 for a in axes}         # native units (deg or mm), Marlin's own bookkeeping
        self.out = bytearray()
        self._inbuf = b""
        self.ok_delay_ticks = ok_delay_ticks
        self._pending_oks = 0
        self.commands_seen = []

    # -- pyserial-like API -----------------------------------------------------
    @property
    def in_waiting(self):
        self._maybe_release()
        return len(self.out)

    def read(self, n=1):
        self._maybe_release()
        n = min(n, len(self.out))
        data, self.out = bytes(self.out[:n]), self.out[n:]
        return data

    def write(self, data: bytes):
        self._inbuf += data
        while b"\n" in self._inbuf:
            line, self._inbuf = self._inbuf.split(b"\n", 1)
            self._handle(line.decode())

    def reset_input_buffer(self):
        self.out = bytearray()

    def close(self):
        pass

    # -- fake firmware ---------------------------------------------------------
    def _maybe_release(self):
        if self._pending_oks and self.ok_delay_ticks == 0:
            self._flush_oks()

    def tick(self):
        """Call once per simulated control step to (eventually) release queued 'ok's."""
        if self._pending_oks:
            self._flush_oks()

    def _flush_oks(self):
        for _ in range(self._pending_oks):
            self.out += b"ok\n"
        self._pending_oks = 0

    def _handle(self, line):
        self.commands_seen.append(line)
        if line.startswith("G1"):
            for a in self.axes:
                m = re.search(rf"{a}(-?[0-9.]+)", line)
                if m:
                    self.q[a] = float(m.group(1))
            self._pending_oks += 1
        elif line.startswith("G92"):
            for a in self.axes:
                m = re.search(rf"{a}(-?[0-9.]+)", line)
                if m:
                    self.q[a] = float(m.group(1))
            self.out += b"ok\n"
        elif line.startswith("M114"):
            parts = " ".join(f"{a}:{self.q[a]:.2f}" for a in self.axes)
            self.out += f"{parts} Count X:0 Y:0 Z:0\nok\n".encode()
        elif line.startswith(("M110", "G21", "G90", "M203", "M410")):
            self.out += b"ok\n"
        else:
            self.out += b"ok\n"
