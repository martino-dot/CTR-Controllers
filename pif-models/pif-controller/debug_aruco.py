"""Standalone webcam diagnostic for 'the ArUco marker won't track'.

Independent of vision.py/ArucoTracker on purpose: this tool makes NO assumption about which
dictionary or marker IDs you're using, and doesn't require a base+tip marker pair. It just
tells you, every frame, exactly what the camera can see.

    python debug_aruco.py                 # webcam 0
    python debug_aruco.py --cam 1 --width 1920 --height 1080

Read the on-screen / console output:
  - "no cv2.aruco module"      -> you have `opencv-python`, you need `opencv-contrib-python`
                                   (pip uninstall opencv-python; pip install opencv-contrib-python)
  - "no markers detected in ANY dictionary"
        -> lighting/focus/printing problem, not a config problem. Check the list printed below.
  - "found in DICT_XXX: ids [..]" but a different dictionary than config.py's `aruco_dict`
        -> that mismatch alone is enough to make vision.py silently find nothing. Set
           config.vision.aruco_dict to the printed name.
  - IDs found don't include your configured base_id/tip_id
        -> update config.vision.base_id / tip_id, or reprint markers with the right IDs.
  - several DICT_5X5_50/100/250/1000-style variants all match at once
        -> normal: dictionaries in the same family (same grid size) share bit layouts, so a
           marker from one often validates against its bigger siblings too. Pick the smallest
           one that covers the IDs you're using (e.g. DICT_5X5_50 for IDs 0-49).
  - only ONE marker ever appears at a time
        -> vision.py's ArucoTracker needs BOTH the base marker and the tip marker visible in the
           SAME frame at the same time (the base marker defines where the robot's origin is; the
           tip marker's pose is only meaningful relative to it). A single marker on the bench by
           itself will never produce a tracked pose from the full pipeline, even though this tool
           happily reports it as "detected".
"""
from __future__ import annotations
import argparse
import time

import cv2
import numpy as np

# every predefined dictionary, de-duplicated by underlying dictionary id (some names alias the same one)
_ALL = {}
for _name in dir(cv2.aruco) if hasattr(cv2, "aruco") else []:
    if _name.startswith("DICT_"):
        _ALL[getattr(cv2.aruco, _name)] = _name
DICTS = sorted(_ALL.items())    # [(cv2_dict_id, name), ...]
NAME_TO_ID = {name: dict_id for dict_id, name in DICTS}


def get_detector(dict_id):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        prm = cv2.aruco.DetectorParameters()
    except AttributeError:
        prm = cv2.aruco.DetectorParameters_create()
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(d, prm)
        return det.detectMarkers
    return lambda gray: cv2.aruco.detectMarkers(gray, d, parameters=prm)


def sweep_all_dicts(gray):
    """Try every dictionary; return {name: (ids, corners)} for every dict that found something."""
    hits = {}
    for dict_id, name in DICTS:
        corners, ids, _ = get_detector(dict_id)(gray)
        if ids is not None and len(ids):
            hits[name] = (ids.ravel().tolist(), corners)
    return hits


def main():
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "cv2.aruco is not available. You almost certainly have plain `opencv-python`\n"
            "installed instead of `opencv-contrib-python` (aruco lives in the contrib modules).\n"
            "Fix:  pip uninstall opencv-python opencv-python-headless -y\n"
            "      pip install opencv-contrib-python")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--sweep-period", type=float, default=1.0,
                    help="seconds between full 27-dictionary sweeps (every frame otherwise just "
                         "reuses the last dictionary that worked, for a smooth live overlay)")
    args = ap.parse_args()

    print(f"[debug_aruco] cv2 {cv2.__version__}, {len(DICTS)} known dictionaries available")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.cam}. Try a different --cam index (0, 1, 2...).")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[debug_aruco] opened camera {args.cam} at {w}x{h}")

    last_sweep, best_dict, last_report = 0.0, None, None
    win = "ArUco diagnostic  [ESC to quit]"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            if best_dict is None or now - last_sweep > args.sweep_period:
                hits = sweep_all_dicts(gray)
                last_sweep = now
                if hits:
                    best_dict = max(hits, key=lambda k: len(hits[k][0]))     # dict that saw the most markers
                    ids, corners = hits[best_dict]
                    cv2.aruco.drawDetectedMarkers(frame, corners, np.array(ids))
                    report = f"found in {best_dict}: ids {sorted(ids)}" + \
                            (f"   (also matched: {', '.join(k for k in hits if k != best_dict)})"
                             if len(hits) > 1 else "")
                else:
                    best_dict = None
                    report = "no markers detected in ANY of the 27 known dictionaries"
                if report != last_report:
                    print(f"[debug_aruco] {report}")
                    last_report = report
            elif best_dict:
                corners, ids, _ = get_detector(NAME_TO_ID[best_dict])(gray)
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            status = last_report or "..."
            color = (0, 200, 0) if best_dict else (0, 0, 255)
            cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "if nothing found: check lighting, focus, and that the marker is flat & printed with a white border",
                       (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
