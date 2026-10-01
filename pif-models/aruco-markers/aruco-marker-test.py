import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
CALIBRATION_FILE = "camera_calib.npz"

# Calibration settings
CHESSBOARD_CORNERS = (9, 6)  # Number of INNER corners (intersections) on the checkerboard
SQUARE_SIZE = 0.025          # Physical size of one checkerboard square in meters (25mm)

# ArUco tracking settings
MARKER_LENGTH = 0.05         # Physical size of the ArUco marker in meters (5cm)

def calibrate_camera():
    print("\n--- CAMERA CALIBRATION ---")
    print(f"Looking for a {CHESSBOARD_CORNERS[0]}x{CHESSBOARD_CORNERS[1]} inner-corner checkerboard.")
    print("Controls:")
    print("  [SPACE] - Capture current frame for calibration (need at least 10)")
    print("  [ENTER] - Finish capturing and compute calibration")
    print("  [Q]     - Quit")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0) scaled by SQUARE_SIZE
    objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    cap = cv2.VideoCapture(0)
    captured_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, None)

        display_frame = frame.copy()
        
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_CORNERS, corners, ret_corners)
            status_color = (0, 255, 0)
            status_text = "Checkerboard Detected - Press SPACE to capture"
        else:
            status_color = (0, 0, 255)
            status_text = "Checkerboard not found"

        cv2.putText(display_frame, f"Captured: {captured_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow('Calibration', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and ret_corners:
            # Refine corner locations before saving
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            captured_frames += 1
            print(f"Captured frame {captured_frames}")
            
        elif key == 13: # ENTER key
            if captured_frames >= 5:
                print("\nComputing calibration parameters. Please wait...")
                break
            else:
                print("Capture at least 5 frames (10+ recommended) before finishing.")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None, None

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(CALIBRATION_FILE, mtx=mtx, dist=dist)
        print("\nCalibration successful and saved to disk!")
        return mtx, dist
    
    return None, None

def track_aruco(mtx, dist):
    print("\n--- ARUCO TRACKING ---")
    print("Press [Q] to quit.")

    # Modern OpenCV >= 4.7.0 ArUco API
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Define the 3D coordinates of the marker corners in its own coordinate system
    # Origin is at the center of the marker
    half_l = MARKER_LENGTH / 2.0
    obj_points = np.array([
        [-half_l,  half_l, 0],
        [ half_l,  half_l, 0],
        [ half_l, -half_l, 0],
        [-half_l, -half_l, 0]
    ], dtype=np.float32)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Draw outlines of detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Flatten the IDs array to avoid subscripting errors
            flat_ids = ids.flatten()

            for i in range(len(flat_ids)):
                # Estimate pose for each marker
                marker_corners = corners[i][0]
                
                # solvePnP calculates the rotation and translation vectors
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    marker_corners, 
                    mtx, 
                    dist, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    # Draw the 3D axes on the marker
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_LENGTH)
                    
                    # Flatten tvec to safely get the Z-axis translation (index 2)
                    distance_cm = tvec.flatten()[2] * 100
                    
                    cv2.putText(frame, f"ID: {flat_ids[i]} Dist: {distance_cm:.1f}cm", 
                                (int(marker_corners[0][0]), int(marker_corners[0][1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("ArUco Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    mtx, dist = None, None

    # Step 1: Load or perform Calibration
    if os.path.exists(CALIBRATION_FILE):
        print(f"Found existing calibration file: {CALIBRATION_FILE}")
        with np.load(CALIBRATION_FILE) as data:
            mtx = data['mtx']
            dist = data['dist']
    else:
        print("No calibration file found. Starting calibration tool...")
        mtx, dist = calibrate_camera()

    # Step 2: Proceed to tracking if calibration was successful
    if mtx is not None and dist is not None:
        track_aruco(mtx, dist)
    else:
        print("Tracking aborted: Valid camera matrix and distortion coefficients are required.")

if __name__ == "__main__":
    main()

def calibrate_and_save_camera(camera_index=0, checkerboard_size=(7, 10), square_size=0.025, save_path="camera_calib.npz"):
    # Prepare 3D object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size # Scale by the 25mm square size

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    cap = cv2.VideoCapture(camera_index)
    print(f"Calibration started. Press 'c' to capture a frame, 's' to save and exit, 'q' to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret_corners:
            cv2.drawChessboardCorners(display_frame, checkerboard_size, corners, ret_corners)

        cv2.imshow('Camera Calibration', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and ret_corners:
            # Refine corner locations before appending
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"Frame captured! Total frames: {len(objpoints)}")

        elif key == ord('s'):
            if len(objpoints) > 0:
                print("Calculating calibration matrices (this may take a moment)...")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                
                np.savez(save_path, K=mtx, dist=dist)
                print(f"Calibration successful! Saved K and dist to {save_path}")
                print(f"RMS Error: {ret:.3f} px")
            else:
                print("No frames captured. Cannot calibrate.")
            break
            
        elif key == ord('q'):
            print("Calibration aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    # cv2.waitKey(1) # macOS requires this to properly close the window

class VisionTracker:
    def __init__(self, camera_index=0, calib_file="camera_calib.npz"):
        self.cap = cv2.VideoCapture(camera_index)
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Load calibration if available, otherwise use fallback
        if os.path.exists(calib_file):
            calib_data = np.load(calib_file)
            self.camera_matrix = calib_data['K']
            self.dist_coeffs = calib_data['dist']
            print(f"Loaded camera calibration from {calib_file}")
        else:
            print(f"WARNING: {calib_file} not found. Using uncalibrated default matrix.")
            self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
            self.dist_coeffs = np.zeros((4,1))
        
        marker_length = 0.01 
        self.marker_3d_edges = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

