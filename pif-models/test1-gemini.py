import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import serial.tools.list_ports
import serial
import numpy as np
import time

# ==========================================
# 1. Physics-Informed Model Architecture
# ==========================================
class RigidKinematics(nn.Module):
    def forward(self, q):
        """
        Placeholder for your baseline rigid kinematic math.
        Must use PyTorch operations (torch.sin, torch.cos) so gradients flow.
        q shape: (6,) -> [z1, z2, z3, theta1, theta2, theta3]
        Returns: (6,) -> [x, y, z, roll, pitch, yaw]
        """
        # Example dummy rigid mapping (replace with actual Cosserat/constant-curve math)
        pose = torch.zeros(6, dtype=torch.float32)
        pose[2] = q[0] + q[1] + q[2] # Z is roughly the sum of extensions
        pose[5] = q[3] + q[4] + q[5] # Yaw is roughly the sum of rotations
        return pose

class CTRResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rigid_model = RigidKinematics()
        # Neural Network for the residual error (unmodeled physics)
        self.residual_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Outputs task-space error adjustments
        )

    def forward(self, q):
        # Base physics + learned physics error
        return self.rigid_model(q) + self.residual_net(q)

# ==========================================
# 2. Hardware Interfaces (Camera & Octopus)
# ==========================================
class VisionTracker:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        
        # New OpenCV 4.7+ ArUco initialization
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Camera matrix & distortion coefficients (calibrate your RealSense/Webcam!)
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((4,1))
        
        # Define the 3D coordinates of the 4 corners of a 10mm (0.01m) marker
        marker_length = 0.01 
        self.marker_3d_edges = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

    def get_tip_pose(self):
        ret, frame = self.cap.read()
        if not ret: return None

        # New OpenCV 4.7+ detection method
        corners, ids, rejected = self.detector.detectMarkers(frame)
        
        if ids is not None and len(corners) > 0:
            # Use solvePnP instead of the deprecated estimatePoseSingleMarkers
            success, rvec, tvec = cv2.solvePnP(
                self.marker_3d_edges, 
                corners[0][0], 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                # Extract task space (x, y, z) and rotation vectors
                pose = [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0]), 
                        float(rvec[0][0]), float(rvec[1][0]), float(rvec[2][0])]
                return torch.tensor(pose, dtype=torch.float32)
                
        return None

class OctopusController:
    def __init__(self, baud=115200, test_mode=False):
        self.test_mode = test_mode
        self.current_q = torch.zeros(6, dtype=torch.float32)
        
        if self.test_mode:
            print("⚠️ TEST MODE ENABLED: Bypassing hardware connection. Simulating movements.")
            return

        # Auto-detect the macOS USB port for the Octopus board
        port = self._find_octopus_port()
        if port is None:
            raise ConnectionError("Could not find the Big TreeTech Octopus board. Is it plugged in and powered?")
            
        print(f"Connecting to Octopus on port: {port}")
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def _find_octopus_port(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Matches standard macOS Marlin CDC naming conventions
            if "usbmodem" in p.device.lower() or "MARLIN" in p.description.upper():
                return p.device
        return None

    def send_velocity_command(self, q_dot, dt):
        """
        Integrates velocity to position and sends G-code to TMC2209 drivers.
        """
        self.current_q += (q_dot * dt)
        
        # Format G-code for 6 axes (X, Y, Z for extension, A, B, C for rotation)
        cmd = f"G1 X{self.current_q[0]:.3f} Y{self.current_q[1]:.3f} Z{self.current_q[2]:.3f} " \
              f"A{self.current_q[3]:.3f} B{self.current_q[4]:.3f} C{self.current_q[5]:.3f} F3000\n"
        
        if self.test_mode:
            # Print the command to the terminal instead of sending it over USB
            print(f"[VIRTUAL ROBOT] -> {cmd.strip()}")
        else:
            self.ser.write(cmd.encode('utf-8'))
            
        return self.current_q.clone()

# ==========================================
# 3. Main Control Loop
# ==========================================
def main():
    # Initialize components
    model = CTRResidualModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    camera = VisionTracker()
    robot = OctopusController(test_mode=True)
    
    # Control loop timing
    dt = 0.05 # 20 Hz loop
    
    print("Starting physics-informed control loop...")
    
    while True:
        start_time = time.time()
        
        # 1. Get current joint state
        q_current = robot.current_q.clone().requires_grad_(True)
        
        # 2. ACTIVE LEARNING: Visual Feedback & Model Update
        actual_pose = camera.get_tip_pose()
        if actual_pose is not None:
            optimizer.zero_grad()
            predicted_pose = model(q_current)
            
            # Calculate error between what the network predicted and what the camera saw
            loss = loss_fn(predicted_pose, actual_pose)
            
            # Backpropagate to update the residual network weights in real-time
            loss.backward()
            optimizer.step()
            
            # Re-detach q_current after backward pass for the next operations
            q_current = q_current.detach().requires_grad_(True)
        
        # 3. Read Virtual Joystick (Placeholder)
        # v_desired is [dx, dy, dz, droll, dpitch, dyaw] in task space
        # Here we simulate pushing the joystick "forward" along the Z axis
        v_desired = torch.tensor([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        # 4. INVERSE KINEMATICS via PyTorch Autograd
        # Dynamically compute the Jacobian J(q) = d(pose) / d(q)
        def model_forward_only(q_in):
            return model(q_in)
            
        J = torch.autograd.functional.jacobian(model_forward_only, q_current)
        
        # Compute the Moore-Penrose pseudo-inverse of the Jacobian
        # We add a small damping factor (lambda) for singularity robustness (Damped Least Squares)
        damping = 0.01
        J_T = J.T
        J_inv = J_T @ torch.linalg.inv(J @ J_T + damping * torch.eye(6))
        
        # Calculate required joint velocities: q_dot = J_inv * v_desired
        q_dot = torch.matmul(J_inv, v_desired)
        
        # 5. Command Hardware
        # Prevent massive spikes if near a singularity
        q_dot = torch.clamp(q_dot, min=-5.0, max=5.0) 
        robot.send_velocity_command(q_dot, dt)
        
        # 6. Loop Timing enforcement
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    main()
