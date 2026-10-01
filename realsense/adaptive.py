import time
import numpy as np
from scipy.linalg import pinv
import tkinter as tk
import cv2
import pyrealsense2 as rs
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

# =============================================================================
# 1. ISOLATED REALSENSE PROCESS (Prevents macOS Segfaults)
# =============================================================================
def realsense_worker(data_queue):
    print("[Camera Process] Starting RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    
    try:
        pipeline.start(config)
        print("[Camera Process] RealSense Active. Tracking Silver Tip...")
        
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # --- TRACKING SILVER/METALLIC ---
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Hue: 0-179 (Ignore, we don't care about color)
            # Sat: 0-50 (Must have very little color/be mostly grayscale)
            # Val: 150-255 (Must be highly reflective/bright)
            lower_silver = np.array([0, 0, 150])
            upper_silver = np.array([179, 50, 255])
            
            mask = cv2.inRange(hsv, lower_silver, upper_silver)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                # Filter out tiny specks of noise
                if cv2.contourArea(c) > 30: 
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # RealSense struggles with shiny metal. We MUST ensure depth is not 0.
                        depth = depth_frame.get_distance(cx, cy)
                        if depth > 0.01: # Must be greater than 1 cm away to be valid
                            depth_mm = depth * 1000 
                            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                            result = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_mm)
                            
                            while not data_queue.empty():
                                try:
                                    data_queue.get_nowait()
                                except:
                                    pass
                            data_queue.put(np.array(result))
                            
    except Exception as e:
        print(f"[Camera Process] Error: {e}")
    finally:
        pipeline.stop()

# =============================================================================
# 2. PHYSICS-ASSISTED ML MODEL & KINEMATICS ENGINE
# =============================================================================
class KinematicResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )
        self.network[4].weight.data.fill_(0.0)
        self.network[4].bias.data.fill_(0.0)

    def forward(self, q):
        return self.network(q)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class AdaptiveCTRKinematics:
    def __init__(self):
        self.ml_model = KinematicResidualNet().to(DEVICE)
        self.optimizer = optim.Adam(self.ml_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def forward_kinematics_math(self, q):
        t1, r1, t2, r2, t3, r3 = q
        exposed = max(0, t1 - t2)
        kappa = 0.005 
        if exposed == 0: return np.array([0.0, 0.0, t1])
        theta = kappa * exposed
        x = ((1 - np.cos(theta)) / kappa) * np.cos(r1)
        y = ((1 - np.cos(theta)) / kappa) * np.sin(r1)
        z = (np.sin(theta) / kappa) + t2
        return np.array([x, y, z])

    def forward_kinematics_hybrid(self, q):
        math_pos = self.forward_kinematics_math(q)
        q_tensor = torch.FloatTensor(q).to(DEVICE)
        with torch.no_grad():
            ml_correction = self.ml_model(q_tensor).cpu().numpy()
        return math_pos + ml_correction

    def calculate_jacobian(self, q, epsilon=1e-5):
        J = np.zeros((3, 6))
        pos_base = self.forward_kinematics_hybrid(q)
        for i in range(6):
            q_perturbed = np.copy(q)
            q_perturbed[i] += epsilon
            pos_perturbed = self.forward_kinematics_hybrid(q_perturbed)
            J[:, i] = (pos_perturbed - pos_base) / epsilon
        return J

    def train_step(self, q_current, target_real_xyz):
        self.ml_model.train()
        self.optimizer.zero_grad()
        
        math_pos = self.forward_kinematics_math(q_current)
        math_pos_tensor = torch.FloatTensor(math_pos).to(DEVICE)
        real_pos_tensor = torch.FloatTensor(target_real_xyz).to(DEVICE)
        
        target_residual = real_pos_tensor - math_pos_tensor
        q_tensor = torch.FloatTensor(q_current).to(DEVICE)
        predicted_residual = self.ml_model(q_tensor)
        
        loss = self.loss_fn(predicted_residual, target_residual)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# =============================================================================
# 3. TKINTER VIRTUAL JOYSTICK (Main Process)
# =============================================================================
class AdaptiveJoystickApp:
    def __init__(self, root, data_queue, camera_process):
        self.root = root
        self.data_queue = data_queue
        self.camera_process = camera_process
        
        self.root.title("macOS Adaptive Joystick")
        self.root.geometry("450x250")
        
        self.kinematics = AdaptiveCTRKinematics()
        
        self.q_current = np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.v_target = np.zeros(3)
        self.joystick_speed = 5.0 
        
        self.active_keys = set()
        self.latest_3d_pos = None
        
        self.setup_ui()
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        
        self.control_loop()
        self.training_loop()

    def setup_ui(self):
        tk.Label(self.root, text="W/S/A/D/Up/Down to move", font=("Arial", 12, "bold")).pack(pady=10)
        self.status_var = tk.StringVar(value="Tip Pos: [0.0, 0.0, 0.0]")
        tk.Label(self.root, textvariable=self.status_var, font=("Courier", 12)).pack(pady=5)
        
        self.loss_var = tk.StringVar(value="ML Loss: Waiting for camera...")
        tk.Label(self.root, textvariable=self.loss_var, font=("Courier", 12), fg="red").pack(pady=5)

    def on_key_press(self, event):
        self.active_keys.add(event.keysym.lower())
        self.update_target_velocity()

    def on_key_release(self, event):
        self.active_keys.discard(event.keysym.lower())
        self.update_target_velocity()

    def update_target_velocity(self):
        vx, vy, vz = 0.0, 0.0, 0.0
        if 'd' in self.active_keys: vx += self.joystick_speed
        if 'a' in self.active_keys: vx -= self.joystick_speed
        if 'w' in self.active_keys: vy += self.joystick_speed
        if 's' in self.active_keys: vy -= self.joystick_speed
        if 'up' in self.active_keys: vz += self.joystick_speed
        if 'down' in self.active_keys: vz -= self.joystick_speed
        self.v_target = np.array([vx, vy, vz])

    def control_loop(self):
        if np.any(self.v_target != 0):
            J = self.kinematics.calculate_jacobian(self.q_current)
            J_pinv = pinv(J)
            q_dot = J_pinv @ self.v_target
            
            delta_q = q_dot * 0.1 
            self.q_current += delta_q

        pos = self.kinematics.forward_kinematics_hybrid(self.q_current)
        self.status_var.set(f"Predicted Tip: X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f}")
        
        self.root.after(100, self.control_loop)

    def training_loop(self):
        # Fetch the latest data from the separated camera process
        if not self.data_queue.empty():
            try:
                self.latest_3d_pos = self.data_queue.get_nowait()
            except:
                pass

        if self.latest_3d_pos is not None:
            loss = self.kinematics.train_step(self.q_current, self.latest_3d_pos)
            self.loss_var.set(f"ML Loss (Error): {loss:.3f}")
        else:
            self.loss_var.set("ML Loss: Target Marker Not Visible")
            
        self.root.after(33, self.training_loop)

    def on_closing(self):
        print("Shutting down cleanly...")
        self.camera_process.terminate() # Kill the background camera process
        self.camera_process.join()
        self.root.destroy()

if __name__ == "__main__":
    # macOS strictly requires 'spawn' instead of 'fork' for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create the data pipe
    data_queue = mp.Queue()
    
    # Launch the camera in an entirely separate OS process
    camera_process = mp.Process(target=realsense_worker, args=(data_queue,))
    camera_process.start()
    
    # Launch GUI in main process
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    app = AdaptiveJoystickApp(root, data_queue, camera_process)
    
    # Ensure background process is killed when window is closed
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()
