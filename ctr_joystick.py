import serial
import serial.tools.list_ports
import time
import numpy as np
from scipy.linalg import pinv
import tkinter as tk

# =============================================================================
# 1. CALIBRATION & CONFIGURATION
# =============================================================================
# Adjust these variables to match your physical robot and board setup
ROBOT_CONFIG = {
    # Tube physical properties
    'tube_1': {'length_mm': 300, 'curvature': 0.005, 'stiffness': 1.0}, # Inner
    'tube_2': {'length_mm': 250, 'curvature': 0.010, 'stiffness': 2.5}, # Middle
    'tube_3': {'length_mm': 200, 'curvature': 0.015, 'stiffness': 5.0}, # Outer
    
    # Motor axis mapping (Translations in mm, Rotations in degrees)
    # T = Translation (Insertion), R = Rotation (Twist)
    'axes': {
        'T1': 'X', 'R1': 'A',
        'T2': 'Y', 'R2': 'B',
        'T3': 'Z', 'R3': 'C'
    },
    
    # Loop rate (Hz) for the joystick update
    'update_rate_hz': 10,
    
    # Maximum safe limits
    'max_insertion_mm': 150.0,
    'max_velocity_mm_s': 20.0
}

# =============================================================================
# 2. KINEMATICS ENGINE
# =============================================================================
class CTRKinematics:
    def __init__(self, config):
        self.config = config

    def forward_kinematics(self, q):
        """
        Calculates the 3D tip position based on 6 motor joint values.
        q = [T1, R1, T2, R2, T3, R3] (Translations in mm, Rotations in radians)
        
        NOTE: This is a simplified constant-curvature placeholder. 
        For true Cosserat Rod mechanics, replace this block with your 
        scipy.integrate.solve_bvp boundary value solver here.
        """
        # --- PLACEHOLDER FK MODEL ---
        # A simple model assuming the dominant outer tube determines the base shape, 
        # and exposed inner tubes extend from there. 
        # Returns [x, y, z] tip coordinate.
        
        # Unpack joints
        t1, r1, t2, r2, t3, r3 = q
        
        # Example math: Calculate arc based on tube 1 extending past tube 2
        exposed_length = max(0, t1 - t2)
        kappa = self.config['tube_1']['curvature']
        
        if exposed_length == 0 or kappa == 0:
            return np.array([0.0, 0.0, t1])
            
        theta = kappa * exposed_length
        x = ((1 - np.cos(theta)) / kappa) * np.cos(r1)
        y = ((1 - np.cos(theta)) / kappa) * np.sin(r1)
        z = (np.sin(theta) / kappa) + t2 # Base offset by tube 2
        
        return np.array([x, y, z])

    def calculate_jacobian(self, q, epsilon=1e-5):
        """
        Calculates the 3x6 numerical Jacobian matrix.
        J maps joint velocities to tip velocities: v = J * q_dot
        """
        J = np.zeros((3, 6))
        pos_base = self.forward_kinematics(q)
        
        for i in range(6):
            q_perturbed = np.copy(q)
            q_perturbed[i] += epsilon
            pos_perturbed = self.forward_kinematics(q_perturbed)
            
            # Partial derivative: (change in position) / (change in joint)
            J[:, i] = (pos_perturbed - pos_base) / epsilon
            
        return J

    def inverse_kinematics_velocity(self, v_target, q_current):
        """
        Converts desired XYZ tip velocity into 6 motor joint velocities using
        the Moore-Penrose pseudo-inverse.
        """
        J = self.calculate_jacobian(q_current)
        
        # Use pseudo-inverse for redundant systems (3 task DOFs, 6 joint DOFs)
        # q_dot = J^+ * v
        J_pinv = pinv(J)
        q_dot = J_pinv @ v_target
        
        return q_dot

# =============================================================================
# 3. LOW-LEVEL HARDWARE DRIVER (Octopus Board)
# =============================================================================
class OctopusController:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  
            self.send_command("M114", silent=True) 
            self.send_command("M302 P1", silent=True) 
            self.send_command("G92 X0 Y0 Z0 A0 B0 C0", silent=True)
            self.send_command("G91", silent=True)  # Relative positioning
        except serial.SerialException as e:
            print(f"Serial Error: {e}")
            self.ser = None

    def send_command(self, cmd, silent=False):
        if not self.ser: return
        self.ser.write((cmd + '\n').encode('utf-8'))
        response = []
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if line: response.append(line)
            if line == 'ok' or not line: break
        return response

    def move_motors_sync(self, moves, speed=1000):
        """Sends a G1 command. Example moves: {'X': 1.2, 'A': 5.0}"""
        if not self.ser: return
        axis_str = " ".join([f"{axis.upper()}{round(dist, 4)}" for axis, dist in moves.items() if dist != 0])
        if axis_str:
            cmd = f"G1 {axis_str} F{speed}"
            self.send_command(cmd, silent=True)

    def close(self):
        if self.ser:
            self.send_command("M84", silent=True)
            self.ser.close()

def find_board():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "OCTOPUS" in p.description.upper(): # or "MARLIN" in p.manufacturer.upper():
            return p.device
    return None

# =============================================================================
# 4. VIRTUAL JOYSTICK & MAIN LOOP (Tkinter GUI)
# =============================================================================
class VirtualJoystickApp:
    def __init__(self, root, kinematics, hardware):
        self.root = root
        self.root.title("CTR Virtual Joystick")
        self.root.geometry("400x300")
        
        self.kinematics = kinematics
        self.hardware = hardware
        
        # Current robot state: [T1, R1, T2, R2, T3, R3]
        # self.q_current = np.zeros(6) 
        self.q_current = np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Start with inner tube extended by 50 mm.
        
        # Desired Cartesian tip velocity [Vx, Vy, Vz]
        self.v_target = np.zeros(3)
        self.joystick_speed = 5.0 # mm/s
        
        self.setup_ui()
        self.setup_keybindings()
        
        # Start the control loop
        self.dt = 1.0 / ROBOT_CONFIG['update_rate_hz']
        self.control_loop()

    def setup_ui(self):
        tk.Label(self.root, text="CTR Camera-Relative Control", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(self.root, text="W / S : Move Tip +Y / -Y").pack()
        tk.Label(self.root, text="A / D : Move Tip -X / +X").pack()
        tk.Label(self.root, text="Up / Down : Move Tip +Z / -Z (In/Out)").pack()
        
        self.status_var = tk.StringVar(value="Tip Pos: [0.0, 0.0, 0.0]")
        tk.Label(self.root, textvariable=self.status_var, font=("Courier", 10), fg="blue").pack(pady=20)

    def setup_keybindings(self):
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        # Prevent key repeating issues in Tkinter
        self.active_keys = set()

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
        # 1. Check if joystick is commanding movement
        if np.any(self.v_target != 0):
            # 2. Calculate Inverse Kinematics (Joint Velocities)
            q_dot = self.kinematics.inverse_kinematics_velocity(self.v_target, self.q_current)
            
            # 3. Calculate distance delta for this time step (distance = velocity * time)
            delta_q = q_dot * self.dt
            self.q_current += delta_q
            
            # 4. Map theoretical joints to physical G-code axes
            axes = ROBOT_CONFIG['axes']
            # Convert radians back to degrees for the rotation stepper motors
            moves = {
                axes['T1']: delta_q[0], axes['R1']: np.degrees(delta_q[1]),
                axes['T2']: delta_q[2], axes['R2']: np.degrees(delta_q[3]),
                axes['T3']: delta_q[4], axes['R3']: np.degrees(delta_q[5])
            }
            
            # Convert update rate to feedrate (mm/minute)
            # Find the max travel time needed to sync the move properly
            feedrate = ROBOT_CONFIG['max_velocity_mm_s'] * 60 
            
            # 5. Send command to hardware
            self.hardware.move_motors_sync(moves, speed=feedrate)

        # 6. Update UI
        pos = self.kinematics.forward_kinematics(self.q_current)
        self.status_var.set(f"Tip Pos: X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f}")
        
        # Re-run loop based on hz
        self.root.after(int(self.dt * 1000), self.control_loop)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Initializing Kinematics Engine...")
    kinematics = CTRKinematics(ROBOT_CONFIG)
    
    port = find_board()
    if not port:
        print("Warning: Octopus board not found. Running in simulation mode.")
        hardware = OctopusController(port=None) # Will swallow commands harmlessly
    else:
        hardware = OctopusController(port=port)
    
    # Launch GUI and Event Loop
    root = tk.Tk()
    app = VirtualJoystickApp(root, kinematics, hardware)
    
    try:
        root.mainloop()
    finally:
        hardware.close()
        print("System shutdown securely.")
