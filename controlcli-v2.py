import serial
import serial.tools.list_ports
import time

def find_octopus_port():
    print("Scanning USB ports for MARLIN_BIGTREE_OCTOPUS_V1...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        device_info = f"{port.description} {port.hwid} {port.manufacturer}".upper()
        if "OCTOPUS_V1" in device_info or "MARLIN_BIGTREE" in device_info:
            print(f"--> Found board on port: {port.device} ({port.description})")
            return port.device
    return None

class OctopusController:
    def __init__(self, port, baudrate=115200):
        print(f"Connecting to {port} at {baudrate} baud...")
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  
            
            # Initialize positions tracker
            self.positions = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'A': 0.0, 'B': 0.0, 'C': 0.0}
            
            # Configure board state
            self.send_command("M114", silent=True) 
            self.send_command("M302 P1", silent=True) # Allow cold extrusion
            
            # Set current physical position as 0 for all axes, then set to Relative mode
            self.send_command("G92 X0 Y0 Z0 A0 B0 C0", silent=True)
            self.send_command("G91", silent=True)  
            
            print("Successfully connected and initialized.\n")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            exit(1)

    def send_command(self, cmd, silent=False):
        self.ser.write((cmd + '\n').encode('utf-8'))
        response = []
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                response.append(line)
            if line == 'ok' or not line:
                break
        if not silent and response:
            for r in response:
                if r != 'ok':
                    print(f"  [Board]: {r}")
        return response

    def move_motor(self, axis, distance, speed=1000):
        cmd = f"G1 {axis}{distance} F{speed}"
        print(f"Executing: {cmd}")
        self.send_command(cmd)
        
        # Update tracked position
        if axis in self.positions:
            self.positions[axis] += distance

    def move_motors_sync(self, moves, speed=1000):
        axis_str = " ".join([f"{axis.upper()}{distance}" for axis, distance in moves.items()])
        cmd = f"G1 {axis_str} F{speed}"
        print(f"Executing Sync Move: {cmd}")
        self.send_command(cmd)
        
        # Update tracked positions
        for axis, distance in moves.items():
            if axis in self.positions:
                self.positions[axis] += distance

    def return_to_home(self, speed=1500):
        """Returns all motors to their starting coordinates."""
        print("\nReturning all motors to start positions...")
        self.send_command("G90", silent=True) # Switch to Absolute Positioning
        self.send_command(f"G1 X0 Y0 Z0 A0 B0 C0 F{speed}") # Move to absolute 0
        self.send_command("G91", silent=True) # Switch back to Relative Positioning
        
        # Reset coordinate tracker
        for axis in self.positions:
            self.positions[axis] = 0.0
        print("Motors are homed.")

    def print_positions(self):
        print("\n--- Current Motor Positions ---")
        for axis, pos in self.positions.items():
            print(f"  {axis}: {pos}")
        print("-------------------------------")

    def enable_motors(self):
        self.send_command("M17", silent=True)
        print("Motors enabled.")

    def disable_motors(self):
        self.send_command("M84", silent=True)
        print("Motors disabled (freewheeling).")

    def close(self):
        self.ser.close()
        print("Connection closed.")


def process_command(board, user_input):
    user_input = user_input.split('#')[0].strip().upper()
    if not user_input:
        return True

    if user_input in ['QUIT', 'EXIT']:
        return False
    elif user_input == 'HELP':
        print_instructions()
        return True
    elif user_input == 'ENABLE':
        board.enable_motors()
        return True
    elif user_input == 'DISABLE':
        board.disable_motors()
        return True
    elif user_input in ['POS', 'POSITION']:
        board.print_positions()
        return True
    elif user_input == 'HOME':
        board.return_to_home()
        return True

    parts = user_input.split()
    cmd_type = parts[0]

    if cmd_type == 'PAUSE':
        if len(parts) < 2:
            print("Error: Provide seconds to pause. (e.g., PAUSE 1.5)")
        else:
            try:
                seconds = float(parts[1])
                ms = int(seconds * 1000)
                print(f"Pausing board for {seconds} seconds...")
                board.send_command(f"G4 P{ms}")
            except ValueError:
                print("Error: Pause duration must be a number.")
        return True

    if cmd_type == 'SYNC':
        args = parts[1:]
        if len(args) < 2:
            print("Error: Provide at least one axis and distance. (e.g., SYNC X 10 A -5 2000)")
            return True
        try:
            speed = 1500
            if len(args) % 2 != 0:
                speed = int(args.pop())
            moves = {args[i]: float(args[i+1]) for i in range(0, len(args), 2)}
            board.move_motors_sync(moves, speed)
        except (ValueError, IndexError):
            print("Error: Invalid SYNC format.")
        return True

    valid_axes = ['X', 'Y', 'Z', 'A', 'B', 'C']
    if cmd_type in valid_axes:
        if len(parts) < 2:
            print(f"Error: Please provide a distance.")
            return True
        try:
            distance = float(parts[1])
            speed = int(parts[2]) if len(parts) >= 3 else 1500
            board.move_motor(cmd_type, distance, speed)
        except ValueError:
            print("Error: Distance and speed must be numbers.")
        return True
        
    if cmd_type == 'FILE':
        if len(parts) < 2:
            print("Error: Provide a filename.")
            return True
        filename = user_input.split(maxsplit=1)[1].lower() 
        try:
            with open(filename, 'r') as f:
                print(f"\n--- Executing file: {filename} ---")
                for line in f:
                    process_command(board, line)
                print(f"--- Finished file: {filename} ---\n")
        except FileNotFoundError:
            print(f"Error: Could not find file '{filename}'.")
        return True

    print(f"Error: Unknown command '{cmd_type}'. Type 'help' for options.")
    return True

def print_instructions():
    print("="*50)
    print("Commands:")
    print("  <Axis> <Dist> [Speed] -> Move motor (Axes: X, Y, Z, A, B, C)")
    print("  SYNC X 10 A 5 [Speed] -> Move multiple motors at once")
    print("  PAUSE 2.5             -> Pause execution for 2.5 seconds")
    print("  FILE sequence.txt     -> Run commands from a text file")
    print("  POS                   -> Show current motor coordinates")
    print("  HOME                  -> Return all motors to starting position")
    print("  quit                  -> Auto-home and exit the script")
    print("="*50)

if __name__ == "__main__":
    target_port = find_octopus_port()
    
    if not target_port:
        print("Error: Could not find MARLIN_BIGTREE_OCTOPUS_V1 connected via USB.")
        exit(1)
        
    board = OctopusController(target_port, baudrate=115200)
    board.enable_motors()
    print_instructions()

    try:
        keep_running = True
        while keep_running:
            user_input = input("\nEnter command: ")
            keep_running = process_command(board, user_input)

    except KeyboardInterrupt:
        print("\nStopping operations by user request...")
    finally:
        # The finally block guarantees this runs even if the script crashes or you type 'quit'
        board.return_to_home()
        board.disable_motors()
        board.close()