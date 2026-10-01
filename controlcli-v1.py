import serial
import serial.tools.list_ports
import time

def find_octopus_port():
    """Scans all USB ports and finds the BTT Octopus board automatically."""
    print("Scanning USB ports for MARLIN_BIGTREE_OCTOPUS_V1...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        device_info = f"{port.description} {port.hwid} {port.manufacturer}".upper()
        if "OCTOPUS_V1" in device_info or "MARLIN_BIGTREE" in device_info:
            print(f"--> Found board on port: {port.device} ({port.description})")
            return port.device
            
    return None



class OctopusController:
    def move_motors_sync(self, moves, speed=1000):
        """
        Move multiple motors simultaneously.
        - moves: Dictionary mapping axis to distance (e.g., {'X': 10, 'Y': -20, 'Z': 5})
        - speed: Feedrate in units/minute
        """
        # Parse the dictionary into a G-code coordinate string like "X10 Y-20 Z5"
        axis_str = " ".join([f"{axis.upper()}{distance}" for axis, distance in moves.items()])
        
        cmd = f"G1 {axis_str} F{speed}"
        print(f"Executing Sync Move: {cmd}")
        self.send_command(cmd)

    def __init__(self, port, baudrate=115200):
        print(f"Connecting to {port} at {baudrate} baud...")
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for the serial connection to initialize
            
            # Wake up the board and set to Relative Positioning
            self.send_command("M114", silent=True) 
            self.send_command("G91", silent=True)  
            
            # Allow cold extrusion so extruder motors can spin
            self.send_command("M302 P1", silent=True) 
            
            print("Successfully connected and initialized.\n")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            exit(1)

    def send_command(self, cmd, silent=False):
        """Sends a G-code command and reads the response."""
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
        """Send the movement command to the board."""
        cmd = f"G1 {axis}{distance} F{speed}"
        print(f"Executing: {cmd}")
        self.send_command(cmd)

    def enable_motors(self):
        self.send_command("M17", silent=True)
        print("Motors enabled.")

    def disable_motors(self):
        self.send_command("M84", silent=True)
        print("Motors disabled (freewheeling).")

    def close(self):
        self.ser.close()
        print("Connection closed.")


def print_instructions():
    print("="*50)
    print("INTERACTIVE MOTOR CONTROL")
    print("="*50)
    print("Commands:")
    print("  <Axis> <Distance> [Speed]  -> Move a motor.")
    print("                                Valid Axes: X, Y, Z, A, B, C")
    print("                                Example 1: X 10       (Moves X by 10 at default speed)")
    print("                                Example 2: Y -5 2000  (Moves Y back by 5 at 2000 speed)")
    print("                                Example 3: E1 15      (Moves Extruder 1 by 15)")
    print("  SYNC                       -> Move multiple motors at the same time")
    print("                                Example 1: SYNC X 10 Y -20 Z 5 2000 (Moves all three axis at once at 2000 speed)")
    print("  enable                     -> Lock motors")
    print("  disable                    -> Unlock motors (freewheel)")
    print("  help                       -> Show this menu again")
    print("  quit                       -> Exit the script")
    print("="*50)

def process_command(board, user_input):
    """Parses and executes a single command string."""
    # Strip comments and whitespace
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

    parts = user_input.split()
    cmd_type = parts[0]

    # Handle Pause/Time control
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

    # Handle SYNC moves
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
            print("Error: Invalid SYNC format. Check distances and speeds.")
        return True

    # Handle single axis moves (Now updated for A, B, C instead of Extruders)
    valid_axes = ['X', 'Y', 'Z', 'A', 'B', 'C']
    if cmd_type in valid_axes:
        if len(parts) < 2:
            print(f"Error: Please provide a distance. (e.g., '{cmd_type} 10')")
            return True
        try:
            distance = float(parts[1])
            speed = int(parts[2]) if len(parts) >= 3 else 1500
            board.move_motor(cmd_type, distance, speed)
        except ValueError:
            print("Error: Distance and speed must be numbers.")
        return True
        
    # Handle reading from a file
    if cmd_type == 'FILE':
        if len(parts) < 2:
            print("Error: Provide a filename. (e.g., FILE sequence.txt)")
            return True
            
        # The filename might contain lowercase letters, so grab it from the original un-uppercased string
        filename = user_input.split(maxsplit=1)[1].lower() 
        try:
            with open(filename, 'r') as f:
                print(f"\n--- Executing file: {filename} ---")
                for line in f:
                    # Pass each line back into this same function
                    process_command(board, line)
                print(f"--- Finished file: {filename} ---\n")
        except FileNotFoundError:
            print(f"Error: Could not find file '{filename}'.")
        return True

    print(f"Error: Unknown command '{cmd_type}'. Type 'help' for options.")
    return True


if __name__ == "__main__":
    target_port = find_octopus_port()
    
    if not target_port:
        print("Error: Could not find MARLIN_BIGTREE_OCTOPUS_V1 connected via USB.")
        exit(1)
        
    board = OctopusController(target_port, baudrate=115200)
    board.enable_motors()
    
    # Update instructions for the new commands
    print("="*50)
    print("Commands:")
    print("  <Axis> <Dist> [Speed] -> Move motor (Axes: X, Y, Z, A, B, C)")
    print("  SYNC X 10 A 5 [Speed] -> Move multiple motors at once")
    print("  PAUSE 2.5             -> Pause execution for 2.5 seconds")
    print("  FILE sequence.txt     -> Run commands from a text file")
    print("  quit                  -> Exit the script")
    print("="*50)

    try:
        keep_running = True
        while keep_running:
            user_input = input("\nEnter command: ")
            keep_running = process_command(board, user_input)

    except KeyboardInterrupt:
        print("\nStopping operations by user request...")
    finally:
        board.disable_motors()
        board.close()
