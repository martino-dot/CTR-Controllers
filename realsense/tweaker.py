import pyrealsense2 as rs
import numpy as np
import cv2

# Global Variables
current_hsv = None
lower_color = np.array([0, 0, 150])
upper_color = np.array([179, 50, 255])
mask_active = False

def on_mouse_click(event, x, y, flags, param):
    global lower_color, upper_color, mask_active, current_hsv
    if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
        h, s, v = current_hsv[y, x]
        print(f"\n[CLICK] Sampled Pixel - Hue: {h}, Sat: {s}, Val: {v}")

        # Since it is silver, Hue doesn't matter (0-179). 
        # We constrain Saturation to be low, and Value (brightness) to be high.
        lower_color = np.array([0, 0, max(0, int(v) - 50)])
        upper_color = np.array([179, max(60, int(s) + 30), 255])

        print(f"--- PASTE THIS INTO YOUR MAIN SCRIPT ---")
        print(f"lower_color = np.array([{lower_color[0]}, {lower_color[1]}, {lower_color[2]}])")
        print(f"upper_color = np.array([{upper_color[0]}, {upper_color[1]}, {upper_color[2]}])")
        mask_active = True

print("Initializing RealSense (Single Thread Mode)...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

cv2.namedWindow("1. Background Removed", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("1. Background Removed", on_mouse_click)

# Start with a clipping distance of 0.4 meters (40 cm)
clipping_distance_in_meters = 0.4 

print("\n--- INSTRUCTIONS ---")
print("Press 'W' to increase background clipping distance.")
print("Press 'S' to decrease background clipping distance.")
print("Adjust W/S until the gray background turns BLACK, but the robot tip is visible.")
print("Then, CLICK the silver tip with your mouse.")
print("Press 'Q' to quit.\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- 1. DEPTH CLIPPING (Delete the gray background) ---
        clipping_distance = clipping_distance_in_meters / depth_scale
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        
        # If the pixel is further than our limit, or if it is 0 (shiny reflection error), make it black
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 0, color_image)
        
        # --- 2. PREPARE FOR MOUSE CLICK ---
        current_hsv = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)
        
        cv2.putText(bg_removed, f"Clipping Dist: {clipping_distance_in_meters:.2f}m (Press W/S)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("1. Background Removed", bg_removed)

        # --- 3. SHOW THE MASK ---
        if mask_active:
            mask = cv2.inRange(current_hsv, lower_color, upper_color)
            cv2.imshow("2. Tracking Mask (White = Tip)", mask)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            clipping_distance_in_meters += 0.05 # Add 5 cm
        elif key == ord('s'):
            clipping_distance_in_meters -= 0.05 # Subtract 5 cm

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
