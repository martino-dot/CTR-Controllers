import pyrealsense2 as rs
import numpy as np
import cv2

print("1. Loading RealSense...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    print("2. Starting hardware (If it segfaults, it happens right here)...")
    pipeline.start(config)
    
    print("3. Grabbing frames...")
    # Pull 10 frames to let the camera warm up and adjust exposure
    for i in range(10):
        frames = pipeline.wait_for_frames()
        
    color_frame = frames.get_color_frame()
    if color_frame:
        img = np.asanyarray(color_frame.get_data())
        cv2.imwrite("mac_test.jpg", img)
        print("4. SUCCESS! Look for 'mac_test.jpg' in your folder.")

finally:
    pipeline.stop()
    print("5. Shut down safely.")
