import cv2
import numpy as np

def nothing(x):
    pass

# Try changing to 0 if the RealSense doesn't open
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 35, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 85, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

print("Press 'q' to quit. Tune the sliders until ONLY your robot tip is white in the Mask window.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nYour tuned HSV values are:")
        print(f"lower_color = np.array([{l_h}, {l_s}, {l_v}])")
        print(f"upper_color = np.array([{u_h}, {u_s}, {u_v}])")
        break

cap.release()
cv2.destroyAllWindows()
