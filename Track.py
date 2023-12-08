'''import cv2

video_path = "path_to_your_video/video.mp4"
video_capture = cv2.VideoCapture(video_path)

while True:
    success, frame = video_capture.read()

    if not success:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (e.g., using Canny edge detector)
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Perform line detection using the Hough line transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)

    if lines is not None:
        # Iterate over the detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the distance between the two ends of the line
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Draw the line on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Rod Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
'''

import cv2
import numpy as np

image_path = "frame_203.jpg"
frame = cv2.imread(image_path)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 0, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Line Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
