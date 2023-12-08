import cv2
import numpy as np
from numpy import sqrt
from PIL import ImageFilter
from PIL import Image

import cv2, os

video_path = "RAW_01.mov"
output_directory = "Cap_Frames_01"


video_capture = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    success, frame = video_capture.read()
    if not success:
        break

    output_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_path, frame)
    frame_count += 1

video_capture.release()
print(f"Total frames extracted: {frame_count}")

def track(frame_number):    
    img = Image.open('Cap_Frames_01/frame_' + str(frame_number) + '.jpg')
    cpr = img.crop(
        (200, 500, 900, 1400)
    )

    ftd = cpr.filter(ImageFilter.GaussianBlur(radius=0))

    ftd.save('Chopped.jpg')
    #image = cv2.imread(cpr)
    image = cv2.imread('Chopped.jpg')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_color = np.array([179, 255, 255])
    lower_color = np.array([28, 6, 122])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    edges = cv2.Canny(mask, 0, 300)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=3, minLineLength=10, maxLineGap=50)

    highest_line = None
    highest = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if sqrt((x2-x1)**2 + (y2-y1)**2) > highest:
                highest_line = line
                highest = sqrt((x2-x1)**2 + (y2-y1)**2)
    else:
        highest = 0

    if highest_line is not None:
        x1, y1, x2, y2 = highest_line[0]
        cv2.line(hsv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return sqrt((330 - x2)**2 + (420 - y2)**2)

x = np.array([])
for i in range(156, 630):
    x = np.append(x, track(i))

from math import asin

data_01 = np.array([])
for i in range(0, len(x)):
    data_01 = np.append(data_01, asin(x[i]/540))
print(len(data_01), data_01)

import matplotlib.pyplot as plt

plt.scatter(range(0, len(data_01[500:600]), data_01[500:600]))
plt.show()