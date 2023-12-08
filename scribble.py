import cv2
import numpy as np

image = cv2.imread('pallet.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_color = np.array([0, 0, 4])
upper_color = np.array([133, 255, 255])

mask = cv2.inRange(hsv, lower_color, upper_color)

cv2.imshow('Binary Image', mask)
cv2.imshow('pallet.jpg', image)
cv2.imshow('hsv', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
