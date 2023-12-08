import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
from PIL import ImageFilter

def filter(lower, upper):

    img = Image.open('Cap_Frames_01/frame_300.jpg')
    cpr = img.crop(
        (200, 500, 900, 1400)
    )

    ftd = cpr.filter(ImageFilter.GaussianBlur(radius=0))

    ftd.save('Chopped.jpg')
    #image = cv2.imread(cpr)
    #image = cv2.imread('Chopped.jpg')
    image = cv2.imread('pallet.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_color = np.array(upper)
    lower_color = np.array(lower)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Step 5: Apply the mask to extract the region of interest
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

import matplotlib.pyplot as plt

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot data in the first subplot (top-left)
axes[0].imshow(Image.open('pallet.jpg'))
axes[0].set_title('Raw Image')

# Plot data in the second subplot (top-right)
lower, upper = [0, 0, 0], [150, 60, 100]

axes[1].imshow(filter(upper, lower))
axes[1].set_title('Segmented')

# Adjust the layout and spacing between subplots
plt.tight_layout()

# Show the figure with all subplots
plt.show()
