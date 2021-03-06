#https://stackoverflow.com/questions/50792812/how-to-remove-watermark-background-in-image-python

import cv2
import numpy as np

img = cv2.imread("veidz.jpg")

alpha = 2.0
beta = -160

new = alpha * img + beta
new = np.clip(new, 0, 255).astype(np.uint8)

cv2.imwrite("cleaned.png", new)
