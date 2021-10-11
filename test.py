import numpy as np
import cv2
from PIL import Image

canvas = np.ones((700, 1500, 3), dtype="uint8") * 255
green = (0, 0, 0)
cv2.line(canvas, (0, 0), (300, 300), green, thickness=3)
im = Image.fromarray(canvas)
im.save("test.jpg")
