import cv2
import numpy as np

image = cv2.imread('original2.jpg')


median = cv2.medianBlur(src=image, ksize=5)
 
cv2.imshow('Original', image)
cv2.imshow('Median Blurred', median)
     
cv2.waitKey(5000)
# cv2.imwrite('median_blur.jpg', median)
cv2.destroyAllWindows()