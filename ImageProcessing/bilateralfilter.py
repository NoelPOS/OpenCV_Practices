import cv2
import numpy as np

image = cv2.imread('original2.jpg')
bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
 
cv2.imshow('Original', image)
cv2.imshow('Bilateral Filtering', bilateral_filter)
 
cv2.waitKey(0)
cv2.imwrite('bilateral_filtering.jpg', bilateral_filter)
cv2.destroyAllWindows()
