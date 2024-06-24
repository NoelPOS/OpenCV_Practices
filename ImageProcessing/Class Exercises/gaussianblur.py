import cv2
import numpy as np

image = cv2.imread('original2.jpg')

gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5,5),
sigmaX=0, sigmaY=0)
 
cv2.imshow('Original', image)
cv2.imshow('Gaussian Blurred', gaussian_blur)
     
cv2.waitKey()
cv2.imwrite('gaussian_blur.jpg', gaussian_blur)
cv2.destroyAllWindows()