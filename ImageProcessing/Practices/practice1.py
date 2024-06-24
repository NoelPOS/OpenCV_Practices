import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# Load an image
image = cv2.imread('example.jpg', 0)

# Point Operator (Brightness Adjustment)
bright_image = cv2.convertScaleAbs(image, alpha=1, beta=50)

# Linear Filtering (Blurring)
kernel = np.ones((5,5),np.float32)/25
blurred_image = cv2.filter2D(image, -1, kernel)

# Non-Linear Filtering (Median Filter)
median_filtered_image = cv2.medianBlur(image, 5)

# Fourier Transform
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Pyramids (Gaussian Pyramid)
G = image.copy()
gpA = [G]
for i in range(3):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Geometric Transformation (Rotation)
rows, cols = image.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Display images
plt.figure(figsize=(10, 10))
plt.subplot(231), plt.imshow(bright_image, cmap='gray'), plt.title('Brightness Adjusted')
plt.subplot(232), plt.imshow(blurred_image, cmap='gray'), plt.title('Blurred')
plt.subplot(233), plt.imshow(median_filtered_image, cmap='gray'), plt.title('Median Filtered')
plt.subplot(234), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Fourier Transform')
plt.subplot(235), plt.imshow(gpA[2], cmap='gray'), plt.title('Gaussian Pyramid')
plt.subplot(236), plt.imshow(rotated_image, cmap='gray'), plt.title('Rotated')
plt.show()
