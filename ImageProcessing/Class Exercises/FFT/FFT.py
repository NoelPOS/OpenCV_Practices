import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
# Convert to float32 for better precision
img_float32 = np.float32(img)

# Perform FFT
f = np.fft.fft2(img_float32)
fshift = np.fft.fftshift(f)

# Adjust cut-off frequency (experiment with different values)
rows, cols = img.shape
crow, ccol = rows//2, cols//2
radius = 30  # Adjust this radius to control the cut-off frequency

# Create a mask to zero out low frequencies
mask = np.zeros_like(fshift)
mask[crow-radius:crow+radius+1, ccol-radius:ccol+radius+1] = 1
fshift = fshift * (1 - mask)

# Normalize magnitude spectrum (optional)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)  # Add small value to avoid log(0)

# Inverse FFT
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)  # Take absolute value to remove imaginary component
img_back = np.uint8(np.clip(img_back, 0, 255))  # Convert back to uint8

# Visualization
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('High-PassFiltered Image'), plt.xticks([]), plt.yticks([])
plt.show()