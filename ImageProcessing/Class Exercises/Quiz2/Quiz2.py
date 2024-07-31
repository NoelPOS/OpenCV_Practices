import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image (example)
color_image = cv2.imread('Quiz2img.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('gray_image.bmp', gray_image)

# Perform Fourier transform
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Create a low-pass filter mask
rows, cols = gray_image.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30  # Radius for the low-pass filter
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Apply the low-pass filter
fshift = dft_shift * mask

# Perform inverse Fourier transform
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# Normalize the inverse FFT image
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

# Save the filtered image
cv2.imwrite('low_pass_filtered_image.bmp', img_back)

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title('Low-pass Filtered Image')

plt.show()

