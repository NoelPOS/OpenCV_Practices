import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# Read the grayscale image
color_image = cv2.imread('candy.jpg')

 
# Check if the image was successfully loaded
if color_image is None:
    print('Could not read color_image')
    exit()
    
grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
 
# Perform the 2D Fourier transform
fourier_transform = np.fft.fft2(grayscale_image)
fourier_transform_shifted = np.fft.fftshift(fourier_transform)
 
# Reduce the magnitude of the frequency component at the origin by 75%
m, n = fourier_transform_shifted.shape
center = (m // 2, n // 2)
fourier_transform_shifted[center] *= 0.25
 
# Perform the inverse Fourier transform
inverse_fourier_shifted = np.fft.ifftshift(fourier_transform_shifted)
modified_image = np.fft.ifft2(inverse_fourier_shifted)
modified_image = np.abs(modified_image)
 

# Save results
cv2.imwrite('modified_image.bmp', modified_image)


# Plot and compare the original and modified images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.title('Modified Image')
plt.imshow(modified_image, cmap='gray')
plt.axis('off')
 
plt.show()

