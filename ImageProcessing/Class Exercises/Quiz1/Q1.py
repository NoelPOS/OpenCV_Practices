import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the color image
color_image = cv2.imread('candy.jpg')

# Check if the image is loaded correctly
if color_image is None:
    print('Could not read color_image')
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Save grayscale image
cv2.imwrite('grayscale_image.bmp', gray_image)

# Define the kernel (make sure it's normalized)
kernel1 = np.array([[1, 8, 3],
                    [8, 1, 8],
                    [3, 8, 1]], dtype=np.float32)
kernel1 = kernel1 / np.sum(kernel1)  # Normalize the kernel

# Apply correlation using OpenCV (correlation is similar to convolution with the kernel flipped)
correlated_image = cv2.filter2D(gray_image, -1, kernel1)

# Flip the kernel for convolution
kernel1_flipped = cv2.flip(kernel1, -1)

# Apply convolution using OpenCV
convolved_image = cv2.filter2D(gray_image, -1, kernel1_flipped)


# Save results
cv2.imwrite('correlated_image.bmp', correlated_image)
cv2.imwrite('convolved_image.bmp', convolved_image)

# Plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Correlation Result')
plt.imshow(correlated_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Convolution Result')
plt.imshow(convolved_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('spatial_comparison.bmp')
plt.show()
