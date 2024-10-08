import numpy as np
import cv2

# Reading the image
input_image = cv2.imread('bottleorg.jpg')

# Resizing the image according to our need
input_image = cv2.resize(input_image, (480, 480))

# Extracting the height and width of an image
rows, cols = input_image.shape[:2]

# Generating vignette mask using Gaussian kernels
X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)

# Generating resultant_kernel matrix
resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

# Creating mask and normalizing
mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
output = np.copy(input_image)

# Applying the mask to each channel in the input image
for i in range(3):
    output[:, :, i] = output[:, :, i] * mask

# Displaying the original image
cv2.imshow('Original', input_image)

# Displaying the vignette filter image
cv2.imshow('VIGNETTE', output)

# Saving the vignette-filtered image
cv2.imwrite('vignette_output.jpg', output)

# Maintain output window until user presses a key
cv2.waitKey(0)

# Destroying present windows on screen
cv2.destroyAllWindows()
