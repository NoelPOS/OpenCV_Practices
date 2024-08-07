# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:09:57 2024

@author: FLEX5-82HU000BSB
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def high_pass_filter(img, radius):
    """Applies a high-pass filter to an image.

    Args:
        img: Input image.
        radius: Cut-off frequency radius.

    Returns:
        High-pass filtered image.
    """

    img_float32 = np.float32(img)
    f = np.fft.fft2(img_float32)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones_like(fshift)
    mask[crow - radius:crow + radius + 1, ccol - radius:ccol + radius + 1] = 0
    fshift = fshift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return img_back

def low_pass_filter(img, radius):
    """Applies a low-pass filter to an image.

    Args:
        img: Input image.
        radius: Cut-off frequency radius.

    Returns:
        Low-pass filtered image.
    """

    img_float32 = np.float32(img)
    f = np.fft.fft2(img_float32)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros_like(fshift)
    mask[crow - radius:crow + radius + 1, ccol - radius:ccol + radius + 1] = 1
    fshift = fshift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return img_back

# Load image
img = cv2.imread('messi5.jpg', 0)  # Load as grayscale

# Apply filters with different radii
radius_high = 30
radius_low = 50

img_high_pass = high_pass_filter(img, radius_high)
img_low_pass = low_pass_filter(img, radius_low)

# Display images
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_high_pass,cmap='gray')
plt.title('High-Pass'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_low_pass, cmap='gray')
plt.title('Low-Pass'), plt.xticks([]), plt.yticks([])
plt.show()