from PIL import Image
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
 
image_path = 'candy.jpg'
 
try:
    # Attempt to open and convert the image
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    kernel1 = np.array([
        [1, 8, 3],
        [1, 8, 3],
        [1, 8, 3]
    ])
    color_image = cv2.imread('candy.jpg')
    kernel1 = kernel1 / np.sum(kernel1)  # Normalize the kernel
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Save grayscale image
    cv2.imwrite('grayscale_image.bmp', gray_image)
 
    # Perform correlation
    correlation_result = scipy.ndimage.correlate(image_array, kernel1)
    correlation_image = Image.fromarray(correlation_result)
    correlation_image.save('correlation_result.bmp')
 
    # Perform convolution
    convolution_result = scipy.ndimage.convolve(image_array, kernel1)
    convolution_image = Image.fromarray(convolution_result)
    convolution_image.save('convolution_result.bmp')
 
    # Show images
    # image.show()
    # correlation_image.show()
    # convolution_image.show()
 
    # Calculate differences
    correlation_diff = np.abs(image_array - correlation_result)
    convolution_diff = np.abs(image_array - convolution_result)
 
    print("Correlation vs Original Image Difference:")
    print(np.mean(correlation_diff))
 
    print("Convolution vs Original Image Difference:")
    print(np.mean(convolution_diff))
    plt.figure(figsize=(12, 12))
 
    plt.subplot(1, 3, 1)
    plt.title('Gray Image')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Convoluted Image')
    plt.imshow(convolution_image, cmap='gray')
    plt.axis('off')
 
    plt.subplot(1, 3, 3)
    plt.title('Correlated Image')
    plt.imshow(correlation_image, cmap='gray')
    plt.axis('off')
 
    plt.show()
    
    
    # Perform the 2D Fourier transform
    fourier_transform = np.fft.fft2(gray_image)
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
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
     
    plt.subplot(1, 2, 2)
    plt.title('Modified Image')
    plt.imshow(modified_image, cmap='gray')
    plt.axis('off')
     
    plt.show()

 
except Exception as e:
    print(f"Error loading or processing image: {e}")

