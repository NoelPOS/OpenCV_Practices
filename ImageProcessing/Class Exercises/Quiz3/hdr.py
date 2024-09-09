import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Loading exposure images into a list
img_fn = ["bottle1.jpg", "bottle2.jpg", "bottle3.jpg", "bottle.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# Merge exposures to HDR image using Debevec method
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

# Merge exposures to HDR image using Robertson method
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR images
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save with error handling
def convert_to_8bit(image):
    # Handle NaN values by replacing them with 0
    image = np.nan_to_num(image)
    # Clip the values to the range [0, 1]
    image = np.clip(image, 0, 1)
    # Convert to 8-bit image
    return np.clip(image * 255, 0, 255).astype('uint8')

res_debevec_8bit = convert_to_8bit(res_debevec)
res_robertson_8bit = convert_to_8bit(res_robertson)
res_mertens_8bit = convert_to_8bit(res_mertens)

# Save the results using imwrite
cv.imwrite("Debevec_HDR_8bit.jpg", res_debevec_8bit)
cv.imwrite("Robertson_HDR_8bit.jpg", res_robertson_8bit)
cv.imwrite("Mertens_Fusion_8bit.jpg", res_mertens_8bit)

# Plot images using matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(res_debevec_8bit, cv.COLOR_BGR2RGB))
plt.title("Debevec HDR")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(res_robertson_8bit, cv.COLOR_BGR2RGB))
plt.title("Robertson HDR")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(res_mertens_8bit, cv.COLOR_BGR2RGB))
plt.title("Mertens Fusion")
plt.axis('off')

plt.tight_layout()
plt.show()
