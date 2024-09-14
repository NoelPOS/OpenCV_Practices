import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

# Generate Model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Load and process multiple images
image_paths = [f'IMG_{i}.jpg' for i in range(8935, 9044)]  # Updated for six images
depth_images = []

# Initialize variables for width and height
new_width = new_height = None

for image_path in image_paths:
    # Load and resize the image
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)
    
    # Preparing the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Getting prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Post-processing
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    
    depth_images.append(output)

# Ensure depth images have the same size
depth_images_resized = []
for output in depth_images:
    resized_depth_image = np.resize(output, (new_height, new_width))
    depth_images_resized.append(resized_depth_image)

# Combine depth images
combined_depth_image = np.mean(depth_images_resized, axis=0)

# Convert the combined depth image for Open3D
depth_image = (combined_depth_image * 255 / np.max(combined_depth_image)).astype('uint8')
image_np = np.array(image)  # Use the last processed image for RGB data

# Create RGBD image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image_np)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

# Creating a camera
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(new_width, new_height, 500, 500, new_width / 2, new_height / 2)

# Creating Open3D point cloud
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

# Post-processing the 3D point cloud
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
pcd = pcd_raw.select_by_index(ind)

# Estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

o3d.visualization.draw_geometries([pcd])

# Surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# Rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# Visualizing the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# Paint the mesh with the color from the original image
colors = np.asarray(image_np / 255, dtype=np.float32)
mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

# Export the mesh with color
o3d.io.write_triangle_mesh('flower.obj', mesh)
