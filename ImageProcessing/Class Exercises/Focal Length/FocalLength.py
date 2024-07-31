# Given:
x_scene = 800  # mm
y_scene = 1200 # mm
z_scene = 2400 # mm
x_image = 48   # mm
y_image = 72   # mm

# Formula: x_image = f * x_scene / z_scene
#          y_image = f * y_scene / z_scene

# Calculate focal length using x-coordinates:
f_x = x_image * z_scene / x_scene
print("Focal length (from x-coordinate):", f_x, "mm")

# Calculate focal length using y-coordinates:
f_y = y_image * z_scene / y_scene
print("Focal length (from y-coordinate):", f_y, "mm")

# Since the aspect ratio is 1, f_x and f_y should be equal.
# In a real scenario, there might be slight differences due to measurement errors.