import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Generate points on two circles
def generate_circle_points(center, radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

# Parameters
r = 5
c1 = [0, 0]  # Center of the first circle
c2 = [2, 2]  # Center of the second circle (inside the first circle)
n_points = 1000  # Number of points per circle

# Generate points
circle1 = generate_circle_points(c1, r, n_points)
circle2 = generate_circle_points(c2, 2*r, n_points)

# Combine all points
X = np.vstack((circle1, circle2))

# Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize the results
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r', label='Centroids')
plt.title('K-means Clustering on Two Circles')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
