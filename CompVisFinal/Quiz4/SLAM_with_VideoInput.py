import cv2
import numpy as np

# Initialize ORB detector
orb = cv2.ORB_create()

# Set up video capture from the input video file (replace 'your_video.mp4' with the video file path)
video_path = 'bottletest.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties to save output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer to save the output video with tracked keypoints and camera motion
output_video = cv2.VideoWriter('slam_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

# Store keypoints and descriptors from the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

# Define the camera matrix (you might need to adjust this for real-world cameras)
focal_length = 800
camera_matrix = np.array([[focal_length, 0, frame_width // 2],
                          [0, focal_length, frame_height // 2],
                          [0, 0, 1]])

# Initialize lists to store 3D points (for pose tracking) and the trajectory
trajectory = []
map_points = []

# Colors for drawing keypoints and trajectory
keypoint_color = (0, 255, 0)  # Green for keypoints
trajectory_color = (255, 0, 0)  # Blue for trajectory

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features in the current frame
    curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)
    
    # Match features between previous and current frame
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_des, curr_des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract the matched keypoints
    prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate essential matrix and recover pose (R = rotation, t = translation)
    E, mask = cv2.findEssentialMat(curr_pts, prev_pts, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, curr_pts, prev_pts, camera_matrix)

    # Use the pose (R, t) to update the camera trajectory (store translation vector)
    map_points.append(t.flatten())
    if len(map_points) > 1:
        # Draw trajectory (connecting translation points)
        for i in range(1, len(map_points)):
            pt1 = (int(map_points[i-1][0]) + frame_width // 2, int(map_points[i-1][2]) + frame_height // 2)
            pt2 = (int(map_points[i][0]) + frame_width // 2, int(map_points[i][2]) + frame_height // 2)
            cv2.line(curr_frame, pt1, pt2, trajectory_color, 2)
    
    # Draw the keypoints being tracked
    for match in matches:
        kp = curr_kp[match.trainIdx].pt
        cv2.circle(curr_frame, (int(kp[0]), int(kp[1])), 5, keypoint_color, -1)
    
    # Update previous frame and keypoints for the next iteration
    prev_gray = curr_gray
    prev_kp = curr_kp
    prev_des = curr_des
    
    # Write the frame with the keypoints and trajectory to the output video
    output_video.write(curr_frame)
    
    # Display current frame with keypoints and trajectory
    cv2.imshow('SLAM Output', curr_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
