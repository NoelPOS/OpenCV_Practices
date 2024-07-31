import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load image
image_path = 'original.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and detect the pose
results = pose.process(image_rgb)

# Draw the pose annotation on the image
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract and print pose landmarks
    for id, lm in enumerate(results.pose_landmarks.landmark):
        print(f'Landmark {id}: ({lm.x}, {lm.y}, {lm.z})')

# Display the image
cv2.imshow('MediaPipe Pose', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = 'annotated_image.jpg'  # Replace with your desired output path
cv2.imwrite(output_path, image)
