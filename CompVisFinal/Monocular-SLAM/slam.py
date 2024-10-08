import cv2
import numpy as np
import os, sys
from triangulation import triangulate
from Camera import denormalize, normalize, Camera
from display import Display
from match_frames import generate_match
from descriptor import Descriptor

F = int(os.getenv("F", "500"))
W, H = 1920 // 2, 1080 // 2
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

desc_dict = Descriptor()
disp = Display(W, H)

# VideoWriter object to save the output video
output_video_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (W, H))

def calibrate(image):
    image = cv2.resize(image, (W, H))
    return image

def generate_SLAM(image, frame_id):
    image = calibrate(image)
    print("Processing frame:", frame_id)
    frame = Camera(desc_dict, image, K)
    if frame.id == 0:
        return
    frame1 = desc_dict.frames[-1]
    frame2 = desc_dict.frames[-2]

    x1, x2, Id = generate_match(frame1, frame2)
    frame1.pose = np.dot(Id, frame2.pose)

    pts4d = triangulate(frame1.pose, frame2.pose, frame1.key_pts[x1], frame2.key_pts[x2])
    pts4d /= pts4d[:, 3:]  # Convert to homogeneous coordinates
    unmatched_points = np.array([frame1.pts[i] is None for i in x1])
    print("Adding: %d points" % np.sum(unmatched_points))

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        desc_dict.points.append(p)  # Store 3D point in a list
        frame1.add_observation(x1[i], p)
        frame2.add_observation(x2[i], p)

    for pt1, pt2 in zip(frame1.key_pts[x1], frame2.key_pts[x2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(image, (u1, v1), color=(0, 255, 0), radius=1)
        cv2.line(image, (u1, v1), (u2, v2), color=(255, 255, 0))

    # Save each processed image as an individual file (optional)
    # cv2.imwrite(f'output/frame_{frame_id}.png', image)

    # Write the frame to the video
    out.write(image)

    # Display the image
    if disp is not None:
        disp.display2D(image)

    desc_dict.display()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s takes in .mp4 as an argument" % sys.argv[0])
        exit(-1)
    
    cap = cv2.VideoCapture(sys.argv[1])  
    frame_id = 0  # Initialize frame ID
    while cap.isOpened():
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (720, 400))  
        if ret:
            cv2.imshow("Frame", frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break
            generate_SLAM(frame, frame_id)  # Pass frame_id to keep track of frames
            frame_id += 1  # Increment frame ID
        else:
            break
    
    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()
