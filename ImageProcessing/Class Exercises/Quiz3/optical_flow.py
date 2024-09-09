import numpy as np
import cv2 as cv
import sys  # Import sys module

# Directly specify the path to the video file
video_path = r'./opticalflowtest.mp4'

# Open the video file
cap = cv.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print('Error: Could not open video file.')
    sys.exit()  # Use sys.exit() to terminate the script

# Get video properties for saving the output video
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the Lucas-Kanade optical flow video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out_lk = cv.VideoWriter('output_lucas_kanade.avi', fourcc, fps, (frame_width, frame_height))

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take the first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print('Error: No frames grabbed!')
    cap.release()
    cv.destroyAllWindows()
    sys.exit()  # Use sys.exit() to terminate the script

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        img = cv.add(frame, mask)
        
        # Save the modified frame
        out_lk.write(img)
        
        cv.imshow('Lucas-Kanade Optical Flow', img)
        
        k = cv.waitKey(30) & 0xff
        if k == 27:  # Esc key to exit
            break
        
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

# Release resources for Lucas-Kanade
cap.release()
out_lk.release()
cv.destroyAllWindows()
