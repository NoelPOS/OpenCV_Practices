import cv2 as cv


video = cv.VideoCapture(0)  
# video = cv.VideoCapture('video.mp4')  


subtractor = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50)

while True:
    ret, frame = video.read()
    if ret:
        
        mask = subtractor.apply(frame)

        
        cv.imshow("Mask", mask)

        
        if cv.waitKey(5) == ord('x'):
            break
    else:
        
        break


cv.destroyAllWindows()
video.release()
