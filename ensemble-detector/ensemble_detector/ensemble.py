import cv2
from detection import yolov7_detect

# video = cv2.VideoCapture('../test images/video one.mp4')
# video = cv2.VideoCapture('../test images/2160p.mp4')
video = cv2.VideoCapture('../test images/video 1.mp4')

while True:
    ret, frame = video.read()
    
    res_frame = ESRGAN(frame)
    new_frame = yolov7_detect(res_frame)
    
    cv2.imshow("Window", new_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
# camera.release()
cv2.destroyAllWindows()    