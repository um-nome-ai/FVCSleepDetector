# import the necessary packages
from scipy.spatial import distance as dist
import imutils
import sys
import cv2


cap = cv2.VideoCapture(sys.argv[1], cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("failed to load video")
    exit()
print(f"frame_rate: {cap.get(cv2.CAP_PROP_FPS)}")
frame_count = 0

# loop over frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)

    cv2.putText(frame, f"frame: {frame_count}", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    while key != ord('n'):
        key = cv2.waitKey(1) & 0xFF

    frame_count += 1

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()