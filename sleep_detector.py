import dlib
#from skimage import io
import numpy as np
from scipy.spatial import distance 
import cv2 as cv


def compute_EAR(vec):

	a = distance.euclidean(vec[1], vec[5])
	b = distance.euclidean(vec[2], vec[4])
	c = distance.euclidean(vec[0], vec[3])
	# compute EAR
	ear = (a + b) / (2.0 * c)

	return ear

# if len(sys.argv) != 3:
#     print(
#         "execute this program by running: \n"
#         "python sleep_detection.py  ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat /path/to/image.jpg"
#         "You can download a trained facial shape predictor from:\n "
#         "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv.VideoCapture('face.mp4', cv.CAP_FFMPEG)
if not cap.isOpened():
    print("failed to load video")
    exit()


status="Not Sleeping"
face_per_frame = 0
sleep_frames = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    frame = cv.resize(frame, (450, 450 * (width / height) ))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    dets = detector(frame, 1)
    vec = np.empty([68, 2], dtype = int)

    face_per_frame += len(dets)
    total_frames += 1
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)

        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
        # Draw the face landmarks on the screen.
        
        for b in range(68):
         vec[b][0] = shape.part(b).x
         vec[b][1] = shape.part(b).y

        right_ear = compute_EAR(vec[42:48])#compute eye aspect ratio for right eye
        left_ear = compute_EAR(vec[36:42])#compute eye aspect ratio for left eye

		# se os olhos estiverem fechados, está dormindo
        if ((right_ear + left_ear)/2) < 0.2:
            print("sleeping")
            sleep_frames += 1

        #print(status)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

print(f"porcentagem de tempo com rostos: {float(face_per_frame)* 100.0/float(total_frames)}")
print(f"porcentagem de tempo dormindo: {float(sleep_frames)* 100.0/float(total_frames)}")
    
cap.release()
cv.destroyAllWindows()



# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# frames the eye must be below the threshold
EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            cv2