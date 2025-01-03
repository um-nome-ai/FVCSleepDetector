import sys
import dlib
from skimage import io
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

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

last5_frames = np.array([0,0,0,0,0])

status="Not Sleeping"

while True:
    ret, frame = cap.read()
    dets = detector(frame, 1)
    vec = np.empty([68, 2], dtype = int)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)

        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        
        for b in range(68):
         vec[b][0] = shape.part(b).x
         vec[b][1] = shape.part(b).y

        right_ear=compute_EAR(vec[42:48])#compute eye aspect ratio for right eye
        left_ear=compute_EAR(vec[36:42])#compute eye aspect ratio for left eye


        last5_frames = np.delete(np.append(last5_frames, (right_ear+left_ear)/2 < 0.2),0,0)
        

        if np.all(last5_frames) and status=="Not Sleeping": #if the avarage eye aspect ratio of lef and right eye less than 0.2, the status is sleeping.
            status="Sleeping"
        elif ~np.any(np.sum(last5_frames)) and status=="Sleeping":
            status="Not Sleeping"
        

        print(status, last5_frames)
        
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()



