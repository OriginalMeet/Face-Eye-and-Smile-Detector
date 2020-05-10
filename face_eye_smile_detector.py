# Homework Solution

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections


def detect(gray, frame):
    # firstly detecting the faces from the webcame image that we are obtaining
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # now detecting the eyes from the faces region of intrest
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.07, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # now detecting the smile from the faces region of intrest
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # converting the video captured to the grey image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # drawing the rectangle around the webcame video.
    canvas = detect(gray, frame)
    # showing the rectangle in the output
    cv2.imshow('Video', canvas)
    # to quit press Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()  # turn off the webcam
cv2.destroyAllWindows()  # close all the windows
