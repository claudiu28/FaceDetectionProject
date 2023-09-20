import cv2
import os
# Initialize the classifier
FaceCascade = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FaceCascade)
video_capture = cv2.VideoCapture(0)
while True:
    # <- frame by frame
    retains, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # faces <- detectMultiScale() <- docs : https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=0)

    # To draw a rectangle, you need top-left corner and bottom-right corner of rectangle. This will be shown around
    # the faces,
    # last 2 parameters are color and
    # thickness of rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 255), 3)

    cv2.imshow('Face detection from webcam...', frames)

    # Press q for stop the program, basic format for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
