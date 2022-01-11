import numpy as np
import cv2
"""
# this chunk of code is to verify the detection of image and make some changes.
img = cv2.imread("image1.jpg", 0)
#img = cv2.resize(img, (300, 300))
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
print(img)
print(img.shape) # returns shape of numpy array
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h. x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitkey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
