import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.getcwd()
os.chdir("C:\\F\\NMIMS\\DataScience\\Sem-3\\AI\\project\\detection")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("smile_haarcascade.xml")


def detect(gray, frame):
    faces= faceCascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray= gray[y:y+h, x:x+w]
        roi_color= frame[y:y+h, x:x+w]
        eyes= eyeCascade.detectMultiScale(roi_gray, 1.1, 3)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smile = smileCascade.detectMultiScale(roi_gray, 1.7, 30)
        for(sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)     
    return frame
# recognition with webcame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame= video_capture.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas= detect(gray, frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()