import numpy as np
import os
import cv2

import face_recognition as fr

# print(fr)

test_img = cv2.imread('test_img.jpg') # Give path of image which you want to test

faces_detected, gray_img = fr.faceDetection(test_img)
print('Face Detected : ', faces_detected)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml') # Give path of where trainingData.yml is saved

name = {0 : 'Neeraj'} # Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[x:x+h, y:y+h]
    label, confidence = face_recognizer.predict(gray_img)
    confidence = round(confidence,0)
    print('Confidence : ', confidence)
    print('Label : ', label)
    fr.draw_rectangle(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, str(confidence),x, y)

resized_img = cv2.resize(test_img, (700,700))

cv2.imshow("Face Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
