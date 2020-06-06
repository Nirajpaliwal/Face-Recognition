import os
import numpy as np
import cv2

import face_recognition as fr

# print(fr)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml') # Give path of where trainingData.yml is saved

cap = cv2.VideoCapture(0) #If you want to recognise face from a video then replace 0 with video path

name = {0 : 'Neeraj'} # Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all

while True:
    ret, test_img = cap.read()
    faces_detected, gray_image = fr.faceDetection(test_img)
    # print('Face Detected : ', faces_detected)

    for(x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w, y+h), (0,255,0),thickness=3)

    for face in faces_detected:
        (x,y,w,h,) = face
        roi_gray = gray_image[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        confidence = round(confidence,0)

        # print('Confidence : ', confidence)
        # print('Label : ', label)
        
        predicted_name = name[label]

        fr.put_text(test_img, predicted_name, str(confidence),x, y)
    
    resized_img = cv2.resize(test_img, (1000,700))
    
    cv2.imshow('Face Detecion', resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

