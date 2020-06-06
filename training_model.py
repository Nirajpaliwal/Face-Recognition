import numpy as np
import cv2
import os

import face_recognition as fr

test_img = cv2.imread(r'/Users/nirajpaliwal/Documents/Python For Data Science & ML/Data-Science-Projects/06 - Face Recognition/test_img.jpg')

faces_detected, gray_img = fr.faceDetection(test_img)

# cv2.imshow(gray_img)
print('Face Detected : ', faces_detected)


# Training will begin from here
faces, faceID = fr.labels_for_training_data(r'/Users/nirajpaliwal/Documents/Python For Data Science & ML/Data-Science-Projects/06 - Face Recognition/images')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save(r'/Users/nirajpaliwal/Documents/Python For Data Science & ML/Data-Science-Projects/06 - Face Recognition/trainingData.yml')

name = {0 : 'Neeraj'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rectangle(test_img, face)
    predict_name = name[label]
    fr.put_text(test_img, predict_name, confidence,x, y)

resized_img = cv2.resize(test_img, (1000,700))

cv2.imshow("Face Detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



