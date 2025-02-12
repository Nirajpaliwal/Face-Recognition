import numpy as np
import cv2
import pandas as pd
import os

# For Detecting Face
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_haar.detectMultiScale(gray_img,scaleFactor=1.3, minNeighbors=3)
    return faces, gray_img


# Labels for training data has been created
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print('Skipping System Files')
                continue

            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print('image_path : ',img_path)
            print('id : ',id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print('Not Loaded Properly')
                continue

            faces_rect, gray_img  = faceDetection(test_img)
            (x,y,w,h) = faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h] # roi --> region of intrest
            faces.append(roi_gray)
            faceID.append(int(id))
    
    return faces, faceID 


# Here training Classifier is called
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

# Drawing a Rectangle on Face Function
def draw_rectangle(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y),  (x+h, y+h),  (0,255,0),  thickness=3)

# Putting text & confidence on image
def put_text(test_img, label_name, confidence, x, y):
    cv2.putText(test_img, label_name + ', ' + confidence+'%', (x,y), cv2.FONT_HERSHEY_TRIPLEX , 2, (255,0,0), 3)

 