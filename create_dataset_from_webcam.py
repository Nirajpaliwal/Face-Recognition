import cv2
import sys
import os

i = 0

videoStream =  cv2.VideoCapture(0)

while True:

    ret,frame = videoStream.read() # read frame and return code

    cv2.imshow("Test Frame", frame) # show image in window

    cv2.imwrite(r'/Users/nirajpaliwal/Documents/Python For Data Science & ML/Data-Science-Projects/06 - Face Recognition/images/0/image%04i.jpg' %i, frame)
    i += 1

    if cv2.waitKey(10) == ord('q'):
        break

# print(os.getcwd())