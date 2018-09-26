import cv2, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time 
#%matplotlib inline

def convertToRGB(img): 
    #convert the test image to gray image as opencv face detector expects gray images 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#test1 = cv2.imread('test/4.jpg')

#plt.imshow(gray_img, cmap='gray')  
#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
#plt.imshow(convertToRGB(test1))

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    print('Faces found: ', len(faces))
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_copy

test2 = cv2.imread('training-data\s1\1.jpg')

#call our function to detect faces
faces_detected_img = detect_faces(haar_face_cascade, test2)

#conver image to RGB and show image
plt.imshow(convertToRGB(faces_detected_img))