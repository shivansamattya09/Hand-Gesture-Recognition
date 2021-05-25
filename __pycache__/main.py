import cv2
import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import hog
import pickle
dirr = '/Users/ilovemarijuana/desktop/Project hand gesture/model Images'
categories = ['1','2']
data = []
labels = []
for i in categories:
    path = os.path.join(dirr,i)
    label = categories.index(i)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        if imgpath[-2]=='p' or imgpath[-2]=='n':
            gest_img = cv2.imread(imgpath,0)
            hog_img = hog.FeatureExtract(gest_img)
            hog_img_final = hog_img.features(gest_img)
            d_img = np.array(hog_img_final).flatten()
            data.append(d_img)
            labels.append(label)
        
        
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size = 0.2,random_state = 0)
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(data,labels)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(200,200))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hog_class = hog.FeatureExtract(gray)
    hog_img = hog_class.features(gray)
    hog_img = np.array(hog_img).flatten()
    y = classifier.predict([hog_img])
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()