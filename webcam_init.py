#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy
import hog
import numpy as np


def start(classifier):
    cap = cv2.VideoCapture(0)
    while True:
        (ret, frame) = cap.read()
        frame = cv2.resize(frame, (200, 200))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog_class = hog.FeatureExtract(gray)
        hog_img = hog_class.features(gray)
        hog_img = np.array(hog_img).flatten()
        y = classifier.predict([hog_img])

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
