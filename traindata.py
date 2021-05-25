#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import hog
import preprocess


class Traindata:

    def __init__(self, data, label):
        self.x = data
        self.y = label

    def start(self):
        print ('\ntraining data')
        x_train, x_test, y_train, y_test = train_test_split(self.x,self.y, test_size=0.2, random_state=0)
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(x_train, y_train)
        return classifier
