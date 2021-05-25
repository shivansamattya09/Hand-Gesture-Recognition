#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import hog
import numpy as np
import os


class Preprocessing:

    def __init__(self, dirr, categories):
        self.dirr = dirr
        self.categories = categories

    def start(self):
        data = []
        labels = []
        for i in self.categories:
            path = os.path.join(self.dirr, i)
            label = self.categories.index(i)
            for img in os.listdir(path):
                imgpath = os.path.join(path, img)
                if imgpath[-2] == 'p' or imgpath[-2] == 'n':
                    gest_img = cv2.imread(imgpath, 0)
                    hog_img = hog.FeatureExtract(gest_img)
                    hog_img_final = hog_img.features(gest_img)
                    d_img = np.array(hog_img_final).flatten()
                    data.append(d_img)
                    labels.append(label)
        return (data, labels)
