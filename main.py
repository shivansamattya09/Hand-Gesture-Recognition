#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import preprocess
import traindata
import webcam_init
import time
import sys
import threading
import itertools
done = False


def animate1():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')


t = threading.Thread(target=animate1)
print ('WELCOME TO PROTOTYPE OF HAND RECOGNITION.')
dirr = input('Enter your Images Directory one tree before : ')
no = int(input('Enter your number Photo categories : '))
categories = []
for i in range(no):
    n = input('Enter your category name ' + str(i + 1) + ' : ')
    categories.append(n)
t.start()
process = preprocess.Preprocessing(dirr, categories)
done = True
(x, y) = process.start()
training = traindata.Traindata(x, y)
model = training.start()
webcam_init.start(model)
