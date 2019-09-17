#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Bilder_Defokus_Modul.py


import cv2
import numpy as np
from matplotlib import pyplot as plt

# fig, ax = plt.subplots(figsize=(100, 500 / 10), dpi=10, subplot_kw={'aspect': 'equal'})

src = cv2.imread('D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/JPEGImages/001371.png', cv2.IMREAD_UNCHANGED)

# apply guassian blur on src image
dst = cv2.GaussianBlur(src, (9, 9), cv2.BORDER_DEFAULT)

# display input and output image
# cv2.imshow("Defokus_Gaussian_Filter", np.vstack((src, dst)))

cv2.imwrite("D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/Defokus.png",dst)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
