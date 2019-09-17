#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Gaussian_Noise_Modul.py

import skimage
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches
from PIL import Image
import time
import xml.dom.minidom
import random
from random import choice
import cv2

img_path = "D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/JPEGImages/001371.png"
img = skimage.io.imread(img_path)
# img = cv2.imread('D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/JPEGImages/001371.png', cv2.IMREAD_UNCHANGED)


fig, ax = plt.subplots(figsize=(100, 50), dpi=10, subplot_kw={'aspect': 'equal'})
gimg = skimage.util.random_noise(img, mode="gaussian")

plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig("D:/Dataset/RRCNN/VOCdevkit/Gaussian_Noise.png",
                transparent=False, bbox_inches='tight', pad_inches=0)
# cv2.imwrite("D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/Gaussian_Noise.png",gimg)

# plt.savefig("D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/Gaussian_Noise.png")

plt.plot()
plt.imshow(gimg)
plt.show()
