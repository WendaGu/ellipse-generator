#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.2
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Bilder_vermehren.py

import matplotlib.patches
import os,sys,shutil
import xml.dom.minidom
import skimage
from skimage import img_as_ubyte
import random
from random import choice
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io

def GaussieNoisy(image,sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype(np.uint8)



def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):


    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])

    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = (a + (b - a) / X.max() * X)
    im = ax.imshow(X, extent=extent, interpolation='bicubic', vmin=0, vmax=1, **kwargs)
    return im


path = 'D:/Dataset/image_plus_augmented_images_YOLO_format/'


# Hintergrund Gradient
for filename_1 in os.listdir(path):
    file_name, file_extend = os.path.splitext(filename_1)
    if file_extend == '.png':
        old_name = file_name + '.txt'
        file_path = os.path.join(path, old_name)
        print(file_name)
        random_1 = np.random.uniform(0, 10)
        if random_1 <= 8:


            fig, ax = plt.subplots(figsize=(160, 120), dpi=10, subplot_kw={'aspect': 'equal'})

            cdict = {'red': ((0.0, 0.1, 0.1),
                             (0.25, 0.2, 0.2),
                             (0.5, 0.3, 0.3),
                             (0.75, 0.8, 0.8),
                             (1.0, 0.9, 0.9)),

                     'green':((0.0, 0.1, 0.1),
                             (0.25, 0.2, 0.2),
                             (0.5, 0.3, 0.3),
                             (0.75, 0.8, 0.8),
                             (1.0, 0.9, 0.9)),

                     'blue': ((0.0, 0.1, 0.1),
                             (0.25, 0.2, 0.2),
                             (0.5, 0.3, 0.3),
                             (0.75, 0.8, 0.8),
                             (1.0, 0.9, 0.9))}

            cmap = LinearSegmentedColormap('Gradient_Grey_Scale', cdict)
            gradient_image(ax, direction=random.uniform(0, 1.3), extent=(-0.1, 1, -0.1, 1), transform=ax.transAxes,
                           cmap=cmap, cmap_range=(0, 1))
            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 1200)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(path + 'tmp.png', transparent=False, bbox_inches='tight', pad_inches=0)

            # 2 Bilder überlappen
            img1 = Image.open(path + filename_1)
            img1 = img1.convert('RGBA')

            img2 = Image.open(path + 'tmp.png')
            img2 = img2.convert('RGBA')

            img = Image.blend(img1, img2, 0.3)
            # img.show()
            filename_1 = filename_1.split('.')[0]+'_Gradient.png'
            img.save(path+filename_1)
            new_name = filename_1.split('.')[0] + '.txt'
            newfile_path = os.path.join(path, new_name)
            shutil.copyfile(file_path, newfile_path)
            os.remove(path+'tmp.png')

    # print(np.array(Image.open(path + filename)).shape)

# Defokus, Blur

        random_2 = np.random.uniform(0, 10)
        if random_2 <= 6:
            # filename_defokus = filename_1.split('.')[0] + '_Gradient.png' + '_Defokus.png'
            src = cv2.imread(path + filename_1, cv2.IMREAD_UNCHANGED)
            dst = cv2.GaussianBlur(src, (9, 9), random.uniform(2.0, 2.5))
            if '_Gradient' in filename_1.split('.')[0]:
                os.remove(path + filename_1)
                os.remove(path + filename_1.split('.')[0] + '.txt')

                filename_1 = filename_1.split('.')[0] + '_Defokus.png'
                cv2.imwrite(path+filename_1, dst)
                new_name = filename_1.split('.')[0] + '.txt'
                newfile_path = os.path.join(path, new_name)
                shutil.copyfile(file_path, newfile_path)
            else:
                filename_1 = filename_1.split('.')[0] + '_Defokus.png'
                cv2.imwrite(path + filename_1, dst)
                new_name = filename_1.split('.')[0] + '.txt'
                newfile_path = os.path.join(path, new_name)
                shutil.copyfile(file_path, newfile_path)
    #Gaussien Filter

        random_3 = np.random.uniform(0, 10)
        if random_3 <= 8:
            img_gauss_tmp = skimage.io.imread(path+filename_1)

            img_gauss = skimage.util.random_noise(img_gauss_tmp, mode="gaussian", var=random.uniform(0.008, 0.015))
            if '_Defokus' in filename_1.split('.')[0] or '_Gradient' in filename_1.split('.')[0]:
                os.remove(path+filename_1)
                os.remove(path+filename_1.split('.')[0] + '.txt')
                filename_1 = filename_1.split('.')[0] + '_Gauss.png'

                io.imsave(path+filename_1, img_gauss)
                new_name = filename_1.split('.')[0] + '.txt'

                newfile_path = os.path.join(path, new_name)
                shutil.copyfile(file_path, newfile_path)
            # elif '_Gradient' in filename_1.split('.')[0]:
            #     os.remove(path + filename_1)
            #     os.remove(path + filename_1.split('.')[0] + '.txt')
            #     filename_1 = filename_1.split('.')[0] + '_Gauss.png'
            #
            #     io.imsave(path + filename_1, img_gauss)
            #     new_name = filename_1.split('.')[0] + '.txt'
            #
            #     newfile_path = os.path.join(path, new_name)
            #     shutil.copyfile(file_path, newfile_path)
            #
            else:
                filename_1 = filename_1.split('.')[0] + '_Gauss.png'

                io.imsave(path + filename_1, img_gauss)
                new_name = filename_1.split('.')[0] + '.txt'

                newfile_path = os.path.join(path, new_name)
                shutil.copyfile(file_path, newfile_path)
