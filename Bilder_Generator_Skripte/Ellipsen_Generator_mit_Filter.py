#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   2.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Ellipsen_Generator_mit_Filter.py

import matplotlib.patches
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


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    # print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):


    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])

    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = (a + (b - a) / X.max() * X)
    im = ax.imshow(X, extent=extent, interpolation='bicubic', vmin=0, vmax=1, **kwargs)
    return im


def ellipse_generator(nx, ny, num_min, num_max, ab_min, ab_max, b_min, b_max,
                      overlapp=False, abschneiden=False, schwarz=False, nbild=0):
    fig, ax = plt.subplots(figsize=(100, 50), dpi=10, subplot_kw={'aspect': 'equal'})

    cdict = {'red': ((0.0, 0.3, 0.3),
                     (0.25, 0.4, 0.4),
                     (0.5, 0.5, 0.5),
                     (0.75, 0.6, 0.6),
                     (1.0, 0.7, 0.7)),

             'green': ((0.0, 0.3, 0.3),
                       (0.25, 0.4, 0.4),
                       (0.5, 0.5, 0.5),
                       (0.75, 0.6, 0.6),
                       (1.0, 0.7, 0.7)),

             'blue': ((0.0, 0.3, 0.3),
                      (0.25, 0.4, 0.4),
                      (0.5, 0.5, 0.5),
                      (0.75, 0.6, 0.6),
                      (1.0, 0.7, 0.7))}

    cmap = LinearSegmentedColormap('Gradient_Grey_Scale', cdict)

    gradient_image(ax, direction=random.uniform(0, 1.3), extent=(-0.1, 1, -0.1, 1), transform=ax.transAxes,
                   cmap=cmap, cmap_range=(0, 1))



    used_point_list = []

    label = open("C:/Users/Wenda Gu/Desktop/rbox/Label/Bild_%d.tif.rbox" % (nbild + 1), 'w')

    durchmesser = []
    proportion = []

    x0_list = []
    y0_list = []
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    x3_list = []
    y3_list = []

    x_min = []
    x_max = []
    y_min = []
    y_max = []

    counter = np.random.randint(num_min, num_max)
    k = 0
    while k < counter:
        b = np.random.uniform(b_min, b_max)
        ab = np.random.uniform(ab_min, ab_max)
        a = ab * b
        d = np.sqrt(b * a)
        alpha = random.uniform(0.1, 0.2)


        ell = matplotlib.patches.Ellipse(xy=(np.random.uniform(0, nx), np.random.uniform(0, ny)),
                                         width=b, height=a, angle=np.random.rand() * 360, alpha=None,
                                         facecolor='%f'%alpha, edgecolor='none')

        ell_kontrolle = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                                   width=ell.width + 2, height=ell.height + 2, angle=ell.angle,
                                                   alpha=None,
                                                   facecolor='black', edgecolor='none')
        print('Add Ell_' + str(k + 1) + ' : ')

        # find the bounding box of the ellipse
        bb = ell_kontrolle.get_window_extent()
        bb_0 = ell.get_window_extent()

        # calculate the x and y points possibly within the ellipse
        x_int = np.arange(np.ceil(bb.x0), np.floor(bb.x1) + 1, dtype='int')
        y_int = np.arange(np.ceil(bb.y0), np.floor(bb.y1) + 1, dtype='int')

        # create a list of possible coordinates
        g = np.meshgrid(x_int, y_int)
        coords = list(zip(*(c.flat for c in g)))
        ellipsepoints = [q for q in coords if ell_kontrolle.contains_point(q, radius=0)]

        if overlapp is False:
            intersection = set(used_point_list).intersection(set(ellipsepoints))
            if len(intersection) == 0:

                if abschneiden is False:
                    if np.ceil(bb.x0) > 2 and np.floor(bb.x1) < nx - 2 and np.ceil(bb.y0) > 2 and np.floor(
                            bb.y1) < ny - 2:
                        ax.add_artist(ell)
                        used_point_list = used_point_list + ellipsepoints
                        durchmesser.append(d)
                        proportion.append(a / b)
                        #print('Durchmesser = ' + str(d))
                        print('not overlapp, not abschneiden')

                        x_min.append(np.floor(bb_0.x0))
                        y_min.append(np.floor(499-bb_0.y1))
                        x_max.append(np.ceil(bb_0.x1))
                        y_max.append(np.ceil(501-bb_0.y0))

                        x_center, y_center = ell.center
                        ell_angle = ell.angle * np.pi / 180
                        x_high = (ell.width * np.cos(ell_angle) / 2) + x_center
                        y_high = (ell.width * np.sin(ell_angle) / 2) + y_center
                        x0 = int(x_high - (ell.height * np.sin(ell_angle) / 2))
                        y0 = int((y_high + (ell.height * np.cos(ell_angle) / 2)))
                        x1 = int(x_high + (ell.height * np.sin(ell_angle) / 2))
                        y1 = int((y_high - (ell.height * np.cos(ell_angle) / 2)))
                        x2 = int(2 * x_center - x0)
                        y2 = int(2 * y_center - y0)
                        x3 = int(2 * x_center - x1)
                        y3 = int(2 * y_center - y1)
                        x0_list.append(x0)
                        y0_list.append(500 - y0)
                        x1_list.append(x1)
                        y1_list.append(500 - y1)
                        x2_list.append(x2)
                        y2_list.append(500 - y2)
                        x3_list.append(x3)
                        y3_list.append(500 - y3)

                        # plt.scatter([x1,x2,x3,x4],[y1,y2,y3,y4],s=400, c='red')
                        # plt.plot([x1,x2,x3,x4,x1],[y1,y2,y3,y4,y1],color='r', linewidth=10)

                        label.write(str(round(ell.center[0],4)) + " " +
                                    str(round(ell.center[1],4)) + " " +
                                    str(round(ell.width,4)) + " " +
                                    str(round(ell.height,4)) + " " +
                                    str(1) + " " +
                                    str(ell.angle))
                        label.write('\n')
                        k = k + 1
                else:
                    ax.add_artist(ell)
                    used_point_list = used_point_list + ellipsepoints
                    durchmesser.append(d)
                    proportion.append(a / b)
                    print('not overlapp, abschneiden')
                    x_min.append(np.floor(bb_0.x0))
                    y_min.append(np.floor(499 - bb_0.y1))
                    x_max.append(np.ceil(bb_0.x1))
                    y_max.append(np.ceil(501 - bb_0.y0))
                    label.write(str(ell.center[0]) + " " +
                                str(ell.center[1]) + " " +
                                str(ell.width) + " " +
                                str(ell.height) + " " +
                                str(1) + " " +
                                str(ell.angle))
                    label.write('\n')
                    k = k + 1
        else:
            if abschneiden is False:
                if np.ceil(bb.x0) > 2 and np.floor(bb.x1) < nx - 2 and np.ceil(bb.y0) > 2 and np.floor(bb.y1) < ny - 2:
                    ax.add_artist(ell)
                    used_point_list = used_point_list + ellipsepoints
                    durchmesser.append(d)
                    proportion.append(a / b)
                    print('overlapp, not abschneiden')
                    x_min.append(np.floor(bb_0.x0))
                    y_min.append(np.floor(499 - bb_0.y1))
                    x_max.append(np.ceil(bb_0.x1))
                    y_max.append(np.ceil(501 - bb_0.y0))
                    # label.write(str(ell.center[0]) + " " +
                    #             str(ell.center[1]) + " " +
                    #             str(ell.width) + " " +
                    #             str(ell.height) + " " +
                    #             str(1) + " " +
                    #             str(ell.angle))
                    # label.write('\n')
                    k = k + 1
            else:
                ax.add_artist(ell)
                used_point_list = used_point_list + ellipsepoints
                durchmesser.append(d)
                proportion.append(a / b)
                print('overlapp abschneiden')
                x_min.append(np.floor(bb_0.x0))
                y_min.append(np.floor(499 - bb_0.y1))
                x_max.append(np.ceil(bb_0.x1))
                y_max.append(np.ceil(501 - bb_0.y0))

                x_center, y_center = ell.center
                ell_angle = ell.angle * np.pi / 180
                x_high = (ell.width * np.cos(ell_angle) / 2) + x_center
                y_high = (ell.width * np.sin(ell_angle) / 2) + y_center
                x0 = int(x_high - (ell.height * np.sin(ell_angle) / 2))
                y0 = int((y_high + (ell.height * np.cos(ell_angle) / 2)))
                x1 = int(x_high + (ell.height * np.sin(ell_angle) / 2))
                y1 = int((y_high - (ell.height * np.cos(ell_angle) / 2)))
                x2 = int(2 * x_center - x0)
                y2 = int(2 * y_center - y0)
                x3 = int(2 * x_center - x1)
                y3 = int(2 * y_center - y1)
                x0_list.append(x0)
                y0_list.append(500 - y0)
                x1_list.append(x1)
                y1_list.append(500 - y1)
                x2_list.append(x2)
                y2_list.append(500 - y2)
                x3_list.append(x3)
                y3_list.append(500 - y3)
                # plt.scatter([x0,x1,x2,x3],[y0,y1,y2,y3],s=400, c='red')
                # plt.plot([x0,x1,x2,x3,x0],[y0,y1,y2,y3,y0],color='r', linewidth=10)
                # label.write(str(ell.center[0]) + " " +
                #             str(ell.center[1]) + " " +
                #             str(ell.width) + " " +
                #             str(ell.height) + " " +
                #             str(1) + " " +
                #             str(ell.angle))
                # label.write('\n')
                k = k + 1
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 500)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    dir1 = 'C:/Users/Wenda Gu/Desktop/RRCNN/VOC/VOCdevkit/VOC2007/tmp/'
    s = str(nbild+1)
    num = s.zfill(6)
    filename1 = dir1 + num + '.png'
    # filename = 'C:/Users/Wenda Gu/Desktop/VOC/VOCdevkit/VOC2007/JPEGImages/Bild_%d.jpeg' % (nbild+1)
    plt.savefig(filename1,
                transparent=False, bbox_inches='tight', pad_inches=0)
    # plt.savefig("C:/Users/Wenda Gu/Desktop/rbox/Image/Bild_%d.tif" % (nbild + 1),
    #             transparent=False, bbox_inches='tight', pad_inches=0)
    img_array = np.array(Image.open(filename1))
    print(img_array.shape)
    img_RGB = Image.fromarray(img_array.astype('uint8')).convert('RGB')

    # dir2 = 'D:/Dataset/YOLO_und_RRCNN/JPEGImages/'
    dir2 = 'D:/Dataset/YOLO_und_RRCNN/JPEGImages/'
    filename2 = dir2 + num + '.png'
    img_RGB.save(filename2)

    # Elastische Deformation
    # img = io.imread(filename2)
    # img_ela = Image.fromarray(elastic_transform(img, 991, 8))
    # img_ela.save(filename2)

    #Defokus
    # src = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)
    # dst = cv2.GaussianBlur(src, (9, 9), random.uniform(1, 2.5))
    # cv2.imwrite(filename2, dst)

    # Gaussian Noise
    img_gauss_tmp = skimage.io.imread(filename2)

    img_gauss = skimage.util.random_noise(img_gauss_tmp ,mode="gaussian", var=random.uniform(0, 0.01))
    io.imsave(filename2,img_gauss)



    print(np.array(Image.open(filename2)).shape)
    label.close()

    dom_RRCNN = xml.dom.minidom.parse("D:/Dataset/YOLO_und_RRCNN/100_RRCNN.xml")
    dom_YOLO = xml.dom.minidom.parse("D:/Dataset/YOLO_und_RRCNN/100_YOLO.xml")

    root_RRCNN = dom_RRCNN.documentElement
    root_YOLO = dom_YOLO.documentElement

    fn_RRCNN = root_RRCNN.getElementsByTagName('filename')
    fn_YOLO = root_YOLO.getElementsByTagName('filename')

    x0 = root_RRCNN.getElementsByTagName('x0')
    y0 = root_RRCNN.getElementsByTagName('y0')
    x1 = root_RRCNN.getElementsByTagName('x1')
    y1 = root_RRCNN.getElementsByTagName('y1')
    x2 = root_RRCNN.getElementsByTagName('x2')
    y2 = root_RRCNN.getElementsByTagName('y2')
    x3 = root_RRCNN.getElementsByTagName('x3')
    y3 = root_RRCNN.getElementsByTagName('y3')
    fn_RRCNN[0].firstChild.data = num + '.png'

    xmin = root_YOLO.getElementsByTagName('xmin')
    ymin = root_YOLO.getElementsByTagName('ymin')
    xmax = root_YOLO.getElementsByTagName('xmax')
    ymax = root_YOLO.getElementsByTagName('ymax')
    fn_YOLO[0].firstChild.data = num + '.png'

    for n in range(num_min):
        x0[n].firstChild.data = x0_list[n]
        y0[n].firstChild.data = y0_list[n]
        x1[n].firstChild.data = x1_list[n]
        y1[n].firstChild.data = y1_list[n]
        x2[n].firstChild.data = x2_list[n]
        y2[n].firstChild.data = y2_list[n]
        x3[n].firstChild.data = x3_list[n]
        y3[n].firstChild.data = y3_list[n]

        xmin[n].firstChild.data = x_min[n]
        ymin[n].firstChild.data = y_min[n]
        xmax[n].firstChild.data = x_max[n]
        ymax[n].firstChild.data = y_max[n]


    filename_RRCNN = "D:/Dataset/YOLO_und_RRCNN/RRCNN/VOCdevkit/VOCdevkit_train/Annotation/" + num + '.xml'
    filename_YOLO = "D:/Dataset/YOLO_und_RRCNN/YOLO/VOC/VOCdevkit/VOC2007/Annotations/" + num + '.xml'
    # filename_RRCNN = "D:/Dataset/YOLO_und_RRCNN/RRCNN/VOCdevkit/VOCdeckit_train/Annotation/" + num + '.xml'
    # filename_YOLO = "D:/Dataset/YOLO_und_RRCNN/YOLO/VOC/VOCdevkit/VOC2007/Annotations/" + num + '.xml'
    with open(filename_RRCNN, "w") as fh_RRCNN:
        dom_RRCNN.writexml(fh_RRCNN)
    with open(filename_YOLO, "w") as fh_YOLO:
        dom_YOLO.writexml(fh_YOLO)


if __name__ == '__main__':
    for i in range(6500, 7000):
        ellipse_generator(1000, 500, 100, 101, 0.1, 1.0, 10, 100,
                          overlapp=False, abschneiden=False, schwarz=False, nbild=i)
