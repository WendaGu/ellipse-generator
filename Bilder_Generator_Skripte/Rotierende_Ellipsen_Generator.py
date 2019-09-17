#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Rotierende_Ellipsen_Generator.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches
from PIL import Image
import time
import xml.dom.minidom
import random
from random import choice
import cv2


def ellipse_generator(nx, ny, num_min, num_max, ab_min, ab_max, b_min, b_max,
                      overlapp=False, abschneiden=False, schwarz=False, nbild=0):
    fig, ax = plt.subplots(figsize=(100, 50), dpi=10, subplot_kw={'aspect': 'equal'})
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
    winkel = [1,2]
    angle_1= choice(winkel) * 90
    angle_2 = np.random.rand() * 360
    counter = np.random.randint(num_min, num_max)
    # counter = 10
    k = 0
    while k < counter:
        b = np.random.uniform(b_min, b_max)
        ab = np.random.uniform(ab_min, ab_max)
        a = ab * b
        d = np.sqrt(b * a)

        ell = matplotlib.patches.Ellipse(xy=(np.random.uniform(0, nx), np.random.uniform(0, ny)),
                                         width=b, height=a, angle=np.random.rand() * 360, alpha=None,
                                         facecolor='black', edgecolor='none')

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
                        y_min.append(np.floor(299-bb_0.y1))
                        x_max.append(np.ceil(bb_0.x1))
                        y_max.append(np.ceil(301-bb_0.y0))

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

                        # print(x0, 500-y0)
                        # print(x1, 500-y1)
                        # print(x2, 500-y2)
                        # print(x3, 500-y3)
                        # plt.scatter([x1,x2,x3,x4],[y1,y2,y3,y4],s=400, c='red')
                        # plt.plot([x1,x2,x3,x4,x1],[y1,y2,y3,y4,y1],color='r', linewidth=10)
                        # print(ell.center)
                        # print(ell.center[0])
                        # print(ell.center[1])
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
                    y_min.append(np.floor(299 - bb_0.y1))
                    x_max.append(np.ceil(bb_0.x1))
                    y_max.append(np.ceil(301 - bb_0.y0))
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
                    y_min.append(np.floor(299 - bb_0.y1))
                    x_max.append(np.ceil(bb_0.x1))
                    y_max.append(np.ceil(301 - bb_0.y0))
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
                y_min.append(np.floor(299 - bb_0.y1))
                x_max.append(np.ceil(bb_0.x1))
                y_max.append(np.ceil(301 - bb_0.y0))

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
    img_bi = Image.fromarray(img_array.astype('uint8')).convert('RGB')

    # dir2 = 'D:/Dataset/YOLO_und_RRCNN/RRCNN/VOCdevkit/VOCdeckit_test/JPEGImages/'
    dir2 = 'D:/Dataset/YOLO_und_RRCNN/RRCNN/Test/YOLO/JPEGImages/'
    filename2 = dir2 + num + '.png'
    img_bi.save(filename2)
    print(np.array(Image.open(filename2)).shape)


    # img = cv2.imread("C:/Users/Wenda Gu/Desktop/VOC2007/JPEGImages/Bild_1.jpeg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # plt.plot()
    # plt.show()
    label.close()
    if schwarz is True:
        plt.savefig(filename1, transparent=False, bbox_inches='tight', pad_inches=0)
        img_array = np.array(Image.open(filename1))
        img_bi = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        img_bi.save(filename1)
        img_bi.show()

    # # Histogram von Durchmesser
    # plt.hist(durchmesser, bins=np.arange(0, 101, 1), facecolor="blue", edgecolor="black", alpha=0.7,
    #          label='Total Anzahl von Ellipse = ' + str(counter))
    #
    # (durchmesser_counts, durchmesser_bins) = np.histogram(durchmesser, bins=np.arange(0, 101, 1))
    #
    # # np.savez("C:/Users/Wenda Gu/Desktop/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    # #          durchmesser_counts=durchmesser_counts, durchmesser_bins=durchmesser_bins)
    # # np.savetxt("C:/Users/Wenda Gu/Desktop/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    # #            durchmesser, fmt='%10.5f', delimiter='\t')
    #
    # plt.xlim(0, 100)
    # plt.xticks(np.arange(0, 101, 5))
    # plt.xlabel("Durchmesser")
    # plt.ylabel("Anzahl von Ellipse")
    # plt.title("Histogram von Durchmesser")
    # plt.legend(loc='best')
    # # plt.show()
    #
    # # Histogram von Proportion
    # plt.hist(proportion, bins=np.arange(0, 1.01, 0.01), facecolor="blue", edgecolor="black", alpha=0.7,
    #          label='Total Anzahl von Ellipse = ' + str(counter))
    #
    # (proportion_counts, proportion_bins) = np.histogram(proportion, bins=np.arange(0, 1.01, 0.01))
    #
    # np.savez("C:/Users/Wenda Gu/Desktop/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    #          durchmesser_counts=durchmesser_counts, durchmesser_bins=durchmesser_bins)
    #
    # np.savetxt("C:/Users/Wenda Gu/Desktop/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    #            durchmesser, fmt='%10.5f', delimiter='\t')
    #
    # plt.xlim(0, 1)
    # plt.xticks(np.arange(0, 1.01, 0.1))
    # plt.xlabel("Proportion")
    # plt.ylabel("Anzahl von Ellipse")
    # plt.title("Histogram von Proportion")
    # plt.legend(loc='best')
    # plt.plot()
    # plt.show()

    # print(x_min)
    # print(y_min)
    # print(x_max)
    # print(y_max)
    dom = xml.dom.minidom.parse("D:/Dataset/YOLO_und_RRCNN/50_RRCNN.xml")
    root = dom.documentElement

    fn = root.getElementsByTagName('filename')
    # path = root.getElementsByTagName('path')
    # print(path)
    x0 = root.getElementsByTagName('x0')
    y0 = root.getElementsByTagName('y0')
    x1 = root.getElementsByTagName('x1')
    y1 = root.getElementsByTagName('y1')
    x2 = root.getElementsByTagName('x2')
    y2 = root.getElementsByTagName('y2')
    x3 = root.getElementsByTagName('x3')
    y3 = root.getElementsByTagName('y3')
    fn[0].firstChild.data = num + '.png'
    # path[0].firstChild.data = filename2
    for n in range(num_min):
        x0[n].firstChild.data = x0_list[n]
        y0[n].firstChild.data = y0_list[n]
        x1[n].firstChild.data = x1_list[n]
        y1[n].firstChild.data = y1_list[n]
        x2[n].firstChild.data = x2_list[n]
        y2[n].firstChild.data = y2_list[n]
        x3[n].firstChild.data = x3_list[n]
        y3[n].firstChild.data = y3_list[n]
    filename3 = "D:/Dataset/YOLO_und_RRCNN/Test/RRCNN/Annotation/" + num + '.xml'
    # filename3 = "C:/Users/Wenda Gu/Desktop/RRCNN/VOC/VOCdevkit/test/Annotations/" + num + '.xml'
    with open(filename3, "w") as fh:
        dom.writexml(fh)


if __name__ == '__main__':
    time1 = time.time()
    for i in range(0, 10):
        ellipse_generator(1000, 500, 50, 51, 0.1, 1.0, 10, 100,
                          overlapp=True, abschneiden=True, schwarz=False, nbild=i)
    time2 = time.time()
    #print('Zeit = ' + str(time2 - time1) + ' Sekunden')

# para
#
#
#
#
#
#
#
#
#
#
#lle CNN tensorflow
