#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Abgeschnittene_Ellipsen_Modul.py





import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

for ibild in range(1):
    NUM = 2
    ells = [Ellipse(xy=(np.random.randint(0, 100), np.random.randint(0, 50)),
                    width=np.random.rand() * 30,
                    height=np.random.rand() * 15,
                    angle=np.random.rand() * 360, facecolor='black', alpha=None)
            for i in range(NUM)]

    #print(a)
    # gleiche Skalierung von Daten zu Diagrammeinheiten für X und Y
    fig, ax = plt.subplots(figsize=(10, 5), dpi=10, subplot_kw={'aspect': 'equal'})

    for e in ells:

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e_array = np.array(plt.imshow)
        print(e_array)



        square = np.pi * e.height * e.width / 4
        #print(square)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    filename = 'Bild_%i.png' % ibild
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # fig.savefig(filename, transparent=False, dpi=10, pad_inches=0)
    plt.savefig(filename, transparent=False, bbox_inches='tight', pad_inches=0)
    plt.show()
    img_array = np.array(Image.open(filename))
    im_Bi = Image.fromarray(img_array.astype('uint8')).convert('1')
    #im_Bi.show()
    im_Bi.save("Bild_Bi_%i.png" % ibild)
    im_Bi_array = np.array(im_Bi)
    a0 = im_Bi_array[0]
    a1 = im_Bi_array[-1]
    b0 = im_Bi_array[:, 0]
    b1 = im_Bi_array[:, -1]
    #print(im_Bi_array.shape)
    if a0.all() == 1 and a1.all() == 1 and b0.all() == 1 and b1.all() == 1:
        im_Bi.show()



