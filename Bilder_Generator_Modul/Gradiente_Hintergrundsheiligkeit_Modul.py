#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Gradiente_Hintergrundsheiligkeit_Modul.py


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches
from PIL import Image
import time
from matplotlib.colors import LinearSegmentedColormap
import random


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
    fig, ax = plt.subplots(figsize=(nx / 10, ny / 10), dpi=10, subplot_kw={'aspect': 'equal'})

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

    # im = [np.linspace(0, 1, 1600)] * 1600
    # fig.figimage(im, cmap=cmap)

    used_point_list = []

    durchmesser = []
    proportion = []

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

        ell_1 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                                   width=ell.width + 2, height=ell.height + 2, angle=ell.angle,
                                                   alpha=None,
                                                   facecolor='0.12', edgecolor='none')
        ell_2 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 4, height=ell.height + 4, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.14', edgecolor='none')
        ell_3 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 6, height=ell.height + 6, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.16', edgecolor='none')
        ell_4 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 8, height=ell.height + 8, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.18', edgecolor='none')
        ell_5 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 10, height=ell.height + 10, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.20', edgecolor='none')
        ell_6 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 12, height=ell.height + 12, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.22', edgecolor='none')
        ell_7 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 14, height=ell.height + 14, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.24', edgecolor='none')
        ell_8 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 16, height=ell.height + 16, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.26', edgecolor='none')
        ell_9 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 18, height=ell.height + 18, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.28', edgecolor='none')
        ell_10 = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                           width=ell.width + 20, height=ell.height + 20, angle=ell.angle,
                                           alpha=None,
                                           facecolor='0.30', edgecolor='none')

        ell_kontrolle = matplotlib.patches.Ellipse(xy=(ell.center[0], ell.center[1]),
                                                   width=ell.width + 2, height=ell.height + 2, angle=ell.angle,
                                                   alpha=None,
                                                   facecolor='0.2', edgecolor='none')
        print('Add Ell_' + str(k + 1) + ' : ')

        # find the bounding box of the ellipse
        bb = ell_kontrolle.get_window_extent()

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
                        print('Durchmesser = ' + str(d))
                        print('not overlapp, not abschneiden')
                        k = k + 1
                else:
                    ax.add_artist(ell)
                    used_point_list = used_point_list + ellipsepoints
                    durchmesser.append(d)
                    proportion.append(a / b)
                    print('not overlapp, abschneiden')
                    k = k + 1
        else:
            if abschneiden is False:
                if np.ceil(bb.x0) > 2 and np.floor(bb.x1) < nx - 2 and np.ceil(bb.y0) > 2 and np.floor(bb.y1) < ny - 2:
                    ax.add_artist(ell)
                    used_point_list = used_point_list + ellipsepoints
                    durchmesser.append(d)
                    proportion.append(a / b)
                    print('overlapp, not abschneiden')
                    k = k + 1
            else:
                # ax.add_artist(ell_10)
                # ax.add_artist(ell_9)
                # ax.add_artist(ell_8)
                # ax.add_artist(ell_7)
                # ax.add_artist(ell_6)
                # ax.add_artist(ell_5)
                # ax.add_artist(ell_4)
                # ax.add_artist(ell_3)
                # ax.add_artist(ell_2)
                # ax.add_artist(ell_1)
                ax.add_artist(ell)
                used_point_list = used_point_list + ellipsepoints
                durchmesser.append(d)
                proportion.append(a / b)
                print('overlapp abschneiden')
                k = k + 1


    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # filename = 'test.png'
    # filename = 'Bild%d.png' % (nbild+1)
    plt.savefig("D:/Dataset/RRCNN/VOCdevkit/VOCdeckit_train/test.png",
                transparent=False, bbox_inches='tight', pad_inches=0)

    img_path = "/Users/GUWENDA/Documents/Bild/Bild_%d.png" % (nbild + 1)

    # img = skimage.io.imread(img_path)
    # gimg = skimage.util.random_noise(img, mode="gaussian")
    plt.plot()
    # plt.imshow(gimg)
    plt.show()

    if schwarz is True:
        plt.savefig(filename, transparent=False, bbox_inches='tight', pad_inches=0)
        img_array = np.array(Image.open(filename))
        img_bi = Image.fromarray(img_array.astype('uint8')).convert('1')
        img_bi.save(filename)
        img_bi.show()

    # Histogram von Durchmesser
    plt.hist(durchmesser, bins=np.arange(0, 101, 1), facecolor="blue", edgecolor="black", alpha=0.7,
             label='Total Anzahl von Ellipse = ' + str(counter))

    (durchmesser_counts, durchmesser_bins) = np.histogram(durchmesser, bins=np.arange(0, 101, 1))
    #
    # np.savez("/Users/GUWENDA/Documents/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    #          durchmesser_counts=durchmesser_counts, durchmesser_bins=durchmesser_bins)
    # np.savetxt("/Users/GUWENDA/Documents/Histogramm_Durchmesser/Durchmesser_%d" % (nbild + 1),
    #            durchmesser, fmt='%10.5f', delimiter='\t')

    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 5))
    plt.xlabel("Durchmesser")
    plt.ylabel("Anzahl von Ellipse")
    plt.title("Histogram von Durchmesser")
    plt.legend(loc='best')
    # plt.show()

    # Histogram von Proportion
    plt.hist(proportion, bins=np.arange(0, 1.01, 0.01), facecolor="blue", edgecolor="black", alpha=0.7,
             label='Total Anzahl von Ellipse = ' + str(counter))

    (proportion_counts, proportion_bins) = np.histogram(proportion, bins=np.arange(0, 1.01, 0.01))

    # np.savez("/Users/GUWENDA/Documents/Histogramm_Proportion/Proportion_%d" % (nbild+1),
    #          proportion_counts=proportion_counts, proportion_bins=proportion_bins)
    #
    # np.savetxt("/Users/GUWENDA/Documents/Histogramm_Proportion/Proportion_%d" % (nbild + 1),
    #            proportion, fmt='%10.5f', delimiter='\t')

    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.xlabel("Proportion")
    plt.ylabel("Anzahl von Ellipse")
    plt.title("Histogram von Proportion")
    plt.legend(loc='best')
    # plt.show()



if __name__ == '__main__':
    time1 = time.time()
    for i in range(0, 1):
        ellipse_generator(1000, 500, 20, 21, 0.1, 1.0, 10, 100,
                          overlapp=False, abschneiden=False, schwarz=False, nbild=i)


    time2 = time.time()
    print('Zeit = ' + str(time2 - time1) + ' Sekunden')
