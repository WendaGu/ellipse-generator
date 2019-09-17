#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.1
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   ContainPoint_berechnen.py


import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np


# create an ellipse
el_0 = matplotlib.patches.Ellipse(xy=(np.random.uniform(0, 100), np.random.uniform(0, 50)),
                    width=np.random.rand() * 30,
                    height=np.random.rand() * 15,
                    angle=np.random.rand() * 360, alpha=None, facecolor='black', edgecolor='none')

# calculate the x and y points possibly within the ellipse
x_int = np.arange(0, 100)
y_int = np.arange(0, 50)

# create a list of possible coordinates
g = np.meshgrid(x_int, y_int)  # Liste in Matrix
coords = list(zip(*(c.flat for c in g)))  # Eine eindimensionale Liste ausgeben
# create the list of valid coordinates (from untransformed) 
used_point_list = np.vstack([p for p in coords if el_0.contains_point(p, radius=0)]).tolist()

fig, ax = plt.subplots(figsize=(10, 5), dpi=50, subplot_kw={'aspect': 'equal'})
ax.add_artist(el_0)

ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

# Entfernung der Bildachse und der umgebenden weißen Ränder
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

# generate new random ellipse
counter = 30
k = 1
while k < counter:
    el_k = matplotlib.patches.Ellipse(xy=(np.random.uniform(0, 100), np.random.uniform(0, 50)),
                                      width=np.random.rand() * 30,
                                      height=np.random.rand() * 15,
                                      angle=np.random.rand() * 360, alpha=None, facecolor='black', edgecolor='none')
    # calculate the x and y points possibly within the ellipse
    x_int = np.arange(0, 100)
    y_int = np.arange(0, 50)

    # create a list of possible coordinates
    g = np.meshgrid(x_int, y_int) 
    coords = list(zip(*(c.flat for c in g)))  

    # create the list of valid coordinates (from untransformed) 
    ellipsepoints = np.vstack([p for p in coords if el_k.contains_point(p, radius=0)]).tolist()

    # ell_0 = [ell[0] for ell in ellipsepoints]
    # ell_1 = [ell[1] for ell in ellipsepoints]
    # if 0 in ell_0 or 89 in ell_0:
    #     print('Ausserhalb der Grenzen')
    # elif 0 in ell_1 or 39 in ell_1:
    #     print('Ausserhalb der Grenzen')

    intersection = []
    for point in used_point_list:
        if point in ellipsepoints:
            intersection.append(point)

    if len(intersection) == 0:
        ax.add_artist(el_k)
        used_point_list = used_point_list + ellipsepoints
        k = k + 1

ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

# Entfernung der Bildachse und der umgebenden weißen Ränder
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.plot()
plt.show()

