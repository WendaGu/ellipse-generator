#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Create_Trainval.py

import os


path = '/home/guwenda/data/Image'
dir = os.listdir(path)
fopen = open('trainval.txt', 'w')
for d in dir:
    string = d + ' ' + d + '.rbox' + '\n'
    print(string)
    fopen.write(string)
fopen.close()
