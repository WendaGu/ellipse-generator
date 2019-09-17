#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   DRBox_Rename.py


import os

path = '/home/guwenda/Dokumente/Label'
files = os.listdir(path)
# print('files',files)
for filename in files:
    portion = os.path.splitext(filename)

    if portion[1] == ".rbox":


        newname = portion[0] + ".tif.rbox"
        os.rename(filename, newname)
