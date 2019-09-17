#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# @Version :   1.0
# @Author  :   WendaGu
# @Software:   Pycharm
# @File    :   Get_Pixel_RGB.py


from PIL import Image

i = 1
j = 1
img = Image.open("C:/Users/Wenda Gu/Desktop/Grey_Scale_Image.png")
print(img.size)
print(img.getpixel((4,4)))

width = img.size[0]
height = img.size[1]
for i in range(0,width):
    for j in range(0,height):
        data = (img.getpixel((i,j)))
        print(data)
        print(data[0])
        if (data[0]>=170 and data[1]>=170 and data[2]>=170):
            img.putpixel((i,j),(234,53,57,255))
img = img.convert("RGB")
#img.save("C:/Users/Wenda Gu/Desktop/test1.jpg")


fig,ax = plt.subplot()
ax.set_facecolor('0.2')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 500)

plt.plot()
plt.show()
plt.savefig('C:/Users/Wenda Gu/Desktop/Grey_Scale_Image.png', transparent=False, bbox_inches='tight', pad_inches=0)



