import cv2
import numpy as np
# img =cv2.imread("clf-data\empty\\00000000_00000161.jpg")
# print(img.shape)
import torch
a = np.array([1,2,3,4,5])
# print(a>3)
# print(a.shape)
# print(np.nonzero(a>3)[0])
# print(len(np.nonzero(a>3)))

import random
a =[1,2,3]
b =[4,5,8]

c = list(zip(a,b))
random.shuffle(c)
a,b = zip(*c)
print(a)
print(b)