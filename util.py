import os

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
def binary_mask(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 255 if img[i, j] > 40 else 0
    return img

def img_show(img):

    plt.imshow(img,'gray')
    plt.show()


def laplacian_edge(img):
    # 阈值处理
    ret, binary = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    # Laplacian算子
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(dst)
    return laplacian
def log(img):
    print(img.shape,img.dtype,np.max(img),np.min(img))

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imgread(img_path):
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def imsave(img,img_path):
    cv2.imwrite(img_path,img)