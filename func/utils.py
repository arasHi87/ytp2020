import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import label

def GetConnectComponet(image):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10, 10), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    binary = cv2.bitwise_not(binary)
    labeled_img, num = label(binary, neighbors=4, background=0, return_num=True)
    
    max_label = 0
    max_num = 0

    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    
    lcc = (labeled_img == max_label)
    kernel = np.ones((15, 15), np.uint8)
    lcc = cv2.dilate(lcc.astype(np.uint8), kernel, iterations = 1)
   
    return lcc

def FillConnectComponet(image, lcc):
    if image.shape != lcc.shape:
        logger.error('Two images\' are not same')
        return
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if lcc[i, j]:
                image[i, j]=60
    
    return image
