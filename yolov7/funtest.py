import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from numpy import random
import threading

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import detect_image_v7

# w, h
cam_size = (1000, 600)

tilki = cv2.imread("../tilki.png")

image = cv2.imread("../Examples/asd.jpeg")
detectedImage = cv2.imread("../Examples/asd.jpeg")
boxes = 0
switch = False

def detectThread():
    global image
    global boxes
    global detectedImage
    global switch

    while True:
        detectedImage, boxes = detect_image_v7.getBoxesFromCvImage(image)
        switch = not switch

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])
ret, frame = cam.read(-1)
image = frame
cam_size = (image.shape[1], image.shape[0])

thr = threading.Thread(target=detectThread)
thr.start()

def drawGUI(image):
    posX = int(cam_size[1]* 1/7)
    X1 = int(cam_size[0] * 1/7)
    X2 = int(cam_size[0] * 6/7)
    Y1 = int(cam_size[1] * 1/7)
    Y2 = int(cam_size[1] * 6/7)
    color = (39,237,250)

    cv2.line(image, (X1,Y1),(X1,Y2),color,2)
    cv2.line(image, (X2,Y1),(X2,Y2),color,2)
    #cv2.line(image, (540,posX+240), (640,posX+240),color,1)
    #cv2.line(image, (0,posX+240), (100,posX+240),color,1)
    times = int((Y2 - Y1) / 17)
    for i in range(17):
        if i%3 == 1:
            cv2.line(image, (X1,posX+i*times), (X1+30,posX+i*times),color,3)
            cv2.line(image, (X2,posX+i*times), (X2-30,posX+i*times),color,3)
        else:
            cv2.line(image, (X1,posX+i*times), (X1+20,posX+i*times),color,2)
            cv2.line(image, (X2,posX+i*times), (X2-20,posX+i*times),color,2) 

    width = 120
    height = 160
    ratio = 120/160
    height = cam_size[1] / 5
    width = int(ratio * height)
    height = int(height)

    img_slice = image[0:0 + height, 0:0 + width]
    logo_slice = cv2.resize(tilki, (width, height), interpolation=cv2.INTER_NEAREST)
    added_image = cv2.addWeighted(img_slice, 0.5, logo_slice, 1-0.5, 0)
    image[0:0 + height, 0:0 + width] = added_image

    return image

last = switch
while True:
    ret, frame = cam.read(-1)

    image = frame

    #image, boxes = detect_image_v7.getBoxesFromCvImage(frame)
    if(last != switch):
        #cv2.imshow("MTRC Ground Control", drawGUI(detectedImage))
        #print("--")
        #cv2.waitKey(10)
        pass
    last = switch
    #for i in detectedImage.shape:
    #    print (i)
    #print(boxes)
