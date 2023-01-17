import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import detect_image_v7

image = '../Examples/asd.jpeg'

#boxes = detect_image_v7.getBoxesFromImg(image)
#boxes1 = detect_image_v7.getBoxesFromImg('../Examples/bus.jpg')

#print("Boxes got: ", boxes)

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read(-1)

    cv2.imshow("asd", frame)

    cv2.waitKey(30)