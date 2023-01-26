import argparse
import time
from pathlib import Path

import numpy as np
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

class BoundingBox:
    (x0, x1, y0, y1) = (0, 0, 0, 0)
    confidence: float
    cls: int
    def __init__(self, x0, x1, y0, y1, confidence, cls):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.confidence = confidence
        self.cls = cls

weights_file = "../yolov7-w6.pt"
#image= "inference/images/image1.jpg"
source = "0"
source_size = 640

camera = False
if source == '0':
    print("Using camera output")
    camera = True 

# Inıt device
set_logging()
device = select_device('')
half = device.type != 'cpu'

# Load model and check image size
model = attempt_load(weights_file, map_location=device)
stride = int(model.stride.max())
source_size = check_img_size(source_size, s=stride)

# TODO: Add trace arg

# Half precision for CUDA
if half:
    model.half()

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#print('Names', names)
#print('Colors', colors)

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, source_size, source_size).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = source_size
old_img_b = 1

def getBoxesFromCvImage(img0: np.ndarray):
    img = letterbox(img0, source_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    old_img_w = old_img_h = source_size
    old_img_b = 1

    img = torch.from_numpy(img).to(device)

    if half:
        img = img.half()
    else:
        img = img.float()
    
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=0)[0]
    
    # Inference
    with torch.no_grad():
        pred = model(img, augment=0)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=None)

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
    

    boxes = []
    # Process detections
    for i,det in enumerate(pred):
        #p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        im0 = img0

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            #    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                #if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #line = (cls, *xywh)  # label format
                #with open(save_path + '.txt', 'a') as f:
                #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                #if save_img or view_img:  # Add bbox to image
                #label = f'{names[int(cls)]} {conf:.2f}'
                #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                x0 = xyxy[0]
                y0 = xyxy[1]
                x1 = xyxy[2]
                y1 = xyxy[3]
                print(f'Class {cls} Conf {conf:.2f} x0 {x0} y0 {y0} x1 {x1} y1 {y1}')
                b = BoundingBox(x0, x1, y0, y1, conf, cls)
                boxes.append(b)

                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    return im0, boxes

def getBoxesFromImg(image):
    print("getBoxesFromImg")

    #dataset = LoadImages(image, img_size=source_size, stride=stride)
    dataset = image
    old_img_w = old_img_h = source_size
    old_img_b = 1
    # Main thing ig
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)

        if half:
            img = img.half()
        else:
            img = img.float()
        
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=0)[0]
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=0)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=None)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        

        boxes = []
        # Process detections
        for i,det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #line = (cls, *xywh)  # label format
                    #with open(save_path + '.txt', 'a') as f:
                    #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #if save_img or view_img:  # Add bbox to image
                    #label = f'{names[int(cls)]} {conf:.2f}'
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    x0 = xyxy[0]
                    y0 = xyxy[1]
                    x1 = xyxy[2]
                    y1 = xyxy[3]
                    print(f'Class {cls} Conf {conf:.2f} x0 {x0} y0 {y0} x1 {x1} y1 {y1}')

                    b = BoundingBox(x0, x1, y0, y1, conf, cls)
                    boxes.append(b)
        return boxes

if __name__ == "__main__":
    dataset = 0
    if camera:
        dataset = LoadStreams("0", img_size=source_size, stride=stride)
    else:
        dataset = LoadImages(source, img_size=source_size, stride=stride)
    print(dataset)

    # Main thing ig
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)

        if half:
            img = img.half()
        else:
            img = img.float()
        
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        #if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #    old_img_b = img.shape[0]
        #    old_img_h = img.shape[2]
        #    old_img_w = img.shape[3]
        #    for i in range(3):
        #        model(img, augment=0)[0]
        
        # Inference
        with torch.no_grad():
            pred = model(img, augment=0)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=None)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i,det in enumerate(pred):
            if camera:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            print('p', p)
            save_path = str(p.name)

            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #print(gn)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #line = (cls, *xywh)  # label format

                    #if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    x0 = xyxy[0]
                    y0 = xyxy[1]
                    x1 = xyxy[2]
                    y1 = xyxy[3]

                    with open(save_path + '.txt', 'a') as f:
                        f.write(f'Class {cls} Conf {conf:.2f} x0 {x0} y0 {y0} x1 {x1} y1 {y1}\n')

                    print(f'Class {cls} Conf {conf:.2f} x0 {x0} y0 {y0} x1 {x1} y1 {y1}')

            if camera:
                cv2.imshow("Webcam", im0)
            else:
                cv2.imwrite(save_path, im0)
