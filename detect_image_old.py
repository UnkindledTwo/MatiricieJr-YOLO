import cv2
import numpy as np

#Change these
modelFileName = "/home/unkindled/Desktop/Teknofest/MatiricieJr-YOLO/yolov7.pt"
classesFileName = "/home/unkindled/Desktop/Teknofest/MatiricieJr-YOLO/yolov7/deploy/ONNX/OpenCV/coco.names"

classes = None
#with open(classesFileName, 'rt') as f:
    #classes = f.read().rstrip('\n').split('\n')

def processDetectionIntoImage(image, detection):
    image = image.copy()
    image_height, image_width = image.shape[:2]

	# Resizing factor.
    x_factor = image_width / 640
    y_factor =  image_height / 640
    
    classIds = []
    confidences = []
    boxes = []

    rows = detection[0].shape[1]

    for i in range(rows):
        row = detection[0][0][i]
        confidence = row[4]

        if(confidence > 0.30):
            classScores = row[5:]
            classId = np.argmax(classScores)

            if(classScores[classId] > 0.3):
                confidences.append(confidence)
                classIds.append(classId)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    print(len(boxes))
    print(len(confidences))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.30, 0.30)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(image, (left, top), (left + width, top + height), (255,255,255), 3)
        label = "{}:{:.2f}".format(classes[classIds[i]], confidences[i])
        #draw_label(input_image, label, left, top)

    return image

imageFileName = "/home/unkindled/Desktop/Teknofest/MatiricieJr-YOLO/asd.jpeg"
frame = cv2.imread(imageFileName)
image = frame.copy()
(height, width) = image.shape[:2]

net = cv2.dnn.readNet(modelFileName)

blob = cv2.dnn.blobFromImage(image, 1/255, (640, 640), [0,0,0], 1, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

cv2.imshow("aaaaaaaa",processDetectionIntoImage(image.copy(), outputs).copy())
cv2.waitKey()
