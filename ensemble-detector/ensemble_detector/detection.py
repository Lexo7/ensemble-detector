# YOLO object detection
import cv2
import numpy as np
import time
import imutils

img = cv2.imread('../test images/foggy-014.jpg')
# cv2.imshow('window',  img)
# cv2.waitKey(1)
img = imutils.resize(img, width=320)

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet("./yolov7/yolov7.cfg", "./yolov7/yolov7.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
new_ln = net.getUnconnectedOutLayersNames()
# print(new_ln)
# print()
# print(len(ln), ln)

# construct a blob from the image
blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), (0,0,0), swapRB=True, crop=False)
r = blob[0, 0, :, :]

# cv2.imshow('blob', r)
# text = f'Blob shape={blob.shape}'
# cv2.displayOverlay('blob', text)
cv2.waitKey(0)

net.setInput(blob)
t0 = time.time()
outputs = net.forward(new_ln)
#outputs = net.forward(ln)
t = time.time()

# cv2.displayOverlay('window', f'forward propagation time={t-t0}')
# cv2.imshow('window',  img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

classes = []

with open("./yolov7/coco.names", 'r') as f:
    classes = f.read().splitlines()

print(len(classes))
#print(blob.shape)
#print(r.shape)

boxes = []
confidences = []
class_ids = []
height, width = img.shape[:2]

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype='uint8')
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x,y,w,h) = boxes[i]
        
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        # color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
        cv2.putText(img, label+" "+confidence,(x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

img = imutils.resize(img, width=500)
cv2.imshow('Yolov3', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


######################## CONVERTING THE CODE ABOVE INTO A FUNCTION ###########################################
# def yolov7_detect(frame):
    
#     # construct a blob from the image
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), (0,0,0), swapRB=True, crop=False)
#     # r = blob[0, 0, :, :]

#     # cv2.imshow('blob', r)
#     # text = f'Blob shape={blob.shape}'
#     # cv2.displayOverlay('blob', text)
#     # cv2.waitKey(0)

#     net.setInput(blob)
#     # t0 = time.time()
#     outputs = net.forward(new_ln)
#     #outputs = net.forward(ln)
#     # t = time.time()
    
#     boxes = []
#     confidences = []
#     class_ids = []
#     height, width = frame.shape[:2]

#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0]*width)
#                 center_y = int(detection[1]*height)
#                 w = int(detection[2]*width)
#                 h = int(detection[3]*height)
                
#                 x = int(center_x - w/2)
#                 y = int(center_y - h/2)
                
#                 boxes.append([x,y,w,h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype='uint8')
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     if len(indices) > 0:
#         for i in indices.flatten():
#             (x,y,w,h) = boxes[i]
            
#             label = str(classes[class_ids[i]])
#             confidence = str(round(confidences[i],2))
#             # color = [int(c) for c in colors[class_ids[i]]]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             # text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
#             cv2.putText(frame, label+" "+confidence,(x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#     frame = imutils.resize(frame, width=500)
#     return frame  
# # print(len(boxes))            