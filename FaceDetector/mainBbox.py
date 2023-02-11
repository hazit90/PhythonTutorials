#face detector using https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface

import cv2
import onnxruntime as ort
import argparse
import numpy as np
from dependencies.box_utils import predict
import time

# modelPath = "model/version-RFB-640.onnx"
# width = 640
# height = 480
modelPath = "model/version-RFB-320.onnx"
width = 320
height = 240


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# face detection method
def faceDetector(face_detector, orig_image, threshold = 0.7):
    if orig_image is None:
        return
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs


def main() -> int:
    #create webcam object
    cap = cv2.VideoCapture(0)

    #if webcam is not opened, exit program
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #load model
    face_detector = ort.InferenceSession(modelPath)
    color = (255, 128, 0)
    while True:
        totalSt = time.time()
        grabSt = time.time()
        ret, orig_image = cap.read()
        grabDelta = time.time() - grabSt
        if not ret:
            continue
        detectSt = time.time()
        boxes, labels, probs = faceDetector(face_detector, orig_image)
        detectDelta = time.time() - detectSt
        #print time in ms
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
        #     cv2.imshow('', orig_image)
        #     cv2.waitKey(1)
        cv2.imshow('', orig_image)
        #if esc key is pressed, break loop
        
        if cv2.waitKey(1) == 27:
            break
        totalDelta = time.time() - totalSt

        print("Grab: {:.2f}ms, Detect: {:.2f}ms, Total: {:.2f}ms".format(grabDelta * 1000, detectDelta * 1000, totalDelta * 1000))

#if main
if __name__ == "__main__":
    main() 