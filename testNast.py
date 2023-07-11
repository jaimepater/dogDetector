import super_gradients
import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import cv2
import os
import numpy as np
from super_gradients.training import models
import supervision as sv
from super_gradients.common.object_names import Models


def get_images_paths(route):
    images_list = os.listdir(route)
    images = []
    for i, name in enumerate(images_list):
        image_path = os.path.join(route, name)
        images.append(image_path)
    return images


def detector():
    dog_count = 0
    detector_count = 0
    bg_count = 0
    yolo_nas = super_gradients.training.models.get("yolo_nas_m", pretrained_weights="coco")
    data, metaData = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
    for i, (img, tag) in enumerate(data['train'].take(20)):
        img_cv = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
        pred = yolo_nas.predict(img_cv)
        detections = sv.Detections.from_yolo_nas(pred[0])
        detections = detections[detections.confidence > 0.2]
        detections = detections[detections.class_id == 16]
        if len(detections) > 0:
            detector_count = detector_count + 1
        if tag == 1:
            dog_count = dog_count + 1
        print("i", i)
    print("big dog count", bg_count)
    print("dog count", dog_count)
    print("detector count", detector_count)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detector()











# See PyCharm help at https://www.jetbrains.com/help/pycharm/
