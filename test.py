import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import cv2
import os
import numpy as np
from super_gradients.training import models
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
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./bd.pt',
                           force_reload=True)
    data, metaData = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
    for i, (img, tag) in enumerate(data['train']):
        img_cv = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
        pred = model(img_cv)
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.2]
        if True:
            df = df[df["class"] == 0]
            if len(df) > 0:
                bg_count = bg_count + 1
        df = df[df["class"] == 16]
        if len(df) > 0:
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
