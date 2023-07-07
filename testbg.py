import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import os
import cv2
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


def detector_bg(images):
    dog_count = 0
    img_count = 0
    big_dog_count = 0
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model_bd = torch.hub.load('ultralytics/yolov5', 'custom', path='./bd.pt',
                           force_reload=True)
    for i, img in enumerate(images):
        pred = model(img)
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.2]
        df = df[df["class"] == 16]
        img_count = img_count + 1
        if len(df) > 0:
            dog_count = dog_count + 1
            pred_bg = model_bd(img)
            df_bg = pred_bg.pandas().xyxy[0]
            df_bg = df_bg[df_bg["confidence"] > 0.2]
            df_bg = df_bg[df_bg["class"] == 0]
            if len(df_bg) > 0:
                big_dog_count = big_dog_count +1

    print("dog_count", dog_count)
    print("img_count", img_count)
    print("big_dog_count", big_dog_count)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images = get_images_paths("./dogs")
    detector_bg(images)











# See PyCharm help at https://www.jetbrains.com/help/pycharm/
