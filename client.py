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

def get_camera():
    x = 1920
    y = 1080
    fps = 30
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, x)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, y)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(width, height, fps)
    return capture
def detector_bg(img, model, model_bd):
    pred = model(img)
    have_dog = False
    have_big_dog = False
    df = pred.pandas().xyxy[0]
    df = df[df["confidence"] > 0.5]
    df = df[df["class"] == 16]
    if len(df) > 0:
        have_dog = True
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin","ymin", "xmax", "ymax"]].values.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        pred_bg = model_bd(img)
        df_bg = pred_bg.pandas().xyxy[0]
        df_bg = df_bg[df_bg["confidence"] > 0.5]
        df_bg = df_bg[df_bg["class"] == 0]
        if len(df_bg) > 0:
            for i in range(df.shape[0]):
                bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    capture = get_camera()
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model_bd = torch.hub.load('ultralytics/yolov5', 'custom', path='./bd.pt',
                              force_reload=True)
    while True:
        ret, frame = capture.read()
        if ret:
           detector_bg(frame, model, model_bd)
           cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()











# See PyCharm help at https://www.jetbrains.com/help/pycharm/
