import os
import torch
import cv2
import shutil
import uuid
import super_gradients
import supervision as sv
from super_gradients.training import models

ORIGIN = "./content"
DESTINATION = "./dataset"


def get_images_paths(route):
    images_list = os.listdir(route)
    images = []
    for i, name in enumerate(images_list):
        image_path = os.path.join(route, name)
        images.append(name)
    return images


def copy_file_with_new_name(source_path, destination_path, new_name):
    # Get the extension of the source file
    _, extension = os.path.splitext(source_path)

    # Create the new file name with the original extension
    new_filename = f"{new_name}{extension}"

    # Generate the full destination path with the new file name
    destination = os.path.join(destination_path, new_filename)

    # Copy the file to the destination path
    shutil.copy2(source_path, destination)


def covert_yolo_v5(df, image, label):
    im = cv2.imread(image)
    h, w, c = im.shape
    xmin = df[0]
    ymin = df[1]
    xmax = df[2]
    ymax = df[3]
    b_center_x = (xmin + xmax) / 2
    b_center_y = (ymin + ymax) / 2
    b_width = (xmax - xmin)
    b_height = (ymax - ymin)
    b_center_x /= w
    b_center_y /= h
    b_width /= w
    b_height /= h
    return f"{label} {b_center_x} {b_center_y} {b_width} {b_height}"


def preprocessing(partition):
    image_path = os.path.join(ORIGIN, partition)
    labels_path = os.path.join(DESTINATION, "labels")
    dataset_path = os.path.join(DESTINATION, "images")
    paths = get_images_paths(image_path)
    yolo_nas = super_gradients.training.models.get("yolo_nas_m", pretrained_weights="coco")
    for i, name in enumerate(paths):
        route = os.path.join(image_path, name)
        pred = list(yolo_nas.predict(route))
        print("aaaaaaa", route)
        detections = sv.Detections.from_yolo_nas(pred[0])
        detections = detections[detections.confidence > 0.2]
        detections = detections[detections.class_id == 16]
        id = uuid.uuid4()
        with open(f"{labels_path}/{partition}/{id}.txt","w") as f:
            print(detections)
            for item in detections.xyxy:
                text = covert_yolo_v5(item, route, 0)
                f.write(f"{text}\n")
        copy_file_with_new_name(route, f"{dataset_path}/{partition}", id)


if __name__ == '__main__':
    preprocessing("train")
    preprocessing("validation")
