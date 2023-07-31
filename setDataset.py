import os
import torch
import cv2
import shutil
import uuid

ORIGIN = "./contentDogs"
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
    xmin = df['xmin']
    ymin = df['ymin']
    xmax = df['xmax']
    ymax = df['ymax']
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
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    for i, name in enumerate(paths):
        route = os.path.join(image_path, name)
        pred = model(route)
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.2]
        df = df[df["class"] == 16]
        id = uuid.uuid4()
        with open(f"{labels_path}/{partition}/{id}.txt","w") as f:
            for index, item in df.iterrows():
                text = covert_yolo_v5(item, route, 1)
                f.write(f"{text}\n")
        copy_file_with_new_name(route, f"{dataset_path}/{partition}", id)


if __name__ == '__main__':
    preprocessing("train")
    preprocessing("validation")
