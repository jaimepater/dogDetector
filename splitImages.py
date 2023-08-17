import os
import random
import shutil
import super_gradients
import supervision as sv
from super_gradients.training import models



def get_images_paths(route):
    images_list = os.listdir(route)
    images = []
    for i, name in enumerate(images_list):
        image_path = os.path.join(route, name)
        images.append(name)
    return images


def validate_images(routes, src):
    valid_routes = []
    yolo_nas = super_gradients.training.models.get("yolo_nas_m", pretrained_weights="coco")
    for name in routes:
        print("aaaa")
        route = os.path.join(src, name)
        pred = list(yolo_nas.predict(route))
        detections = sv.Detections.from_yolo_nas(pred[0])
        detections = detections[detections.confidence > 0.2]
        detections = detections[detections.class_id == 16]
        if len(detections) > 0:
            valid_routes.append(name)
    return valid_routes


def get_arrays(routes):
    num_images = len(routes)
    num_train = int(num_images * 0.8)
    random.shuffle(routes)
    train_images = routes[:num_train]
    test_images = routes[num_train:]
    return train_images, test_images


def move_images(routes, src_path, dst_path):
    for image in routes:
        src = os.path.join(src_path, image)
        dst = os.path.join(dst_path, image)
        shutil.copy(src, dst)


def split_images():
    route = "./generateImages"
    images = get_images_paths(route)
    valid_routes = validate_images(images,route)
    print("base", len(images))
    print("out", len(valid_routes))
    train_dir = "./content/train"
    test_dir = "./content/validation"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    train_images, test_images = get_arrays(valid_routes)
    move_images(train_images, route,train_dir)
    move_images(test_images, route, test_dir)



if __name__ == '__main__':
    split_images()
