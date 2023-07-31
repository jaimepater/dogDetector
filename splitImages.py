import os
import torch
import random
import shutil



def get_images_paths(route):
    images_list = os.listdir(route)
    images = []
    for i, name in enumerate(images_list):
        image_path = os.path.join(route, name)
        images.append(name)
    return images


def validate_images(routes, src):
    valid_routes = []
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    for name in routes:
        route = os.path.join(src, name)
        pred = model(route)
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.2]
        df = df[df["class"] == 16]
        if len(df) > 0:
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
    route = "./dogs"
    images = get_images_paths(route)
    valid_routes = validate_images(images,route)
    print("base", len(images))
    print("out", len(valid_routes))
    train_dir = "./contentDogs/train"
    test_dir = "./contentDogs/validation"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    train_images, test_images = get_arrays(valid_routes)
    move_images(train_images, route,train_dir)
    move_images(test_images, route, test_dir)



if __name__ == '__main__':
    split_images()
