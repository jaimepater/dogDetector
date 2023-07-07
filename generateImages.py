import os
import cv2
import uuid
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

def save_images(route, images):
    for img in images:
        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(route, unique_filename)
        img_encoded = tf.image.encode_jpeg(tf.cast(img, tf.uint8))
        tf.io.write_file(image_path, img_encoded)


def load_images(route):
    images_list = os.listdir(route)
    images = []
    for i, name in enumerate(images_list):
        image_path = os.path.join(route, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def new_images(images):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=15,
        zoom_range=[0.7, 1.4],
        horizontal_flip=True,
        vertical_flip=True
    )

    augmented_images = []
    for image in images:
        augmented_batch = datagen.flow(np.expand_dims(image, axis=0), batch_size=5, shuffle=False)
        for i in range(5):
            augmented_image = augmented_batch.next()
            augmented_images.append(augmented_image[0])
    return augmented_images


def generate_images():
    route = "./akita"
    images = load_images(route)
    augmented_images = new_images(images)
    new_route = "./generateImages"
    save_images(new_route, augmented_images)


if __name__ == '__main__':
    generate_images()
