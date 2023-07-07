import os
import cv2
import uuid
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import tensorflow_datasets as tfds

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
        augmented_images.append(image)
        image = np.expand_dims(image, axis=0)
        augmented_batch = datagen.flow(image, batch_size=5, shuffle=False)
        for i in range(5):
            augmented_image = augmented_batch.next()
            augmented_images.append(augmented_image[0])
    return augmented_images


def get_dogs_images():
    dogs = []
    data, metaData = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
    for i, (img, tag) in enumerate(data['train'].take(100)):
        if tag == 1:
            dogs.append(img.numpy())
    return dogs


def generate_images():
    dogs = get_dogs_images()
    augmented_images = new_images(dogs)
    new_route = "./dogs"
    save_images(new_route, augmented_images)


if __name__ == '__main__':
    generate_images()
