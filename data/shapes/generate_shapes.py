import os
import pickle
import numpy as np
from random import shuffle

import torch

from .generate_images import get_image

dir_path = os.path.dirname(os.path.realpath(__file__))

SEED = 42


def generate_image_dataset(size):
    """
    Generates an image dataset using the seed passed
    """
    images = []
    for i in range(size):
        images.append(get_image())
    shuffle(images)
    return images


def generate_shapes_dataset(train_size):
    """
    Generates shapes dataset and extract features
    @TODO - add parameters to extend generation and feature extraction process
    """
    np.random.seed(SEED)

    # --- Generate Datasets ----
    train_data = generate_image_dataset(train_size)
    train_data = generate_image_dataset(train_size)

    sets = {"dataset": train_data}

    # --- Save Generated Datasets ----
    for set_name, set_data in sets.items():
        set_inputs = np.asarray([image.data[:, :, 0:3] for image in set_data])
        np.save("{}/{}.input{}".format(dir_path, set_name, train_size), set_inputs)

        set_metadata = [image.metadata for image in set_data]
        pickle.dump(
            set_metadata,
            open("{}/{}.metadata{}.p".format(dir_path, set_name, train_size), "wb"),
        )

        oh = np.asarray([image.one_hot for image in set_data])
        np.save("{}/{}.encoded_metadata{}".format(dir_path, set_name, train_size), oh)


if __name__ == "__main__":
    generate_shapes_dataset(1000)
