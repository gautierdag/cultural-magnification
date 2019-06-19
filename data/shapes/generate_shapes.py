import os
import pickle
import numpy as np
from random import shuffle

import torch

from generate_images import get_image

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


def generate_shapes_dataset():
    """
    Generates shapes dataset and extract features
    @TODO - add parameters to extend generation and feature extraction process
    """
    np.random.seed(SEED)

    folder_name = "balanced"

    # From Serhii's original experiment
    train_size = 1000

    # --- Generate Datasets ----
    train_data = generate_image_dataset(train_size)
    train_data = generate_image_dataset(train_size)

    sets = {"dataset": train_data}

    # --- Save Generated Datasets ----
    folder_name = os.path.join(dir_path, folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for set_name, set_data in sets.items():
        set_inputs = np.asarray([image.data[:, :, 0:3] for image in set_data])
        np.save("{}/{}.input".format(folder_name, set_name), set_inputs)

        set_metadata = [image.metadata for image in set_data]
        pickle.dump(
            set_metadata, open("{}/{}.metadata.p".format(folder_name, set_name), "wb")
        )

        oh = np.asarray([image.one_hot for image in set_data])
        np.save("{}/{}.encoded_metadata".format(folder_name, set_name), oh)

if __name__ == "__main__":
    generate_shapes_dataset()
