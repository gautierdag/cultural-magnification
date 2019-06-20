import os
import numpy as np
from .generate_shapes import generate_shapes_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_encoded_metadata(size=10000):
    encoded_metadata_path = dir_path + "/dataset.encoded_metadata{}.npy".format(size)
    if not os.path.exists(encoded_metadata_path):
        print("Generating dataset")
        generate_shapes_dataset(size)
    meta = np.load(encoded_metadata_path)
    return meta.astype(np.float32)
