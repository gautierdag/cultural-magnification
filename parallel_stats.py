"""
File to generate new topo using previous languages and multiproc
"""
from model import *
from utils import *
from data import *
from multiprocessing import Pool
from tqdm import tqdm
import torch
import itertools
import numpy as np
import scipy.spatial
import scipy.stats
import warnings


def one_hot(a, ncols):
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def process_similarity(file: str, vocab_size=27):
    representation = torch.load(open(file, "rb"))
    dataset_length = representation.shape[0]
    representation = representation.cpu().numpy()

    if "hidden" in file:
        distance = scipy.spatial.distance.cosine
    else:
        representation = one_hot(representation - 1, vocab_size).reshape(
            dataset_length, -1
        )
        distance = scipy.spatial.distance.hamming

    combinations = list(itertools.combinations(range(dataset_length), 2))
    sim_representation = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]
        sim_representation[i] = distance(representation[s1], representation[s2])

    new_file_name = file[:-2] + "_sim.npy"
    np.save(new_file_name, sim_representation)


if __name__ == "__main__":
    folders = list(glob.glob("runs/gru*0.7/*/"))

    # Loading metrics from saved pickles
    for folder in tqdm(folders):
        languages = glob.glob(folder + "language_at_*.p")
        hiddens = glob.glob(folder + "hidden_states_at_*.p")
        files = languages + hiddens
        p = Pool(4)
        p.map(process_similarity, files)
