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

meta = get_encoded_metadata(size=10000)
meaning_space = np.unique(meta, axis=0)
np.save("meaning_space.npy", meaning_space)
metrics = {}


def get_run_name(n: str):
    h = n.split("_")[2]
    s = n.split("_")[-1]
    i = n.split("_")[6]
    return h + "_" + s + "_" + i


def complement_topo(compositional_representation, generated_sequences, vocab_size=27):
    """
    Calculates Topological Similarity using all possible pair combinations
    Args:
        compositional_representation (np.array): one-hot encoded compositional, size N*C
        messages (torch.tensor): messages, size N*M
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in representation/messages
    """
    dataset_length = compositional_representation.shape[0]

    combinations = list(itertools.combinations(range(dataset_length), 2))

    if hasattr(generated_sequences, "numpy"):
        generated_sequences = generated_sequences.cpu().numpy()

    sim_representation = np.zeros(len(combinations))
    sim_sequences = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]
        sim_representation[i] = scipy.spatial.distance.hamming(
            compositional_representation[s1], compositional_representation[s2]
        )
        complement = (
            vocab_size
            - len(set(generated_sequences[s1]).union(set(generated_sequences[s2])))
        ) / vocab_size
        sim_sequences[i] = 1 - complement

    # check if standard deviation is not 0
    if sim_sequences.std() == 0.0 or sim_representation.std() == 0.0:
        warnings.warn("Standard deviation of 0.0 for passed parameter in custom_topo")
        topographic_similarity = 0
    else:
        topographic_similarity = scipy.stats.pearsonr(
            sim_sequences, sim_representation
        )[0]

    return topographic_similarity


def process_language(language_path: str):
    l = torch.load(open(language_path, "rb"))
    return complement_topo(meaning_space, l)


if __name__ == "__main__":
    folders = list(glob.glob("runs/*/*/"))

    # Loading metrics from saved pickles
    for folder in tqdm(folders):
        run = folder.split("/")[1]
        run_name = get_run_name(run)
        if run_name not in metrics:
            metrics[run_name] = {}
        seed = folder.split("/")[2]
        try:
            metrics[run_name][seed] = pickle.load(open(folder + "metrics.pkl", "rb"))
        except:
            continue

        languages = glob.glob(folder + "language_at_*")
        p = Pool(4)
        custom_topos = p.map(process_language, languages)
        for language, topo in zip(languages, custom_topos):
            generation = int(language.split("_")[-1].split(".")[0])
            metrics[run_name][seed][generation]["complement_topo"] = topo

        # update pickle
        pickle.dump(metrics[run_name][seed], open(folder + "metrics.pkl", "wb"))

