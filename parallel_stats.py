"""
File to generate new topo using previous languages and multiproc
"""

import torch
from model import *
from utils import *
from data import *
from multiprocessing import Pool
from tqdm import tqdm

meta = get_encoded_metadata(size=10000)
meaning_space = np.unique(meta, axis=0)
metrics = {}


def get_run_name(n: str):
    h = n.split("_")[2]
    s = n.split("_")[-1]
    i = n.split("_")[6]
    return h + "_" + s + "_" + i


def process_language(language_path: str):
    l = torch.load(open(language_path, "rb"))
    return custom_topo(meaning_space, l)


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
        p = Pool(6)
        jaccard_topographics = p.map(process_language, languages)
        for language, topo in zip(languages, jaccard_topographics):
            generation = int(language.split("_")[-1].split(".")[0])
            metrics[run_name][seed][generation]["custom_topo"] = topo

        # update pickle
        pickle.dump(metrics[run_name][seed], open(folder + "metrics.pkl", "wb"))

