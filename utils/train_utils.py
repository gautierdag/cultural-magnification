import random
import numpy as np
import torch
import pickle
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from functools import partial
import glob
import os

# Training and Evaluation helper functions


class AverageMeter:
    def __init__(self):
        """
        Computes and stores the average and current value
        Taken from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_batch(model, batch, targets, optimizer):
    """
    Train for single batch
    """
    model.train()
    optimizer.zero_grad()
    loss, acc, _ = model(batch, targets)
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def train_one_epoch(model, data, optimizer):
    """
    Train for a whole epoch
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for d in tqdm(data, total=len(data)):
        loss, acc = train_one_batch(model, d, optimizer)
        loss_meter.update(loss)
        acc_meter.update(acc)

    return loss_meter, acc_meter


def evaluate(model, data):
    """
    Evaluates model on data
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    sequences = []

    model.eval()
    with torch.no_grad():
        for (batch, targets) in data:
            loss, acc, seq = model(batch, targets)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            sequences.append(seq)

    return loss_meter, acc_meter, torch.cat(sequences, 0)


def infer_new_language(model, full_dataset, batch_size=64):
    """
    Go over entire dataset and return the infered sequences.
    """
    dataloader = DataLoader(full_dataset, batch_size=batch_size)
    loss, acc, sequences = evaluate(model, dataloader)
    return sequences


# Folder and Dataset functions


def split_dataset_into_dataloaders(dataset, batch_size=32):
    """
    Splits a pytorch dataset into train, valid, and test dataloaders
    """
    # 50 % of dataset used in train
    train_length = int(0.5 * len(dataset))
    # 10 % of dataset used in validation set
    valid_length = int(0.1 * len(dataset))
    # rest used in test set
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_length, valid_length, test_length]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


def get_filename(params):
    """
    Generates a filename from baseline params (see baseline.py)
    """

    if params.name:
        return params.name
    name = params.model_type
    name += "_h_{}".format(params.hidden_size)
    name += "_lr_{}".format(params.lr)
    name += "_iters_{}".format(params.iterations)
    name += "_max_len_{}".format(params.max_length)
    name += "_vocab_{}".format(params.vocab_size)
    name += "_btch_size_{}".format(params.batch_size)
    if params.debugging:
        name += "_debug"
    return name


def seed_torch(seed=42):
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_folder_if_not_exists(folder_name):
    """
    Creates folder at folder name if folder does not exist
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def check_language_exists(path):
    """
    Checks if a language file exists at the path
    """
    files = glob.glob(path + "/language_at_*.p")
    return len(files) > 0


def get_latest_language(path):
    """
    Gets the latest language and metrics
    """
    files = glob.glob(path + "/language_at_*.p")
    last_g = 0
    last_file = path + "/language_at_0.p"
    for f in files:
        g = int(f.split("_")[-1].split(".")[0])
        if last_g < g:
            last_file = f
            last_g = g

    language = torch.load(last_file)
    metrics = pickle.load(open(path + "/metrics.pkl", "rb"))
    return last_g + 1, language, metrics
