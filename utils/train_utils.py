import random
import numpy as np
import torch
import pickle
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import BatchSampler
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
    loss, acc, _, _ = model(batch, targets)
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
    hidden_states = []

    model.eval()
    with torch.no_grad():
        for (batch, targets) in data:
            loss, acc, seq, hid = model(batch, targets)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            sequences.append(seq)
            hidden_states.append(hid)

    return loss_meter, acc_meter, torch.cat(sequences, 0), torch.cat(hidden_states, 0)


def infer_new_language(model, full_dataset, batch_size=64):
    """
    Go over entire dataset and return the infered sequences.
    """
    dataloader = DataLoader(full_dataset, batch_size=batch_size)
    loss, acc, sequences, hidden_states = evaluate(model, dataloader)
    return sequences, hidden_states


# Folder and Dataset functions


def split_dataset_into_dataloaders(dataset, sizes=[], batch_size=32, sampler=None):
    """
    Splits a pytorch dataset into different sizes of dataloaders
    """

    # 50 % of dataset used in train
    train_length = int(0.5 * len(dataset))
    # 10 % of dataset used in validation set
    valid_length = int(0.1 * len(dataset))
    # rest used in test set
    test_length = len(dataset) - train_length - valid_length

    if len(sizes) == 0:
        sizes = [train_length, valid_length, test_length]

    datasets = random_split(dataset, sizes)

    return (
        DataLoader(
            d,
            batch_size=batch_size if sampler is None else False,
            batch_sampler=BatchSampler(
                sampler(d), batch_size=batch_size, drop_last=False
            )
            if sampler is not None
            else False,
        )
        for d in datasets
    )


def get_filename(params):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = "lstm"  # params.model_type
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


def save_model_state(model, model_path: str, epoch: int, iteration: int):
    checkpoint_state = {}
    if model.sender:
        checkpoint_state["sender"] = model.sender.state_dict()
    if model.receiver:
        checkpoint_state["receiver"] = model.receiver.state_dict()
    if epoch:
        checkpoint_state["epoch"] = epoch
    if iteration:
        checkpoint_state["iteration"] = iteration

    torch.save(checkpoint_state, model_path)


def load_model_state(model, model_path):
    if not os.path.isfile(model_path):
        raise Exception(f'Model not found at "{model_path}"')
    checkpoint = torch.load(model_path)
    if "sender" in checkpoint.keys() and checkpoint["sender"]:
        model.sender.load_state_dict(checkpoint["sender"])
    if "receiver" in checkpoint.keys() and checkpoint["receiver"]:
        model.receiver.load_state_dict(checkpoint["receiver"])
    if "epoch" in checkpoint.keys() and checkpoint["epoch"]:
        epoch = checkpoint["epoch"]
    if "iteration" in checkpoint.keys() and checkpoint["iteration"]:
        iteration = checkpoint["iteration"]
    return epoch, iteration
