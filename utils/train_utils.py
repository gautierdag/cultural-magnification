import random
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
import os


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


def train_one_batch(model, batch, optimizer):
    """
    Train for single batch
    """
    model.train()
    optimizer.zero_grad()
    loss, acc = model(batch)
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
    
    model.eval()
    with torch.no_grad():
        for d in data:
            loss, acc = model(batch)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
       
    return loss_meter, acc_meter

def get_filename(params):
    """
    Generates a filename from baseline params (see baseline.py)
    """

    if params.name:
        return params.name
    name = ""
    name += "h_{}".format(params.hidden_size)
    name += "_lr_{}".format(params.lr)
    name += "_max_len_{}".format(params.max_length)        
    name += "_vocab_{}".format(params.vocab_size)
    name += "_seed_{}".format(params.seed)
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
