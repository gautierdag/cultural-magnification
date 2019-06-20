import pickle
import argparse
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tensorboardX import SummaryWriter

from model import ILTrainer, LSTMModel
from utils import *
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(description="Shapes Iterated Learning Setup")
    parser.add_argument(
        "--debugging",
        help="Enable debugging mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--greedy",
        help="Use argmax at prediction time instead of sampling (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="number of batch iterations to train (default: 10k)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        metavar="N",
        help="number of generations to iterate over (default: 10)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="number of iterations between logs (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=5,
        metavar="N",
        help="max sentence length allowed for communication (default: 5)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5,
        metavar="N",
        help="Size of vocabulary (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--model-path", type=str, default=False, metavar="S", help="Model to be loaded"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=False,
        metavar="S",
        help="Name to append to run file name",
    )
    parser.add_argument("--disable-print", help="Disable printing", action="store_true")

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5

    return args


def main(args):

    args = parse_arguments(args)
    seed_torch(seed=args.seed)

    model_name = get_filename(args)
    run_folder = "runs/" + model_name + "/" + str(args.seed)
    writer = SummaryWriter(log_dir=run_folder)

    # dump arguments
    pickle.dump(args, open("{}/experiment_params.pkl".format(run_folder), "wb"))

    # get encoded metadata, vocab and original language
    vocab = AgentVocab(args.vocab_size)
    meta = get_encoded_metadata()
    language = generate_uniform_language_fixed_length(vocab, len(meta), args.max_length)

    metrics = {}
    for g in range(args.generations):
        dataset = ILDataset(meta, language)

        train_dataloader, valid_dataloader, test_dataloader = split_dataset_into_dataloaders(
            dataset, batch_size=args.batch_size
        )

        model = LSTMModel(vocab.full_vocab_size, args.max_length)
        model_file = "{}/model{}.p".format(run_folder, g)
        torch.save(model, model_file)

        print("----------------------------------------")
        print(
            "Model name: {} \n|V|: {}\nL: {}".format(
                model_name, args.vocab_size, args.max_length
            )
        )
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters: {}".format(pytorch_total_params))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer = ILTrainer(model)
        trainer.to(device)

        # Train
        i = 0
        best_valid_acc = -1
        metrics[g] = {"validation_loss": {}, "validation_acc": {}}
        while i < args.iterations:

            for (batch, targets) in train_dataloader:
                loss, acc = train_one_batch(trainer, batch, targets, optimizer)
                if i % args.log_interval == 0:
                    valid_loss_meter, valid_acc_meter, sequences = evaluate(
                        trainer, valid_dataloader
                    )
                    print(
                        "{}/{} Iterations: val loss: {}, val accuracy: {}".format(
                            i,
                            args.iterations,
                            valid_loss_meter.avg,
                            valid_acc_meter.avg,
                        )
                    )
                    # Save if best model so far according to validation accuracy
                    if valid_acc_meter.avg > best_valid_acc:
                        best_valid_acc = valid_acc_meter.avg
                        torch.save(trainer.model, model_file)

                    metrics[g]["validation_loss"][i] = valid_loss_meter.avg
                    metrics[g]["validation_acc"][i] = valid_acc_meter.avg

                i += 1

        model = torch.load(model_file)
        trainer = ILTrainer(model)
        trainer.to(device)

        # Evaluate best model on test data
        test_loss_meter, test_acc_meter, test_sequences = evaluate(
            trainer, test_dataloader
        )
        topographic_similarity = get_topographical_similarity(trainer, test_dataloader)

        language = infer_new_language(trainer, dataset, batch_size=args.batch_size)
        torch.save(language.cpu(), "{}/language_at_{}.p".format(run_folder, g))

        num_unique_messages = len(torch.unique(test_sequences, dim=0))

        metrics[g]["num_unique_messages"] = num_unique_messages
        metrics[g]["test_loss"] = test_loss_meter.avg
        metrics[g]["test_acc"] = test_acc_meter.avg
        metrics[g]["topographic_similarity"] = topographic_similarity

        writer.add_scalar("num_unique_messages", num_unique_messages, g)
        writer.add_scalar("test_loss", test_loss_meter.avg, g)
        writer.add_scalar("test_acc", test_acc_meter.avg, g)
        writer.add_scalar("topographic_similarity", topographic_similarity, g)

        # dump metrics
        pickle.dump(metrics, open("{}/metrics.pkl".format(run_folder, i), "wb"))

    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
