import pickle
import argparse
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from model import *
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
        "--resume",
        help="Resume iterated learning (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
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
        default=500,
        metavar="N",
        help="number of iterations between logs (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size for hidden layer (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        metavar="N",
        help="max sentence length allowed for communication (default: 10)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        metavar="N",
        help="Size of generated dataset",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.5,
        metavar="N",
        help="Proportional size of the train set in IL",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=25,
        metavar="N",
        help="Size of vocabulary (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=False,
        metavar="S",
        help="Name to append to run file name",
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        const="lstm",
        nargs="?",
        choices=["gru", "lstm"],
        help="Model to use (default: %(default)s)",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5
        args.hidden_size = 64

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
    meta = get_encoded_metadata(size=args.dataset_size)
    meaning_space = np.unique(meta, axis=0)
    print("Meaning Space Length: {}".format(len(meaning_space)))

    if args.resume and check_language_exists(run_folder):
        g, language, metrics = get_latest_language(run_folder)
    else:
        g = 0
        language = generate_uniform_language_fixed_length(
            vocab, len(meaning_space), args.max_length
        )
        language = torch.Tensor(language).type(torch.long)
        torch.save(language, "{}/initial_language.p".format(run_folder))
        metrics = {}

    while g < args.generations:

        dataset = ILDataset(meaning_space, language)

        train_dataloader, valid_dataloader, test_dataloader = split_dataset_into_dataloaders(
            dataset, batch_size=args.batch_size, train_size=args.train_size
        )

        if args.model_type == "lstm":
            model = LSTMModel(vocab.full_vocab_size, args.max_length)
        elif args.model_type == "gru":
            model = GRUModel(vocab.full_vocab_size, args.max_length)
        else:
            return ValueError("invalid model type")

        model_file = "{}/model{}.p".format(run_folder, g)
        torch.save(model, model_file)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer = ILTrainer(model)
        trainer.to(device)

        # Train
        i = 0
        metrics[g] = {"validation_loss": {}, "validation_acc": {}}
        while i < args.iterations:

            for (batch, targets) in train_dataloader:
                loss, acc = train_one_batch(trainer, batch, targets, optimizer)
                if i % args.log_interval == 0:
                    valid_loss_meter, valid_acc_meter, sequences = evaluate(
                        trainer, valid_dataloader
                    )
                    metrics[g]["validation_loss"][i] = valid_loss_meter.avg
                    metrics[g]["validation_acc"][i] = valid_acc_meter.avg

                i += 1

        torch.save(trainer.model, model_file)

        # Evaluate best model on test data
        test_loss_meter, test_acc_meter, test_sequences = evaluate(
            trainer, test_dataloader
        )
        print(
            "{}/{} Generation: test loss: {}, test accuracy: {}".format(
                g, args.generations, test_loss_meter.avg, test_acc_meter.avg
            )
        )

        new_language = infer_new_language(trainer, dataset, batch_size=args.batch_size)
        torch.save(new_language.cpu(), "{}/language_at_{}.p".format(run_folder, g))

        total_distance, perfect_matches = message_distance(
            new_language, language, vocab.full_vocab_size
        )
        jaccard_sim = jaccard_similarity(new_language, language)
        num_unique_messages = len(torch.unique(new_language, dim=0))
        topographic_similarity = calc_topographical_similarity(
            meaning_space, new_language, vocab.full_vocab_size
        )

        metrics[g]["total_distance"] = total_distance
        metrics[g]["perfect_matches"] = perfect_matches
        metrics[g]["jaccard_sim"] = jaccard_sim
        metrics[g]["num_unique_messages"] = num_unique_messages
        metrics[g]["test_loss"] = test_loss_meter.avg
        metrics[g]["test_acc"] = test_acc_meter.avg
        metrics[g]["topographic_similarity"] = topographic_similarity

        writer.add_scalar("num_unique_messages", num_unique_messages, g)
        writer.add_scalar("test_loss", test_loss_meter.avg, g)
        writer.add_scalar("test_acc", test_acc_meter.avg, g)
        writer.add_scalar("topographic_similarity", topographic_similarity, g)

        # dump metrics
        pickle.dump(metrics, open("{}/metrics.pkl".format(run_folder), "wb"))

        language = new_language
        g += 1

    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
