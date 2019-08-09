import argparse
import sys
import torch
import os
import time

from polybius.helpers.game_helper import get_trainer, get_training_data, get_meta_data
from polybius.helpers.train_helper import TrainHelper
from polybius.helpers.file_helper import FileHelper
from polybius.utils.logger import Logger

from polybius.models import Sender, Receiver
from polybius.data import AgentVocab

from utils import get_filename


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender/Receiver Agent on a task"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="number of batch iterations to train (default: 10k)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        metavar="N",
        help="embedding size for embedding layer (default: 64)",
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
        default=1024,
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

    # Arguments not specific to the training process itself
    parser.add_argument(
        "--debugging",
        help="Enable debugging mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="number of iterations between logs (default: 200)",
    )
    parser.add_argument(
        "--sender-path",
        type=str,
        default=False,
        metavar="S",
        help="Sender to be loaded",
    )
    parser.add_argument(
        "--receiver-path",
        type=str,
        default=False,
        metavar="S",
        help="Receiver to be loaded",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=False,
        metavar="S",
        help="Name to append to run file name",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=False,
        metavar="S",
        help="Additional folder within runs/",
    )
    parser.add_argument("--disable-print", help="Disable printing", action="store_true")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Amount of epochs to check for not improved validation score before early stopping",
    )
    parser.add_argument(
        "--test-mode",
        help="Only run the saved model on the test set",
        action="store_true",
    )
    parser.add_argument(
        "--resume-training",
        help="Resume the training from the saved model state",
        action="store_true",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5
        args.batch_size = 16

    return args


def save_model_state(
    model, checkpoint_path: str, epoch: int, iteration: int, best_score: int
):
    checkpoint_state = {}
    if model.sender:
        checkpoint_state["sender"] = model.sender.state_dict()
    if model.receiver:
        checkpoint_state["receiver"] = model.receiver.state_dict()
    if epoch:
        checkpoint_state["epoch"] = epoch
    if iteration:
        checkpoint_state["iteration"] = iteration
    if best_score:
        checkpoint_state["best_score"] = best_score

    torch.save(checkpoint_state, checkpoint_path)


def load_model_state(model, model_path):
    if not os.path.isfile(model_path):
        raise Exception(f'Model not found at "{model_path}"')
    checkpoint = torch.load(model_path)
    if "sender" in checkpoint.keys() and checkpoint["sender"]:
        model.sender.load_state_dict(checkpoint["sender"])
    if "receiver" in checkpoint.keys() and checkpoint["receiver"]:
        model.receiver.load_state_dict(checkpoint["receiver"])
    best_score = -1.0
    if "best_score" in checkpoint.keys() and checkpoint["best_score"]:
        best_score = checkpoint["best_score"]
    epoch = 0
    if "epoch" in checkpoint.keys() and checkpoint["epoch"]:
        epoch = checkpoint["epoch"]
    iteration = 0
    if "iteration" in checkpoint.keys() and checkpoint["iteration"]:
        iteration = checkpoint["iteration"]
    return epoch, iteration, best_score


def baseline(args):
    args = parse_arguments(args)
    args.dataset_type = "meta"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_helper = FileHelper()
    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    model_name = get_filename(args)
    run_folder = file_helper.get_run_folder(args.folder, model_name)

    logger = Logger(run_folder, print_logs=(not args.disable_print))
    logger.log_args(args)

    vocab = AgentVocab(args.vocab_size)

    # get sender and receiver models and save them
    sender = Sender(
        args.vocab_size,
        args.max_length,
        vocab.bound_idx,
        device,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=True,
        gumbel_softmax=True,
        input_size=15,
    )

    receiver = Receiver(
        args.vocab_size,
        device,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=15,
    )

    sender_file = file_helper.get_sender_path(run_folder)
    receiver_file = file_helper.get_receiver_path(run_folder)

    if receiver:
        torch.save(receiver, receiver_file)

    model = get_trainer(sender, device, "meta", receiver=receiver)

    model_path = file_helper.create_unique_model_path(model_name)

    best_accuracy = -1.0
    epoch = 0
    iteration = 0

    if args.resume_training or args.test_mode:
        epoch, iteration, best_accuracy = load_model_state(model, model_path)
        print(
            f"Loaded model. Resuming from - epoch: {epoch} | iteration: {iteration} | best accuracy: {best_accuracy}"
        )

    if not os.path.exists(file_helper.model_checkpoint_path):
        print("No checkpoint exists. Saving model...\r")
        torch.save(model.visual_module, file_helper.model_checkpoint_path)
        print("No checkpoint exists. Saving model...Done")

    train_data, valid_data, test_data, valid_meta_data, _ = get_training_data(
        device=device,
        batch_size=args.batch_size,
        k=3,
        debugging=args.debugging,
        dataset_type="meta",
    )

    train_meta_data, valid_meta_data, test_meta_data = get_meta_data()

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    if not args.disable_print:
        # Print info
        print("----------------------------------------")
        print(
            "Model name: {} \n|V|: {}\nL: {}".format(
                model_name, args.vocab_size, args.max_length
            )
        )
        print(sender)
        if receiver:
            print(receiver)

        print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    current_patience = args.patience
    best_accuracy = -1.0
    converged = False

    start_time = time.time()

    if args.test_mode:
        test_loss_meter, test_acc_meter, _ = train_helper.evaluate(
            model, test_data, test_meta_data, device, args.rl
        )

        average_test_accuracy = test_acc_meter.avg
        average_test_loss = test_loss_meter.avg

        print(
            f"TEST results: loss: {average_test_loss} | accuracy: {average_test_accuracy}"
        )
        return

    while iteration < args.iterations:
        for train_batch in train_data:
            print(f"{iteration}/{args.iterations}       \r", end="")

            # !!! This is the complete training procedure. Rest is only logging!
            _, _ = train_helper.train_one_batch(
                model, train_batch, optimizer, train_meta_data, device
            )

            if iteration % args.log_interval == 0:
                valid_loss_meter, valid_acc_meter, _, = train_helper.evaluate(
                    model, valid_data, valid_meta_data, device, False
                )

                new_best = False
                average_valid_accuracy = valid_acc_meter.avg

                if (
                    average_valid_accuracy < best_accuracy
                ):  # No new best found. May lead to early stopping
                    current_patience -= 1

                    if current_patience <= 0:
                        print("Model has converged. Stopping training...")
                        converged = True
                        break
                else:  # new best found. Is saved.
                    new_best = True
                    best_accuracy = average_valid_accuracy
                    current_patience = args.patience
                    save_model_state(model, model_path, epoch, iteration, best_accuracy)

                metrics = {
                    "loss": valid_loss_meter.avg,
                    "accuracy": valid_acc_meter.avg,
                }

                logger.log_metrics(iteration, metrics)

            iteration += 1
            if iteration >= args.iterations:
                break

        epoch += 1

        if converged:
            break

    return run_folder


if __name__ == "__main__":
    baseline(sys.argv[1:])
