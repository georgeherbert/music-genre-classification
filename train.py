import argparse
import pickle
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import GTZAN

torch.backends.cudnn.benchmark = True

# torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda")


class ShallowCNN(nn.Module):
    """A shallow convolutional neural network.

    This class implements a shallow convolutional neural network that takes
    in a batch of spectrograms and outputs a music genre predictions.
    """

    def __init__(self, batch_norm: bool = False) -> None:
        """Initialise the shallow CNN.

        This method initialises the layers of the shallow CNN, including the
        convolutional layers, batch normalisation layers (if applicable),
        pooling layers, fully-connected layers, and the leaky ReLU activation
        function.

        Args:
            batch_norm (bool, optional): Whether to use batch normalisation.
            Defaults to False.

        Returns:
            None
        """
        super().__init__()
        self.batch_norm = batch_norm

        self.conv_left = nn.Conv2d(1, 16, (10, 23), 1, "same")
        self.bn_conv_left = nn.BatchNorm2d(16)
        self.pool_left = nn.MaxPool2d((1, 20), (1, 20))

        self.conv_right = nn.Conv2d(1, 16, (21, 20), 1, "same")
        self.bn_conv_right = nn.BatchNorm2d(16)
        self.pool_right = nn.MaxPool2d((20, 1), (20, 1))

        self.fc_1 = nn.Linear(10240, 200)
        self.dropout = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(200, 10)

        self.leaky_relu = nn.LeakyReLU(0.3)

        self.initialise_layer(self.conv_left)
        self.initialise_layer(self.conv_right)
        self.initialise_layer(self.fc_1)
        self.initialise_layer(self.fc_2)

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shallow CNN.

        This method performs a forward pass through the shallow CNN, applying
        the convolutional, pooling, and fully-connected layers, as well as
        the leaky ReLU activation function and dropout regularisation.

        Args:
            images (torch.Tensor): A batch of input spectrograms with shape
                (N, C, H, W), where N is the batch size, C is the number of
                channels, and H and W are the height and width of the
                spectrograms.

        Returns:
            torch.Tensor: The output of the shallow CNN.
        """
        x_left = self.conv_left(spectrograms)
        if self.batch_norm:
            x_left = self.bn_conv_left(x_left)
        x_left = self.leaky_relu(x_left)
        x_left = self.pool_left(x_left)
        x_right = self.conv_right(spectrograms)
        if self.batch_norm:
            x_right = self.bn_conv_right(x_right)
        x_right = self.leaky_relu(x_right)
        x_right = self.pool_right(x_right)
        x = torch.cat([torch.flatten(x_left, 1), torch.flatten(x_right, 1)], 1)
        x = self.fc_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x

    @staticmethod
    def initialise_layer(layer) -> None:
        """Initialise a layer in the shallow CNN.

        This method initialises the weights and biases of a layer in the
        shallow CNN using the Kaiming normal initialisation method with a
        Leaky ReLU slope of 0.3.

        Args:
            layer (torch.nn.Module): The layer to initialise.

        Returns:
            None
        """
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight, 0.3)


class Trainer:
    """A class for training a model.

    This class provides methods for training a model on a given dataset,
    evaluating the model on validation data, and logging training progress.
    """

    def __init__(
            self,
            device: torch.device,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimiser: torch.optim.Optimizer,
            summary_writer: SummaryWriter
    ) -> None:
        """Initialises a new Trainer instance.

        Args:
            device (torch.device): The device to use for training (e.g. "cpu"
                or "cuda").
            model (nn.Module): The model to train.
            train_loader (DataLoader): A DataLoader for the training data.
            val_loader (DataLoader): A DataLoader for the validation data.
            criterion (nn.Module): The loss function to use for training.
            optimiser (torch.optim.Optimizer): The optimiser to use for
                training.
            summary_writer (SummaryWriter): A SummaryWriter for logging
                training progress.

        Returns:
            None
        """
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.summary_writer = summary_writer
        self.step = 0

    def l1_penalty(self, penalty: float) -> torch.Tensor:
        """Calculate the L1 penalty for the model's weights.

        Args:
            penalty (float): The penalty factor to apply to the L1 penalty.

        Returns:
            torch.Tensor: The calculated L1 penalty.
        """
        params = self.model.named_parameters()
        weights = torch.cat([p.view(-1) for n, p in params if ".weight" in n])
        return weights.abs().sum() * penalty

    def calc_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss for the model.

        Args:
            logits (torch.Tensor): The model's predicted logits.
            labels (torch.Tensor): The true labels for the input data.

        Returns:
            torch.Tensor: The calculated loss.
        """
        return self.criterion(logits, labels) + self.l1_penalty(0.0001)

    def calc_maj_max_preds(
        self,
        preds_by_file: Dict[str, List[int]],
        probs_by_file: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Calculate the majority vote and maximum probability predictions for
        each file.

        Args:
            preds_by_file (Dict[str, List[int]]): A dictionary mapping file
                names to lists of predicted labels for that file.
            probs_by_file (Dict[str, torch.Tensor]): A dictionary mapping file
                names to tensors of predicted probabilities for that file.

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]: A tuple containing two
            dictionaries: the first maps file names to majority vote
            predictions, and the second maps file names to maximum probability
            predictions.
        """
        maj_vote_preds = {}
        max_prob_preds = {}

        for file, preds in preds_by_file.items():
            maj_vote_preds[file] = statistics.mode(preds)

        for file, probs in probs_by_file.items():
            max_prob_preds[file] = torch.argmax(probs).item()

        return maj_vote_preds, max_prob_preds

    def log_accuracies(
        self,
        labels_by_file: Dict[str, int],
        maj_vote_preds: Dict[str, int],
        max_prob_preds: Dict[str, int],
        raw_num_correct: float
    ) -> None:
        """Log the raw, majority vote, and maximum probability accuracies for
        the validation set.

        Args:
            labels_by_file (Dict[str, int]): A dictionary mapping file names
                to true labels for that file.
            maj_vote_preds Dict[str, int]: A dictionary mapping file names to
                majority vote predictions for that file.
            max_prob_preds Dict[str, int]: A dictionary mapping file names to
                maximum probability predictions for that file.
            raw_num_correct (float): The number of correct predictions made by
                the raw model output.

        Returns:
            None
        """
        maj_vote_num_correct = 0
        max_prob_num_correct = 0

        for file, target_label in labels_by_file.items():
            maj_vote_num_correct += target_label == maj_vote_preds[file]
            max_prob_num_correct += target_label == max_prob_preds[file]

        raw_accuracy = raw_num_correct / len(self.val_loader.dataset) * 100
        maj_vote_accuracy = maj_vote_num_correct / len(labels_by_file) * 100
        max_prob_accuracy = max_prob_num_correct / len(labels_by_file) * 100

        self.summary_writer.add_scalars(
            "accuracy", {"raw_val": raw_accuracy}, self.step)
        self.summary_writer.add_scalars(
            "accuracy", {"maj_val": maj_vote_accuracy}, self.step)
        self.summary_writer.add_scalars(
            "accuracy", {"max_prob_val": max_prob_accuracy}, self.step)

    def save_preds(
        self,
        preds_by_file: Dict[str, List[int]],
        maj_vote_preds: Dict[str, int],
        max_prob_preds: Dict[str, int]
    ) -> None:
        """Save the raw, majority vote, and maximum probability predictions
        for each file.

        Args:
            preds_by_file (Dict[str, List[int]]): A dictionary mapping file
                names to lists of predicted labels for that file.
            maj_vote_preds (Dict[str, int]): A dictionary mapping file names
                to majority vote predictions for that file.
            max_prob_preds (Dict[str, int]): A dictionary mapping file names
                to maximum probability predictions for that file.

        Returns:
            None
        """
        with open("preds_raw.pkl", "wb") as file:
            pickle.dump(preds_by_file, file)
        with open("preds_maj_vote.pkl", "wb") as file:
            pickle.dump(maj_vote_preds, file)
        with open("preds_max_prob.pkl", "wb") as file:
            pickle.dump(max_prob_preds, file)

    def log_validation(
        self,
        preds_by_file: Dict[str, List[int]],
        probs_by_file: Dict[str, torch.Tensor],
        labels_by_file: Dict[str, int],
        raw_num_correct: int,
        total_loss: float,
        save_preds: bool
    ) -> None:
        """Log the validation results for the model.

        Args:
            preds_by_file (Dict[str, List[int]]): A dictionary mapping file
                names to lists of predicted labels for that file.
            probs_by_file (Dict[str, torch.Tensor]): A dictionary mapping file
                names to tensors of predicted probabilities for that file.
            labels_by_file (Dict[str, int]): A dictionary mapping file names
                to true labels for that file.
            raw_num_correct (int): The number of correct predictions made by
                the raw model output.
            total_loss (float): The total loss for the validation set.
            save_preds (bool): A flag indicating whether to save the
                predictions.

        Returns:
            None
        """
        maj_max_preds = self.calc_maj_max_preds(preds_by_file, probs_by_file)
        self.log_accuracies(
            labels_by_file,
            maj_max_preds[0],
            maj_max_preds[1],
            raw_num_correct
        )
        self.summary_writer.add_scalars(
            "loss", {"val": total_loss / len(self.val_loader)}, self.step
        )
        if save_preds:
            self.save_preds(preds_by_file, maj_max_preds[0], maj_max_preds[1])

    def validate(self, save_preds: bool = False) -> None:
        """
        Validate the model on a validation set.

        This method iterates over the validation set, makes predictions
        using the model, and calculates the validation loss and accuracy.
        It also logs the predictions and probabilities for each example
        in the validation set.

        Args:
            save_preds (bool, optional): A flag indicating whether to save the
            predictions.

        Returns:
            None
        """
        self.model.eval()
        raw_num_correct = 0
        total_loss = 0.0
        labels_by_file = defaultdict(int)
        preds_by_file = defaultdict(list)
        probs_by_file = defaultdict(lambda: torch.zeros(10))

        with torch.no_grad():
            for files, spectrograms, labels, _ in self.val_loader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(spectrograms)
                total_loss += self.calc_loss(logits, labels).item()
                preds = torch.argmax(logits, 1)
                raw_num_correct += int(torch.sum(preds == labels).item())
                probs = torch.nn.LogSoftmax(1)(logits)
                for i, file in enumerate(files):
                    labels_by_file[file] = labels[i].cpu()
                    preds_by_file[file].append(int(preds[i].cpu().item()))
                    probs_by_file[file] += probs[i].cpu()

            self.log_validation(
                preds_by_file,
                probs_by_file,
                labels_by_file,
                raw_num_correct,
                total_loss,
                save_preds
            )

    def train_batch(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor,
        log_frequency: int
    ) -> None:
        """Trains the model on a single batch of data.

        Args:
            spectrograms (torch.Tensor): A batch of spectrograms.
            labels (torch.Tensor): The corresponding labels for the
                spectrograms.
            log_frequency (int): The frequency with which to log training
                metrics.

        Returns:
            None
        """
        spectrograms = spectrograms.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(spectrograms)
        loss = self.calc_loss(logits, labels)
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
        if (self.step + 1) % log_frequency == 0:
            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = (labels == preds).sum().item() / len(labels) * 100
            self.summary_writer.add_scalars("loss", {"train": loss}, self.step)
            self.summary_writer.add_scalars(
                "accuracy", {"train": accuracy}, self.step)
        self.step += 1

    def train(
        self,
        epochs: int,
        save_preds: bool = False,
        log_frequency: int = 1,
        val_frequency: int = 1
    ) -> None:
        """Trains the model for the given number of epochs.

        Args:
            epochs (int): The number of epochs to train the model for.
            save_preds (bool, optional): If True, saves the model's
                predictions on validation data at the end of training.
                Defaults to False.
            log_frequency (int, optional): The frequency at which to log
                training progress. Defaults to 1 (log progress after every
                batch).
            val_frequency (int, optional): The frequency at which to evaluate
                the model on validation data. Defaults to 1 (evaluate after
                every epoch).

        Returns:
            None
        """
        self.model.train()
        for epoch in range(epochs):
            for _, spectrograms, labels, _ in self.train_loader:
                self.train_batch(spectrograms, labels, log_frequency)
            if (epoch + 1) % val_frequency == 0:
                self.validate()
                self.model.train()
            if save_preds and epoch == epochs - 1:
                self.validate(save_preds)
            self.summary_writer.add_scalar("epoch", epoch + 1, self.step)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    This function parses the command-line arguments for the shallow CNN
    training script. It uses the argparse module to define and parse the
    arguments, and returns the parsed arguments as a Namespace object.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=200,
        help="Number of epochs for training (default: 200)"
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "-vbs",
        "--val-batch-size",
        type=int,
        default=3750,
        help="Validation batch size (default: 3750)"
    )
    parser.add_argument(
        "-lf",
        "--log-frequency",
        type=int,
        default=100,
        help="Training log frequency (iterations, default: 100)"
    )
    parser.add_argument(
        "-vf",
        "--val-frequency",
        type=int,
        default=5,
        help="Validation log frequency (epochs, default: 5)"
    )
    parser.add_argument(
        "-bn",
        "--batch-norm",
        action='store_true',
        help="Use batch normalisation after convolution (default: False)"
    )
    parser.add_argument(
        "-s",
        "--save",
        action='store_true',
        help="Save the final predictions (default: False)"
    )
    return parser.parse_args()


def log_dir(args) -> str:
    """Create a unique log directory based on the command-line arguments.

    This function creates a unique log directory for the shallow CNN training
    script based on the command-line arguments provided by the user. The log
    directory will contain all the training logs generated during the training
    process.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The log directory path.
    """
    log_dir = f"logs/{datetime.now().strftime('%d-%H%M%S')}"
    log_dir += f"_bs{args.batch_size}"
    log_dir += "_bn" if args.batch_norm else ""
    return log_dir


def main() -> None:
    """Sets up and trains a shallow convolutional neural network.

    Returns:
        None
    """
    args = parse_args()
    model = ShallowCNN(batch_norm=args.batch_norm)
    train_loader = DataLoader(
        dataset=GTZAN("data/train.pkl"),
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=GTZAN("data/val.pkl"),
        shuffle=False,
        batch_size=args.val_batch_size,
        pin_memory=True
    )
    optimiser = torch.optim.Adam(
        params=model.parameters(),
        lr=0.00005,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    summary_writer = SummaryWriter(log_dir(args), flush_secs=5)
    trainer = Trainer(
        device=DEVICE,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimiser=optimiser,
        summary_writer=summary_writer
    )
    trainer.train(
        epochs=args.n_epochs,
        save_preds=args.save,
        log_frequency=args.log_frequency,
        val_frequency=args.val_frequency
    )
    summary_writer.close()


if __name__ == "__main__":
    main()
