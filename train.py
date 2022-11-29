import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from datetime import datetime

from dataset import GTZAN

import argparse

import pickle

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda")


class ShallowCNN(nn.Module):
    def __init__(self, batch_norm: bool = False):
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x_left = self.conv_left(images)
        if self.batch_norm:
            x_left = self.bn_conv_left(x_left)
        x_left = self.leaky_relu(x_left)
        x_left = self.pool_left(x_left)
        x_right = self.conv_right(images)
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
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight, 0.3)


class Trainer:
    def __init__(
            self,
            device: torch.device,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimiser: torch.optim.Optimizer,
            summary_writer: SummaryWriter
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.summary_writer = summary_writer
        self.step = 0

    def l1_penalty(self, penalty) -> torch.Tensor:
        params = self.model.named_parameters()
        weights = torch.cat([p.view(-1) for n, p in params if ".weight" in n])
        return weights.abs().sum() * penalty

    def calc_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(logits, labels) + self.l1_penalty(0.0001)

    def raw_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        preds = torch.argmax(logits, 1)
        accuracy = torch.mean((preds == labels).float()).item() * 100
        return accuracy

    def max_prob_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        labels = labels[::15]
        logits_grouped = logits.reshape(-1, 15, 10)
        results_summed = torch.sum(logits_grouped, 1)
        preds = torch.argmax(results_summed, 1)
        accuracy = torch.mean((preds == labels).float()).item() * 100
        return accuracy

    def maj_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        labels = labels[::15]
        logits_grouped = logits.reshape(-1, 15, 10)
        preds_grouped = torch.argmax(logits_grouped, 2)
        preds = torch.mode(preds_grouped).values
        accuracy = torch.mean((preds == labels).float()).item() * 100
        return accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0
        logits_all = torch.Tensor()
        labels_all = torch.Tensor()
        with torch.no_grad():
            for _, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.calc_loss(logits, labels)
                total_loss += loss.item()
                logits_all = torch.cat((logits_all, logits.cpu()))
                labels_all = torch.cat([labels_all, labels.cpu()])
            mean_loss = total_loss / len(self.val_loader)
            raw_accuracy = self.raw_accuracy(logits_all, labels_all)
            max_prob_accuracy = self.max_prob_accuracy(logits_all, labels_all)
            maj_accuracy = self.maj_accuracy(logits_all, labels_all)
        self.summary_writer.add_scalars(
            "accuracy", {"raw_val": raw_accuracy}, self.step)
        self.summary_writer.add_scalars(
            "accuracy", {"max_prob_val": max_prob_accuracy}, self.step)
        self.summary_writer.add_scalars(
            "accuracy", {"maj_val": maj_accuracy}, self.step)
        self.summary_writer.add_scalars(
            "loss", {"val": mean_loss}, self.step
        )

    def train_batch(
        self, batch: torch.Tensor,
        labels: torch.Tensor,
        log_frequency: int
    ):
        batch = batch.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(batch)
        loss = self.calc_loss(logits, labels)
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
        if (self.step + 1) % log_frequency == 0:
            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = (labels == preds).sum().item() / len(labels) * 100
            self.summary_writer.add_scalars(
                "loss", {"train": loss}, self.step)
            self.summary_writer.add_scalars(
                "accuracy", {"train": accuracy}, self.step)
        self.step += 1

    def save_final_preds(self):
        self.model.eval()
        labels_all = torch.Tensor()
        logits_all = torch.Tensor()
        preds_all = torch.Tensor()
        names_all = []
        with torch.no_grad():
            for names, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                preds = logits.argmax(-1)
                labels_all = torch.cat((labels_all, labels.cpu()))
                logits_all = torch.cat((logits_all, logits.cpu()))
                preds_all = torch.cat((preds_all, preds.cpu()))
                names_all += names
            logits_grouped = logits_all.reshape(-1, 15, 10)
            results_summed = torch.sum(logits_grouped, 1)
            preds_grouped = torch.argmax(results_summed, 1)
            with open("preds_raw.pkl", "wb") as file:
                pickle.dump((labels_all, preds_all, names_all), file)
            with open("preds_max.pkl", "wb") as file:
                pickle.dump((labels_all[::15], preds_grouped, names_all[::15]), file)

    def train(
        self,
        epochs: int,
        save: bool = False,
        log_frequency: int = 1,
        val_frequency: int = 1
    ):
        self.model.train()
        for epoch in range(epochs):
            for _, batch, labels, _ in self.train_loader:
                self.train_batch(batch, labels, log_frequency)
            if (epoch + 1) % val_frequency == 0:
                self.validate()
                self.model.train()
            if save and epoch == epochs - 1:
                self.save_final_preds()
            self.summary_writer.add_scalar("epoch", epoch + 1, self.step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=200,
        help="Number of epochs"
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "-lf",
        "--log-frequency",
        type=int,
        default=100,
        help="Training log frequency (iterations)"
    )
    parser.add_argument(
        "-vf",
        "--val-frequency",
        type=int,
        default=5,
        help="Validation log frequency (epochs)"
    )
    parser.add_argument(
        "-bn",
        "--batch-norm",
        action='store_true',
        help="Batch normalisation after convolutional layers"
    )
    parser.add_argument(
        "-s",
        "--save",
        action='store_true',
        help="Save the final predictions"
    )
    return parser.parse_args()


def log_dir(args):
    log_dir = f"logs/{datetime.now().strftime('%d-%H%M%S')}"
    log_dir += f"_bs{args.batch_size}"
    log_dir += "_bn" if args.batch_norm else ""
    return log_dir


def main():
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
        batch_size=3750,
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
        save=args.save,
        log_frequency=args.log_frequency,
        val_frequency=args.val_frequency
    )
    summary_writer.close()


if __name__ == "__main__":
    main()
