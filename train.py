import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from datetime import datetime

from dataset import GTZAN
from evaluation import evaluate

import argparse

import pickle

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda")


class ShallowCNN(torch.nn.Module):
    def __init__(self, batch_norm: bool = False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv_left = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding="same"
        )
        self.bn_conv_left = torch.nn.BatchNorm2d(16)
        self.pool_left = torch.nn.MaxPool2d(
            kernel_size=(1, 20),
            stride=(1, 20)
        )

        self.conv_right = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 20),
            padding="same"
        )
        self.bn_conv_right = torch.nn.BatchNorm2d(16)
        self.pool_right = torch.nn.MaxPool2d(
            kernel_size=(20, 1),
            stride=(20, 1)
        )

        self.fc_1 = torch.nn.Linear(10240, 200)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc_2 = torch.nn.Linear(200, 10)

        self.leaky_relu = torch.nn.LeakyReLU(0.3)

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
        x = torch.cat(
            [
                torch.flatten(x_left, start_dim=1),
                torch.flatten(x_right, start_dim=1)
            ],
            dim=1
        )
        x = self.fc_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            torch.nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_uniform_(
                layer.weight,
                a=0.3
            )


class Trainer:
    def __init__(
            self,
            device: torch.device,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: torch.nn.Module,
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

    def l1_penalty(self, penalty: float = 0.0001) -> torch.Tensor:
        params = self.model.named_parameters()
        weights = torch.cat([p.view(-1) for n, p in params if ".weight" in n])
        return weights.abs().sum() * penalty

    def calc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(logits, labels) + self.l1_penalty()

    def log_curves(
        self,
        type: str,
        loss: float,
        accuracy: float
    ):
        self.summary_writer.add_scalars(
            "loss",
            {type: loss},
            self.step
        )
        self.summary_writer.add_scalars(
            "accuracy",
            {type: accuracy},
            self.step
        )

    def validate(self):
        self.model.eval()
        results = torch.Tensor()
        total_loss = 0
        with torch.no_grad():
            for _, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.calc_loss(logits, labels)
                total_loss += loss.item()
                results = torch.cat((results, logits.cpu()), 0)
        mean_loss = total_loss / len(self.val_loader)
        accuracy = evaluate(results, "data/val.pkl")
        self.log_curves("val", mean_loss, float(accuracy))

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
            self.log_curves("train", loss.item(), accuracy)
        self.step += 1

    def save_final_preds(self):
        self.model.eval()
        labels_all = torch.Tensor()
        preds_all = torch.Tensor()
        with torch.no_grad():
            for _, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                preds = logits.argmax(-1)
                labels_all = torch.cat((labels_all, labels.cpu()))
                preds_all = torch.cat((preds_all, preds.cpu()))
            with open("preds.pkl", "wb") as file:
                pickle.dump((labels_all, preds_all), file)

    def train(
        self,
        epochs: int,
        save: bool = False,
        log_frequency: int = 1,
        val_frequency: int = 1
    ):
        self.model.train()
        for epoch in range(epochs):
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            for _, batch, labels, _ in self.train_loader:
                self.train_batch(batch, labels, log_frequency)
            if (epoch + 1) % val_frequency == 0:
                self.validate()
                self.model.train()
            if save and epoch == epochs - 1:
                self.save_final_preds()


def parse_arguments() -> argparse.Namespace:
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
        "-a",
        "--augment",
        action='store_true',
        help="Train with the augmented dataset"
    )
    parser.add_argument(
        "-s",
        "--save",
        action='store_true',
        help="Save the final predictions"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    model = ShallowCNN(
        batch_norm=args.batch_norm
    )
    train_loader = DataLoader(
        dataset=GTZAN(
            f"data/{'augment' if args.augment else 'train'}.pkl"
        ),
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
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        params=model.parameters(),
        lr=0.00005,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    log_dir = f"logs/{datetime.now().strftime('%d-%H%M%S')}_bs{args.batch_size}"
    if args.batch_norm:
        log_dir += "_bn"
    if args.augment:
        log_dir += "_a"
    summary_writer = SummaryWriter(
        log_dir=log_dir,
        flush_secs=5
    )
    trainer = Trainer(
        device=DEVICE,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
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
