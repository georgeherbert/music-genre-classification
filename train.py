import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from datetime import datetime

from dataset import GTZAN
from evaluation import evaluate

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda")


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            torch.nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_normal_(layer.weight)


class ShallowCNN(CNN):
    def __init__(self):
        super().__init__()
        self.conv_left = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding="same"
        )
        self.conv_right = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 20),
            padding="same"
        )
        self.pool_left = torch.nn.MaxPool2d(
            kernel_size=(1, 20),
            stride=(1, 20)
        )
        self.pool_right = torch.nn.MaxPool2d(
            kernel_size=(20, 1),
            stride=(20, 1)
        )
        self.leaky_relu = torch.nn.LeakyReLU(0.3)
        self.fc_1 = torch.nn.Linear(10240, 200)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc_2 = torch.nn.Linear(200, 10)
        self.initialise_layer(self.conv_left)
        self.initialise_layer(self.conv_right)
        self.initialise_layer(self.fc_1)
        self.initialise_layer(self.fc_2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x_left = self.leaky_relu(self.conv_left((images)))
        x_left = self.pool_left(x_left)
        x_right = self.leaky_relu(self.conv_right((images)))
        x_right = self.pool_right(x_right)
        x = torch.cat(
            [
                torch.flatten(x_left, start_dim=1),
                torch.flatten(x_right, start_dim=1)
            ],
            dim=1
        )
        x = self.leaky_relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


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
                logits = self.model.forward(batch)
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
        logits = self.model.forward(batch)
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

    def train(
        self,
        epochs: int,
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


if __name__ == "__main__":
    model = ShallowCNN()
    train_loader = DataLoader(
        dataset=GTZAN("data/augment.pkl"),
        shuffle=True,
        batch_size=64,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=GTZAN("data/val.pkl"),
        shuffle=False,
        batch_size=64,
        pin_memory=True
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        params=model.parameters(),
        lr=0.00005,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    summary_writer = SummaryWriter(
        log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
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
        epochs=200,
    )
    summary_writer.close()
