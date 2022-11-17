import torch

from dataset import GTZAN

DEVICE = torch.device("cuda")

class CNN(torch.nn.Module):
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

        x = torch.cat([torch.flatten(x_left, start_dim = 1), torch.flatten(x_right, start_dim = 1)], dim = 1)
        x = torch.nn.LeakyReLU(self.fc_1(x))
        x = self.dropout(x)

        x = self.fc_2(x)

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            torch.nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            optimiser: torch.optim.Adam
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.step = 0

    def train(self):
        self.model.train()
        for epoch in range(200):
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = float((labels == preds).sum()) / len(labels)
                    print(epoch, accuracy)


def main():
    train_loader = torch.utils.data.DataLoader(
        GTZAN("data/train.pkl"),
        shuffle=True,
        batch_size=256,
    )
    val_loader = torch.utils.data.DataLoader(
        GTZAN("data/val.pkl"),
        shuffle=False,
        batch_size=256,
    )

    model = CNN()
    criterion = torch.nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=0.00005,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    trainer = Trainer(model, train_loader, val_loader, criterion, optimiser)

    trainer.train()

if __name__ == "__main__":
    main()
