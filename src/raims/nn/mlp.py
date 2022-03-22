from typing import List, Any

import torch
import torch.nn.functional
from pytorch_lightning import LightningModule


class MultilayerPerceptron(LightningModule):
    def __init__(self, max_mz: int, hidden_layers: List[int], learning_rate: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.in_features = max_mz
        self.out_features = max_mz
        self.learning_rate = learning_rate

        layers = [torch.nn.Linear(self.in_features, hidden_layers[0])]

        for n, m in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(n, m))

        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_layers[-1], self.out_features))
        layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y, reduction='mean')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
